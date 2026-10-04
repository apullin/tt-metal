# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in: move the target's token embedding table to the host (memory enabler for long windows).

The TP embedding table is vocab x dim/tp bf16 per device (248320 x 2560 x 2 B = 1.21 GiB on each
N300 ASIC). Every caller knows its tokens on the host, so the lookup moves there and only the
[rows, dim/tp] result crosses PCIe. The host table is read back from the device table, so the
embeddings are bit-identical to the device lookup.

After `enable_host_embedding(model)` the device table is freed and `model.embd` raises if any
device-side lookup is attempted. Every TP path of Qwen36Model takes embedded rows instead
(`Qwen36Model.host_embedding`): the masked-bucket prefill, the chunk-prefill trace and decode
(`prepare_decode_inputs_host` returns the embedded rows), so the plain `_run_tp_generation`, the
MTP head and the speculative verify and drafter all run unchanged on top of it. Call it before the
chunk-prefill trace is captured (`_run_tp_generation` captures per call).
"""
import torch

import ttnn


class HostEmbedding:
    def __init__(self, table, mesh_device):
        self.table = table  # [vocab, dim] bf16 (host)
        self.mesh = mesh_device
        self.mapper = ttnn.ShardTensorToMesh(mesh_device, dim=-1)

    def host(self, tokens):
        """Host multi-device tensor [1, 1, N, dim/tp] per device (fractured like the device lookup)."""
        idx = torch.as_tensor(tokens, dtype=torch.long).reshape(-1)
        rows = self.table[idx][None, None].contiguous()
        return ttnn.from_torch(rows, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=self.mapper)

    def device(self, tokens, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        idx = torch.as_tensor(tokens, dtype=torch.long).reshape(-1)
        rows = self.table[idx][None, None].contiguous()
        return ttnn.from_torch(
            rows, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh, mesh_mapper=self.mapper, memory_config=memory_config
        )


class _NoDeviceEmbedding:
    def __call__(self, *args, **kwargs):
        raise RuntimeError("the device embedding table was moved to the host (enable_host_embedding)")


def enable_host_embedding(model):
    """Read the device table back, free it, and route the TP prefill through the host table."""
    if model.host_embedding is not None:
        return model.host_embedding
    w = model.embd.weights
    table = ttnn.to_torch(w, mesh_composer=ttnn.ConcatMeshToTensor(model.mesh_device, dim=3))
    table = table.reshape(-1, table.shape[-1])[: model.args.vocab_size].contiguous()
    assert table.shape == (model.args.vocab_size, model.args.dim), table.shape
    ttnn.deallocate(w)
    model.embd = _NoDeviceEmbedding()
    model.host_embedding = HostEmbedding(table, model.mesh_device)
    return model.host_embedding
