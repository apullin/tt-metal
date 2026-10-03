# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5/3.6 multi-token-prediction (MTP) head on the TP path.

Checkpoint tensors `mtp.*` (the HF text model the target loads through never reads them):
fc [dim, 2*dim], pre_fc_norm_embedding, pre_fc_norm_hidden, norm, and one full-attention decoder
layer `mtp.layers.0.*` shaped like the target's. It shares the target's embedding and LM head.

Forward for a row at position i (llama.cpp qwen35.cpp graph_mtp; vLLM qwen3_next_mtp.py):
    e = norm_emb(embed(x_{i+1}));  h = norm_hid(h_i)      h_i: target hidden after the final norm
    x = fc(cat([e, h]))                                    column-parallel -> fractured residual
    x = decoder_layer(x)                                   own paged KV, entry written at position i
    logits = lm_head(norm(x))                              predicts x_{i+2}

All norms are zero-centred (x * (1 + w)); the +1 is added at load like the target's norms. The
decoder layer is a Qwen36DecoderLayer (same TP attention + MLP + norms as a target full-attention
layer) built from the mtp.layers.0 weights under a separate tensor-cache directory.
"""
import json
from pathlib import Path

import torch

import ttnn
from models.common.rmsnorm import RMSNorm
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.layer import Qwen36DecoderLayer
from models.tt_transformers.tt.common import Mode, get_block_size, num_blocks_in_seq
from models.tt_transformers.tt.distributed_norm import DistributedNorm

MTP_PREFIX = "mtp."
MTP_LAYER_PREFIX = "mtp.layers.0."


def load_mtp_state_dict(checkpoint_dir):
    """Read the `mtp.*` tensors straight from the checkpoint's safetensors shards."""
    from safetensors import safe_open

    checkpoint_dir = Path(checkpoint_dir)
    weight_map = json.loads((checkpoint_dir / "model.safetensors.index.json").read_text())["weight_map"]
    by_file = {}
    for key, filename in weight_map.items():
        if key.startswith(MTP_PREFIX):
            by_file.setdefault(filename, []).append(key)
    assert by_file, f"no {MTP_PREFIX}* tensors in {checkpoint_dir}"
    state_dict = {}
    for filename, keys in by_file.items():
        with safe_open(str(checkpoint_dir / filename), framework="pt") as f:
            for key in keys:
                state_dict[key] = f.get_tensor(key)
    return state_dict


class MTPHead:
    def __init__(self, model, state_dict, tensor_cache_path):
        args = model.args
        mesh = model.mesh_device
        assert model.num_devices > 1, "the MTP head is implemented for the TP path"
        self.model = model
        self.args = args
        self.mesh = mesh
        cache = Path(tensor_cache_path) / "mtp" if tensor_cache_path else None
        if cache is not None:
            cache.mkdir(parents=True, exist_ok=True)

        # The decoder layer reuses the target's full-attention layer class; any full-attention
        # checkpoint index gives it the right type, and the separate cache dir keeps its files apart.
        layer_num = max(i for i, t in enumerate(args.attention_type_list) if t == "full_attention")
        layer_sd = {
            f"layers.{layer_num}.{k[len(MTP_LAYER_PREFIX):]}": v
            for k, v in state_dict.items()
            if k.startswith(MTP_LAYER_PREFIX)
        }
        self.layer = Qwen36DecoderLayer(mesh, args, layer_sd, layer_num, cache, tt_ccl=model.tt_ccl)
        assert self.layer.is_full_attention

        def rms(key):
            # Built like the target's final norm (Qwen36Model.norm): TP-aware RMSNorm + gather wrapper.
            norm = RMSNorm(
                device=mesh,
                dim=args.dim,
                state_dict=state_dict,
                weight_key=key,
                state_dict_prefix=MTP_PREFIX,
                weight_cache_path=cache,
                weight_dtype=ttnn.bfloat16,
                add_unit_offset=True,
                eps=args.norm_eps,
                is_distributed=args.is_distributed_norm,
                ccl_topology=args.ccl_topology(),
                tt_ccl=model.tt_ccl,
            )
            return DistributedNorm(norm, args, tt_ccl=model.tt_ccl, TG=args.is_galaxy)

        # The norms see a fractured embedding, the replicated target hidden, and the fractured
        # residual. A full-width interleaved rms_norm over dim=5120 overflows Wormhole L1, so the
        # hidden norm runs sharded in decode (the final-norm config) and distributed over its mesh
        # partition in prefill.
        self.norm_emb = rms("pre_fc_norm_embedding")
        self.norm_hid = rms("pre_fc_norm_hidden")
        self.norm = rms("norm")
        # fc [dim, 2*dim] column-parallel: device d computes residual slice d, so its output is the
        # fractured residual the decoder layer expects (no collective).
        self.fc = tpc.shard_w(
            state_dict[MTP_PREFIX + "fc.weight"],
            mesh,
            dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_path=str(cache / "fc.weight") if cache is not None else None,
            dtype=ttnn.bfloat8_b,
        )
        self.kv_cache = None

    def allocate_kv_cache(self, kv_cache_shape, dtype=ttnn.bfloat16):
        assert self.kv_cache is None

        def mk():
            return ttnn.as_tensor(
                torch.zeros(kv_cache_shape, dtype=torch.bfloat16),
                device=self.mesh,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )

        self.kv_cache = (mk(), mk())
        self.layer.attention.set_paged_kv_cache(*self.kv_cache)

    def free_kv_cache(self):
        if self.kv_cache is None:
            return
        for t in self.kv_cache:
            ttnn.deallocate(t)
        self.kv_cache = None
        self.layer.attention.paged_k = self.layer.attention.paged_v = None
        self.layer.attention.use_paged = False

    def _decode_norm_config(self):
        # The target's final-norm config (_final_norm_decode): sharded multi-core norm, DRAM output.
        nc = dict(self.args.get_norm_config("lm_head", Mode.DECODE))
        nc["output_mem_config"] = ttnn.DRAM_MEMORY_CONFIG
        return nc

    def _fc(self, e_n, h_n):
        cat = ttnn.concat([e_n, h_n], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(e_n)
        ttnn.deallocate(h_n)
        x = ttnn.linear(cat, self.fc, compute_kernel_config=tpc.COMPUTE_HIFI2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(cat)
        return x

    def decode(self, tokens, hidden, cos, sin, cur_pos, page_table, kv_update_idxs):
        """MTP rows at positions cur_pos (traceable; same input conventions as TargetVerifier.forward).

        tokens [rows, 1]: x_{i+1} per row; hidden [1, 1, rows, dim]: the target's post-norm h_i.
        Returns vocab-sharded logits [1, 1, rows, vocab/devices]."""
        model = self.model
        e = model.embd(tokens)
        e = ttnn.reshape(e, (1, 1, e.shape[0] * e.shape[1], e.shape[-1]))
        nc = self._decode_norm_config()
        e_n = self.norm_emb(e, mode=Mode.DECODE, norm_config=nc)
        ttnn.deallocate(e)
        h_sh = ttnn.to_memory_config(hidden, nc["sharded_output_config"])
        h_n = self.norm_hid.norm(h_sh, mode=Mode.DECODE, in_sharded=True, out_sharded=True, norm_config=nc)
        ttnn.deallocate(h_sh)
        x = self._fc(e_n, h_n)
        attention = self.layer.attention
        attention.kv_update_idxs = kv_update_idxs
        try:
            y = self.layer.forward(x, cos, sin, mode="decode", position_tensor=cur_pos, page_table=page_table)
        finally:
            attention.kv_update_idxs = None
        ttnn.deallocate(x)
        y_n = self.norm(y, mode=Mode.DECODE, norm_config=nc)
        ttnn.deallocate(y)
        logits = ttnn.linear(y_n, model.lm_head_weight)
        ttnn.deallocate(y_n)
        return logits

    def prefill(self, next_tokens, hidden, valid_len, page_table):
        """MTP prompt pass at positions 0..valid_len-1 (eager, one masked bucket).

        next_tokens [1, valid_len] torch: x_1..x_T (the prompt shifted by one, ending with the
        target's first generated token); hidden [1, 1, bucket, dim]: the target's post-norm hidden
        of every prompt row (Qwen36Model.prefill_hidden); page_table [1, blocks] torch.
        Fills the MTP KV for positions < valid_len and returns replicated logits [1, 1, vocab] for
        the last row (the draft for position valid_len + 1)."""
        model = self.model
        mesh = self.mesh
        rep = ttnn.ReplicateTensorToMesh(mesh)
        bucket = hidden.shape[-2]
        assert next_tokens.shape[1] == valid_len <= bucket
        token_buf = torch.zeros(1, bucket, dtype=torch.int32)
        token_buf[:, :valid_len] = next_tokens.to(torch.int32)
        tok = ttnn.from_torch(token_buf, dtype=ttnn.uint32, device=mesh, mesh_mapper=rep)
        e = model.embd(tok)
        ttnn.deallocate(tok)
        e = ttnn.reshape(e, (1, 1, bucket, e.shape[-1]))
        e = ttnn.to_memory_config(e, ttnn.DRAM_MEMORY_CONFIG)
        e_n = self.norm_emb(e, mode=Mode.PREFILL)
        ttnn.deallocate(e)
        h_frac = ttnn.mesh_partition(hidden, dim=3, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        h_n = self.norm_hid(h_frac, mode=Mode.PREFILL)
        ttnn.deallocate(h_frac)
        x = self._fc(e_n, h_n)

        # Same per-chunk attention inputs as Qwen36Model._forward_prefill_chunk_masked_tp.
        block_size = get_block_size(model._paged_kv_caches)
        cos_t, sin_t = model._rope_tp_cos_sin_torch(0, bucket)
        cos = ttnn.from_torch(cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        sin = ttnn.from_torch(sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        full_pt = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
        chunk_pt = ttnn.from_torch(
            page_table[:, : num_blocks_in_seq(valid_len, block_size)].contiguous(),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
        )
        csi = ttnn.from_torch(
            torch.tensor([0], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh
        )
        y = self.layer.forward(
            x,
            cos=cos,
            sin=sin,
            mode="prefill",
            page_table=full_pt,
            chunk_page_table=chunk_pt,
            chunk_start_idx=0,
            chunk_start_idx_tensor=csi,
            valid_len=valid_len,
        )
        for t in (x, cos, sin, full_pt, chunk_pt, csi):
            ttnn.deallocate(t)

        sel = torch.zeros(1, 1, 1, bucket, dtype=torch.float32)
        sel[0, 0, 0, valid_len - 1] = 1.0
        sel_tt = ttnn.from_torch(sel, dtype=y.dtype, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        y_last = ttnn.matmul(sel_tt, y)
        ttnn.deallocate(sel_tt)
        ttnn.deallocate(y)
        y_last = ttnn.to_memory_config(y_last, ttnn.DRAM_MEMORY_CONFIG)
        y_n = self.norm(y_last, mode=Mode.PREFILL)
        ttnn.deallocate(y_last)
        logits = model._lm_head(y_n)
        ttnn.deallocate(y_n)
        return ttnn.reshape(logits, (1, 1, logits.shape[-1]))
