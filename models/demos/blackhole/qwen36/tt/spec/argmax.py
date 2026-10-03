# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Greedy argmax over vocab-sharded logits, one result per decode row.

The LM head is vocab-sharded across the mesh, so each device reduces its own shard to a per-row
(local index, max value) pair on device (traceable, tiny readback), and the host picks the winning
shard per row. For one row this is the demo's `_argmax_dev` / `_read_tok` (text_demo.py) op for op.
"""
import torch

import ttnn

# reduce(dim=-1) parallelizes over tile rows, so each row's vocab shard is reshaped into a tall
# [R, 32] grid before the max (as text_demo._maxval_dev does for one row).
_MAXVAL_C = 32


def _maxval_rows(r):
    return (((r + _MAXVAL_C - 1) // _MAXVAL_C) + 31) // 32 * 32


def sharded_argmax_rows(logits, rows, per_shard):
    """logits: [1, 1, rows, per_shard] on every device (its vocab shard). Returns device tensors
    (idx [1, 1, rows] uint32 local index, val [1, 1, rows] max logit) per device."""
    logits_rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
    idx = ttnn.argmax(logits_rm, dim=-1, keepdim=False)
    ttnn.deallocate(logits_rm)
    r = _maxval_rows(per_shard)
    padded = ttnn.pad(logits, [(0, 0), (0, 0), (0, 0), (0, r * _MAXVAL_C - per_shard)], value=-1e30)
    grid = ttnn.reshape(padded, (1, 1, rows * r, _MAXVAL_C))
    part = ttnn.max(grid, dim=-1)
    part_rows = ttnn.reshape(part, (1, 1, rows, r))
    val = ttnn.max(part_rows, dim=-1)
    for t in (padded, grid, part, part_rows):
        ttnn.deallocate(t)
    return idx, val


def combine_argmax_host(idx, val, mesh_device, rows, per_shard):
    """Global greedy token per row: the shard with the largest max wins (first shard on ties);
    token = shard * per_shard + local index."""
    comp = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    idxs = ttnn.to_torch(idx, mesh_composer=comp).reshape(-1, rows)
    vals = ttnn.to_torch(val, mesh_composer=comp).float().reshape(-1, rows)
    out = []
    for row in range(rows):
        d = int(torch.argmax(vals[:, row]).item())
        out.append(d * per_shard + int(idxs[d, row].item()))
    return out
