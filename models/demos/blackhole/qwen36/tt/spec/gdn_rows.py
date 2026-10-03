# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gated DeltaNet over R rows of ONE sequence (speculative verify), with per-row state snapshots.

`TPGatedDeltaNet.forward_decode` treats decode rows as independent users, each with its own state
row. A verify step instead feeds R consecutive tokens of the same sequence, so the recurrence must
run row by row on the single sequence state. Everything that does not read the recurrent state
runs once over all rows:
  - the in- and out-projections, the gated norm and the output gate;
  - the conv1d shift register: row r's window is rows r..r+K-1 of
    S = [conv_states[1..K-1]; qkv rows 0..R-1], so the weighted sum of K row-shifted slices of S
    gives every row's conv output, and the conv state after row r is S[r..r+K-1]. S holds one
    row per tile (shape [K-1+R, 1, D]): slices along dim 0 are whole tiles, and cost a fraction
    of row slices inside a tile;
  - q/k/v split, GQA expand, beta, g, and the recurrence's own input prep (fp32 casts, q/k L2
    norms, q scale, reshapes).
Per row only the state chain runs (recurrent_gated_delta_rule_decode_ttnn's state ops on that
row's slices): decay, read, delta, rank-1 write, query. Every elementwise op and norm acts per row,
so row r computes exactly the values a B=1 `forward_decode` computes for that token.

Snapshots: the state after row r < R-1 is kept for rollback. The recurrent state goes to snapshot
buffer r (the copy a plain step makes into the layer state); the conv state needs no copy, as S
(kept on device as the verify's output) holds every row's conv state. Rolling back to "after row
j" is one copy per buffer into the layer's fixed-address state (trace-safe).
"""
import os

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.gdn.tp import _silu_mul, _softplus_add
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    _recurrent_read_query_program_config,
    l2_norm_ttnn,
)
from models.tt_transformers.tt.ccl import tt_all_reduce

_L1 = ttnn.L1_MEMORY_CONFIG
# recurrent_gated_delta_rule_decode_ttnn's matmul configs (read/query, and the rank-1 write).
_READ_QUERY_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)
_WRITE_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
)


class GDNRowState:
    """Rollback state for one TP GDN layer (B=1): recurrent-state snapshots after rows 0..rows-2,
    and the verify's conv window S (set by gdn_decode_rows)."""

    def __init__(self, dn, rows):
        assert dn.B == 1, "row-wise verify runs on a single-sequence (B=1) GDN state"
        assert dn.conv_states is not None and dn.rec_state is not None, "allocate the GDN state first"
        self.dn = dn
        self.rows = rows
        self.rec = [
            ttnn.from_torch(
                torch.zeros(tuple(dn.rec_state.shape), dtype=torch.float32 if dn.rec_state.dtype == ttnn.float32 else torch.bfloat16),
                dtype=dn.rec_state.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=dn.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(dn.mesh),
            )
            for _ in range(rows - 1)
        ]
        self.window = None

    def set_window(self, window):
        if self.window is not None:
            ttnn.deallocate(self.window)
        self.window = window

    def restore(self, row):
        """Roll the layer state back to the snapshot after `row` (device copies, no persistent allocation)."""
        dn = self.dn
        ttnn.copy(self.rec[row], dn.rec_state)
        _write_conv_state(dn, self.window, row)

    def free(self):
        for t in self.rec:
            ttnn.deallocate(t)
        self.rec = []
        self.set_window(None)


def _write_conv_state(dn, window, row):
    """conv_states[j] = S[row + j]: the conv shift register after verify row `row`."""
    for j in range(dn.K):
        ttnn.slice(window, (row + j, 0, 0), (row + j + 1, 1, dn.qkv_dim_tp), output_tensor=dn.conv_states[j])


def _row(t, r):
    """Row r of a tensor batched over dim 0."""
    shape = tuple(t.shape)
    return ttnn.slice(t, (r,) + (0,) * (len(shape) - 1), (r + 1,) + shape[1:], memory_config=_L1)


def gdn_decode_rows(dn, x, state):
    """Verify-step GDN over x's rows (consecutive tokens of one sequence).

    x: [1, 1, R, dim] replicated (gathered norm output). Returns [1, 1, R, dim] fractured, like
    `forward_decode`. Leaves the state after row R-1 in dn and after rows 0..R-2 in `state`."""
    tw, Nk, Nv, Dk, Dv, K = dn.tw, dn.Nk, dn.Nv, dn.Dk, dn.Dv, dn.K
    if len(x.shape) == 4:
        x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))
    R = x.shape[-2]
    assert R == state.rows, f"verify rows {R} != snapshot rows {state.rows}"
    D = dn.qkv_dim_tp

    qkv, z, a, b = dn._project_qkvzab(x, R, out_mc=_L1)

    # Conv1d over all rows: S = [conv_states[1..K-1]; qkv], row r's window = S[r..r+K-1].
    qkv_rows = ttnn.reshape(qkv, (R, 1, D), memory_config=_L1) if R > 1 else qkv
    window = ttnn.concat([*dn.conv_states[1:], qkv_rows], dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(qkv_rows)
    if R > 1:
        ttnn.deallocate(qkv)
    taps = [ttnn.slice(window, (j, 0, 0), (j + R, 1, D), memory_config=_L1) for j in range(K)]
    conv = ttnn.multiply(taps[0], tw["conv_taps"][0], memory_config=_L1)
    for j in range(1, K):
        conv = ttnn.mac(taps[j], tw["conv_taps"][j], conv)
    for t in taps:
        ttnn.deallocate(t)
    conv = ttnn.silu(conv, memory_config=_L1)
    _write_conv_state(dn, window, R - 1)
    state.set_window(window)

    kd = dn.key_dim_tp
    q = ttnn.reshape(ttnn.slice(conv, (0, 0, 0), (R, 1, kd)), (R, Nk, Dk))
    k = ttnn.reshape(ttnn.slice(conv, (0, 0, kd), (R, 1, 2 * kd)), (R, Nk, Dk))
    v = ttnn.reshape(ttnn.slice(conv, (0, 0, 2 * kd), (R, 1, D)), (R, Nv, Dv))
    ttnn.deallocate(conv)
    rf = Nv // Nk
    q = ttnn.repeat_interleave(q, rf, dim=1)
    k = ttnn.repeat_interleave(k, rf, dim=1)
    q = ttnn.reshape(q, (R, 1, Nv, Dk), memory_config=_L1)
    k = ttnn.reshape(k, (R, 1, Nv, Dk), memory_config=_L1)
    v = ttnn.reshape(v, (R, 1, Nv, Dv), memory_config=_L1)
    beta = ttnn.reshape(ttnn.sigmoid(b, memory_config=_L1), (R, 1, Nv))
    ttnn.deallocate(b)
    g = ttnn.multiply(tw["neg_exp_A"], _softplus_add(a, tw["dt_bias"]), memory_config=_L1)
    ttnn.deallocate(a)
    g = ttnn.reshape(g, (R, 1, Nv))

    # recurrent_gated_delta_rule_decode_ttnn's input prep, over all rows (fp32 step by default,
    # QWEN35_GDN_DECODE_BF16=1 reverts, as in forward_decode).
    high_precision = os.environ.get("QWEN35_GDN_DECODE_BF16") != "1"
    if high_precision:
        q, k, v, beta, g = (ttnn.typecast(t, ttnn.float32) for t in (q, k, v, beta, g))
    q = l2_norm_ttnn(q, dim=-1)
    k = l2_norm_ttnn(k, dim=-1)
    q = ttnn.multiply(q, dn.scale, memory_config=_L1)
    q_row = ttnn.reshape(q, [R, Nv, 1, Dk], memory_config=_L1)
    k_row = ttnn.reshape(k, [R, Nv, 1, Dk], memory_config=_L1)
    k_col = ttnn.reshape(ttnn.reshape(k_row, [R, Nv, Dk], memory_config=_L1), [R, Nv, Dk, 1], memory_config=_L1)
    v_row = ttnn.reshape(v, [R, Nv, 1, Dv], memory_config=_L1)
    g_bhkv = ttnn.reshape(ttnn.reshape(g, [R, Nv], memory_config=_L1), [R, Nv, 1, 1], memory_config=_L1)
    beta_bhkv = ttnn.reshape(ttnn.reshape(beta, [R, Nv], memory_config=_L1), [R, Nv, 1, 1], memory_config=_L1)
    read_query_pc = _recurrent_read_query_program_config(dn.mesh, Dk, Dv)
    # The rank-1 write k ⊗ delta has a single K tile, so every output tile is one tile product and
    # the core split cannot change the result. Half a head's output tiles per core replays in 32 us,
    # against 51 us for the auto-selected config.
    grid = dn.mesh.compute_with_storage_grid_size()
    write_pc = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=Dv // 32,
        per_core_M=Dk // 64,
        per_core_N=Dv // 32,
    )

    # The state chain, row by row; h stays in L1 from row to row.
    h = dn.rec_state
    if high_precision and h.dtype != ttnn.float32:
        h = ttnn.typecast(h, ttnn.float32)
    h = ttnn.to_memory_config(h, _L1)
    outs = []
    for r in range(R):
        g_r, k_r, kc_r, v_r, beta_r, q_r = (_row(t, r) for t in (g_bhkv, k_row, k_col, v_row, beta_bhkv, q_row))
        h_new = ttnn.multiply(h, g_r, input_tensor_b_activations=[ttnn.UnaryOpType.EXP], memory_config=_L1)
        ttnn.deallocate(h)
        v_read = ttnn.matmul(k_r, h_new, memory_config=_L1, program_config=read_query_pc, compute_kernel_config=_READ_QUERY_CFG)
        delta = ttnn.subtract(v_r, v_read, memory_config=_L1)
        outer = ttnn.matmul(kc_r, delta, memory_config=_L1, program_config=write_pc, compute_kernel_config=_WRITE_CFG)
        outer = ttnn.multiply(outer, beta_r, memory_config=_L1)
        h = ttnn.add(h_new, outer, memory_config=_L1)
        for t in (h_new, v_read, delta, outer, g_r, k_r, kc_r, v_r, beta_r):
            ttnn.deallocate(t)
        outs.append(ttnn.matmul(q_r, h, memory_config=_L1, program_config=read_query_pc, compute_kernel_config=_READ_QUERY_CFG))
        ttnn.deallocate(q_r)
        ttnn.copy(h, dn.rec_state if r == R - 1 else state.rec[r])
    ttnn.deallocate(h)
    for t in (q, k, v, beta, g, q_row, k_row, k_col, v_row, g_bhkv, beta_bhkv):
        ttnn.deallocate(t)

    o_all = outs[0] if R == 1 else ttnn.concat(outs, dim=0, memory_config=_L1)
    if R > 1:
        for o in outs:
            ttnn.deallocate(o)
    out_r = ttnn.reshape(o_all, (R, Nv, Dv))
    out_n = ttnn.rms_norm(out_r, weight=tw["norm_w"], epsilon=1e-6, memory_config=_L1)  # gated norm (no +1)
    ttnn.deallocate(out_r)
    out_f = ttnn.reshape(out_n, (1, R, dn.value_dim_tp))
    ttnn.deallocate(out_n)
    gated = _silu_mul(out_f, z, _L1)
    ttnn.deallocate(out_f)
    ttnn.deallocate(z)

    partial = dn._row_proj(gated, tw["out"])
    ttnn.deallocate(gated)
    partial = ttnn.reshape(partial, (1, 1, R, partial.shape[-1]))
    return tt_all_reduce(
        partial,
        dn.mesh,
        dn.tt_ccl,
        cluster_axis=0,
        dim=3,
        topology=dn.args.ccl_topology(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
