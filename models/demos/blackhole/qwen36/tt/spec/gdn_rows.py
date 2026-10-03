# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gated DeltaNet over k rows of ONE sequence (speculative verify), with per-row state snapshots.

`TPGatedDeltaNet.forward_decode` treats decode rows as independent users, each with its own state
row. A verify step instead feeds k consecutive tokens of the same sequence, so the conv shift
register and the T=1 recurrence must run row by row on the single sequence state. The in- and
out-projections, gating and norm stay batched over the rows (rows are independent there).

State ping-pong: row r reads the state left by row r-1 and writes the state after row r into
snapshot buffer r; the last row writes the layer's own buffers. So the snapshots after rows
0..k-2 cost no extra copies over k plain decode steps, and rolling back to "after row j" is one
device-to-device copy per buffer into the layer's fixed-address state (trace-safe).

Row r runs exactly the ops of a B=1 `forward_decode` on its slice of the projections.
"""
import os

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.gdn.tp import _silu_mul, _softplus_add
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    recurrent_gated_delta_rule_decode_ttnn,
)
from models.tt_transformers.tt.ccl import tt_all_reduce


def _zeros_like_replicated(t, mesh):
    return ttnn.from_torch(
        torch.zeros(tuple(t.shape), dtype=torch.float32 if t.dtype == ttnn.float32 else torch.bfloat16),
        dtype=t.dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


class GDNRowState:
    """Snapshot buffers after rows 0..rows-2 for one TP GDN layer (B=1 state)."""

    def __init__(self, dn, rows):
        assert dn.B == 1, "row-wise verify runs on a single-sequence (B=1) GDN state"
        assert dn.conv_states is not None and dn.rec_state is not None, "allocate the GDN state first"
        self.dn = dn
        self.rows = rows
        self.rec = [_zeros_like_replicated(dn.rec_state, dn.mesh) for _ in range(rows - 1)]
        self.conv = [[_zeros_like_replicated(c, dn.mesh) for c in dn.conv_states] for _ in range(rows - 1)]

    def restore(self, row):
        """Roll the layer state back to the snapshot after `row` (device copies, no allocation)."""
        ttnn.copy(self.rec[row], self.dn.rec_state)
        for src, dst in zip(self.conv[row], self.dn.conv_states):
            ttnn.copy(src, dst)

    def free(self):
        for t in self.rec:
            ttnn.deallocate(t)
        for convs in self.conv:
            for t in convs:
                ttnn.deallocate(t)
        self.rec, self.conv = [], []


def gdn_decode_rows(dn, x, state):
    """Verify-step GDN over x's rows (consecutive tokens of one sequence).

    x: [1, 1, R, dim] replicated (gathered norm output). Returns [1, 1, R, dim] fractured, like
    `forward_decode`. Leaves the state after row R-1 in dn and after rows 0..R-2 in `state`."""
    tw, Nk, Nv, Dk, Dv, K = dn.tw, dn.Nk, dn.Nv, dn.Dk, dn.Dv, dn.K
    _L1 = ttnn.L1_MEMORY_CONFIG
    if len(x.shape) == 4:
        x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))
    R = x.shape[-2]
    assert R == state.rows, f"verify rows {R} != snapshot rows {state.rows}"

    qkv, z, a, b = dn._project_qkvzab(x, R, out_mc=_L1)

    kd = dn.key_dim_tp
    rf = Nv // Nk
    high_precision = os.environ.get("QWEN35_GDN_DECODE_BF16") != "1"
    src_conv, src_rec = dn.conv_states, dn.rec_state
    outs = []
    for r in range(R):
        last = r == R - 1
        dst_conv = dn.conv_states if last else state.conv[r]
        dst_rec = dn.rec_state if last else state.rec[r]

        # Conv shift register (src -> dst, one step) + weighted sum + SiLU.
        qkv_r = ttnn.slice(qkv, (0, r, 0), (1, r + 1, dn.qkv_dim_tp), memory_config=_L1)
        for j in range(K - 1):
            ttnn.copy(src_conv[j + 1], dst_conv[j])
        ttnn.copy(qkv_r, dst_conv[K - 1])
        ttnn.deallocate(qkv_r)
        conv = ttnn.multiply(dst_conv[0], tw["conv_taps"][0], memory_config=_L1)
        for j in range(1, K):
            conv = ttnn.mac(dst_conv[j], tw["conv_taps"][j], conv)
        conv = ttnn.silu(conv, memory_config=_L1)

        q = ttnn.reshape(ttnn.slice(conv, (0, 0, 0), (1, 1, kd)), (1, Nk, Dk))
        k = ttnn.reshape(ttnn.slice(conv, (0, 0, kd), (1, 1, 2 * kd)), (1, Nk, Dk))
        v = ttnn.reshape(ttnn.slice(conv, (0, 0, 2 * kd), (1, 1, dn.qkv_dim_tp)), (1, Nv, Dv))
        ttnn.deallocate(conv)
        q = ttnn.repeat_interleave(q, rf, dim=1)
        k = ttnn.repeat_interleave(k, rf, dim=1)
        q = ttnn.reshape(q, (1, 1, Nv, Dk), memory_config=_L1)
        k = ttnn.reshape(k, (1, 1, Nv, Dk), memory_config=_L1)
        v = ttnn.reshape(v, (1, 1, Nv, Dv), memory_config=_L1)

        b_r = ttnn.slice(b, (0, r, 0), (1, r + 1, Nv), memory_config=_L1)
        a_r = ttnn.slice(a, (0, r, 0), (1, r + 1, Nv), memory_config=_L1)
        beta = ttnn.reshape(ttnn.sigmoid(b_r, memory_config=_L1), (1, 1, Nv))
        ttnn.deallocate(b_r)
        g = ttnn.multiply(tw["neg_exp_A"], _softplus_add(a_r, tw["dt_bias"]), memory_config=_L1)
        ttnn.deallocate(a_r)
        g = ttnn.reshape(g, (1, 1, Nv))

        o, new_rec = recurrent_gated_delta_rule_decode_ttnn(
            q,
            k,
            v,
            beta,
            g,
            scale=dn.scale,
            initial_state=src_rec,
            device=dn.mesh,
            high_precision=high_precision,
        )
        ttnn.copy(new_rec, dst_rec)
        ttnn.deallocate(new_rec)
        outs.append(o)
        src_conv, src_rec = dst_conv, dst_rec
    ttnn.deallocate(qkv)
    ttnn.deallocate(a)
    ttnn.deallocate(b)

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
