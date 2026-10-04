# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DFlash2 block drafter (z-lab/Qwen3.8-27B-DFlash2) on the TP path, for one sequence.

Reference: z-lab/dflash dflash/model.py (DFlash2DraftModel, dflash_generate); llama.cpp
src/models/dflash.cpp and common/speculative.cpp (draft-dflash).

Model (config.json `dflash_config`): 5 Qwen3 decoder layers (hidden 5120, 32 q / 8 kv heads,
head_dim 128, standard x*w RMSNorms, full-dim RoPE theta 1e7, non-causal attention, sliding window
2048), block_size 8, mask token 248070. No embedding or LM head of its own: the target's are used.

Context. The target's residual stream after layers [5, 19, 33, 47, 61] (HF hidden_states[l+1]) of
every committed position is fused as g = hidden_norm(fc(cat(taps))) and written, per drafter layer,
as K = rope(k_norm(k_proj(g))), V = v_proj(g) into the drafter's own KV cache at that position.

Block. Rows [anchor, mask x 7] at positions p'..p'+7 (anchor = the last committed token, not yet in
the target state). Per layer: h = conv_side0(input_norm(x)); q/k/v from h, written to the cache at
p'..p'+7; attention of every row over cache positions max(0, p'+8-W)..p'+7 (non-causal, sliding window W);
out = conv_side1(o_proj);
then the same around the MLP. The convs are two-tap causal convolutions within the block:
out_t = w0_t * h_t + w1_t * h_{t-1} (h_{-1} = 0), with w_{tap,t} = base[side, tap] + per-group
coefficients from kernel_projection(normed input) (group 16). Final norm -> target LM head.

Selector (host). Rows 1..7 give top-16 candidates and their logits; s = hidden_projection(final
hidden). Walking positions 1..7 from pred = anchor: score_k = logit_k + <prev_cb[pred] * s_pos,
succ_cb[cand_k]>, pick argmax, pred = that candidate. The 7 picks are the drafts.

TP layout follows the target: the residual is fractured over the hidden dim (dim/2 per device),
norms gather, q/k/v are column-parallel by heads (16 q / 4 kv heads per device), o_proj and the
MLP down projection are row-parallel with a reduce-scatter, and fc is row-parallel over the
fractured taps. The conv on a replicated (normed) input works on the full width; the conv on a
fractured output on the local half.

Decode KV writes run one paged_update_cache per row (others -1), like the target verify, because
rows of one sequence that share a cache tile race inside one call. One decode call first writes the
injected context rows (the last verify's 8 rows at p..p+7), then the block rows at p'..p'+7, so
injected rows past the accepted ones are overwritten by the block before anything reads them.

Sliding window (W = config `sliding_window`, 2048). The paged SDPA decode masks keys before
cur_pos + 1 - W, with cur_pos = p'+7 for every row (the non-causal block needs every row to see all
8 block positions). The reference windows each row by its own position (query - key < W), so rows
0..6 here see up to 7 more of the oldest context entries; that only moves the drafts, never the
committed text. Since nothing older than W is read, the KV cache is a ring of `kv_ring_blocks`
blocks: position p lives in slot p mod kv_capacity (update indices are taken modulo it, and the
SDPA reads through `cache_position_modulo`).
"""
import json
from pathlib import Path

import torch

import ttnn
from models.common.rmsnorm import RMSNorm
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.attention.rope_tp import apply_partial_rope_decode, apply_partial_rope_prefill
from models.demos.blackhole.qwen36.tt.mlp import Qwen36MLP
from models.tt_transformers.tt.ccl import tt_all_reduce
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.distributed_norm import DistributedNorm

_L1 = ttnn.L1_MEMORY_CONFIG
_DRAM = ttnn.DRAM_MEMORY_CONFIG
# Exact for 0/1 matrices (shift, group expansion): full-fidelity multiplies, fp32 accumulation.
_EXACT = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
)


def load_dflash2_checkpoint(checkpoint_dir):
    """(config dict, state dict) from a DFlash2 checkpoint directory (config.json + model.safetensors)."""
    from safetensors.torch import load_file

    checkpoint_dir = Path(checkpoint_dir)
    config = json.loads((checkpoint_dir / "config.json").read_text())
    assert config["architectures"] == ["DFlash2DraftModel"], config["architectures"]
    return config, load_file(str(checkpoint_dir / "model.safetensors"))


def rope_host(positions, head_dim, theta):
    """Packed [cos; sin] host tensor [2, R, 1, head_dim] (HF split halves), as prepare_decode_inputs_host."""
    pos = torch.as_tensor(positions, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    emb = torch.cat([torch.outer(pos, inv_freq)] * 2, dim=-1)
    R = pos.shape[0]
    packed = torch.cat([emb.cos().reshape(1, R, 1, head_dim), emb.sin().reshape(1, R, 1, head_dim)], dim=0)
    return ttnn.from_torch(packed.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)


class DFlash2Drafter:
    def __init__(self, model, config, state_dict, tensor_cache_path):
        args = model.args
        mesh = model.mesh_device
        tp = model.num_devices
        assert tp == 2, "laid out for the N300's two devices"
        dc = config["dflash_config"]
        self.model, self.args, self.mesh, self.tp = model, args, mesh, tp
        self.dim = config["hidden_size"]
        assert self.dim == args.dim
        self.n_layers = config["num_hidden_layers"]
        self.HD = config["head_dim"]
        self.NH = config["num_attention_heads"] // tp
        self.NKV = config["num_key_value_heads"] // tp
        self.eps = config["rms_norm_eps"]
        self.theta = config["rope_parameters"]["rope_theta"]
        self.block_size = dc["block_size"]
        self.mask_token_id = dc["mask_token_id"]
        self.target_layer_ids = list(dc["target_layer_ids"])
        self.top_k = dc["selector_top_k"]
        self.group = dc["conv_group_size"]
        self.kernel = dc["conv_kernel_size"]
        assert self.kernel == 2, "two-tap convolution"
        assert not config.get("is_causal", False)
        self.sliding_window = config.get("sliding_window")
        self.scale = self.HD**-0.5
        self.vocab = args.vocab_size
        self.per_shard = self.vocab // tp
        cache = Path(tensor_cache_path) / "dflash2" if tensor_cache_path else None
        if cache is not None:
            cache.mkdir(parents=True, exist_ok=True)

        def c(name):
            return str(cache / name) if cache is not None else None

        def rep(t, name, dtype=ttnn.bfloat16):
            return ttnn.as_tensor(
                t.to(torch.bfloat16),
                dtype=dtype,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                layout=ttnn.TILE_LAYOUT,
                memory_config=_DRAM,
                cache_file_name=c(name),
            )

        def shard_last(t, name, dtype=ttnn.bfloat16):
            return ttnn.as_tensor(
                t.to(torch.bfloat16),
                dtype=dtype,
                device=mesh,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
                layout=ttnn.TILE_LAYOUT,
                memory_config=_DRAM,
                cache_file_name=c(name),
            )

        def norm(key, ag_key=None):
            n = RMSNorm(
                device=mesh,
                dim=self.dim,
                state_dict=state_dict,
                weight_key=key,
                state_dict_prefix="",
                weight_cache_path=cache,
                weight_dtype=ttnn.bfloat16,
                add_unit_offset=False,  # Qwen3 RMSNorm: x * w
                eps=self.eps,
                is_distributed=args.is_distributed_norm,
                ccl_topology=args.ccl_topology(),
                tt_ccl=model.tt_ccl,
            )
            return DistributedNorm(n, args, tt_ccl=model.tt_ccl, TG=args.is_galaxy, ag_config_key=ag_key)

        # fc [dim, 5*dim] row-parallel over the fractured taps: device d holds the rows for its hidden
        # half of every tap, in the order its local concat [tap_0 .. tap_4] produces.
        D, half, ntap = self.dim, self.dim // tp, len(self.target_layer_ids)
        fc = state_dict["fc.weight"]
        fc_re = fc.view(D, ntap, tp, half).permute(0, 2, 1, 3).reshape(D, ntap * D)
        self.fc = tpc.shard_w(fc_re, mesh, dim=0, memory_config=_DRAM, cache_path=c("fc.weight.tp"), dtype=ttnn.bfloat8_b)
        self.hidden_norm = norm("hidden_norm", "attention_norm")
        self.final_norm = norm("norm")
        self.hproj = rep(state_dict["candidate_selector.hidden_projection.weight"].T, "selector.hidden_projection")
        # Host codebooks (two row gathers per step; 2 x 127 MB stay off the device).
        self.prev_cb = state_dict["candidate_selector.predecessor_codebook"].float()
        self.succ_cb = state_dict["candidate_selector.successor_codebook"].float()

        # Group expansion E [groups, dim]: E[g, g*group + i] = 1; the local half for fractured outputs.
        groups = D // self.group
        E = torch.zeros(groups, D)
        E[torch.arange(D) // self.group, torch.arange(D)] = 1.0
        self.E_full = rep(E, f"conv.expand.g{self.group}")
        self.E_local = shard_last(E, f"conv.expand.g{self.group}.tp")
        self.groups = groups
        # Block shift S [R, R]: (S h)_t = h_{t-1}, row 0 -> 0.
        R = self.block_size
        S = torch.zeros(R, R)
        S[torch.arange(1, R), torch.arange(0, R - 1)] = 1.0
        self.shift = rep(S, f"conv.shift.r{R}")

        self.layers = []
        for i in range(self.n_layers):
            p = f"layers.{i}."
            if cache is not None:
                (cache / f"layers.{i}").mkdir(parents=True, exist_ok=True)
            q, k, v = (state_dict[p + f"self_attn.{n}_proj.weight"] for n in "qkv")
            qd, kd = self.NH * self.HD, self.NKV * self.HD
            fused = torch.cat(
                [torch.cat([q[d * qd : (d + 1) * qd], k[d * kd : (d + 1) * kd], v[d * kd : (d + 1) * kd]]) for d in range(tp)]
            )
            layer = {
                "wqkv": tpc.shard_w(fused, mesh, dim=-1, memory_config=_DRAM, cache_path=c(p + "wqkv.tp"), dtype=ttnn.bfloat8_b),
                "wo": tpc.shard_w(
                    state_dict[p + "self_attn.o_proj.weight"], mesh, dim=0, memory_config=_DRAM, cache_path=c(p + "wo.tp"), dtype=ttnn.bfloat8_b
                ),
                "q_norm": rep(state_dict[p + "self_attn.q_norm.weight"].reshape(1, 1, 1, -1), p + "q_norm"),
                "k_norm": rep(state_dict[p + "self_attn.k_norm.weight"].reshape(1, 1, 1, -1), p + "k_norm"),
                "attn_norm": norm(p + "input_layernorm", "attention_norm"),
                "ffn_norm": norm(p + "post_attention_layernorm", "ff_norm"),
                "mlp": Qwen36MLP(
                    mesh,
                    {n: state_dict[p + f"mlp.{n}"] for n in ("gate_proj.weight", "up_proj.weight", "down_proj.weight")},
                    cache / f"layers.{i}" if cache is not None else None,
                    args=args,
                    tt_ccl=model.tt_ccl,
                    use_gateup_agmm=False,
                ),
            }
            for conv in ("attention_conv", "mlp_conv"):
                base = state_dict[p + f"{conv}.base_kernel"]  # [side, tap, dim]
                layer[conv] = {
                    "proj": rep(state_dict[p + f"{conv}.kernel_projection.weight"].T, p + f"{conv}.proj", ttnn.bfloat8_b),
                    # side 0 acts on the replicated normed input (full width), side 1 on the fractured output.
                    "base0": [rep(base[0, j].reshape(1, -1), p + f"{conv}.base0.{j}") for j in range(2)],
                    "base1": [shard_last(base[1, j].reshape(1, -1), p + f"{conv}.base1.{j}.tp") for j in range(2)],
                }
            self.layers.append(layer)
        self.kv = None

    # ---- KV cache -------------------------------------------------------------------------------
    def kv_ring_blocks(self, block_size, max_prefill_rows):
        """KV blocks of the ring, or None without a sliding window (then the cache needs a block per
        position block). Nothing aliases inside the window: the oldest entry read is p'+8-W, and the
        farthest write ahead is a prompt chunk's padding (< max_prefill_rows past the prompt end) or
        the block (8 rows)."""
        if not self.sliding_window:
            return None
        return -(-(self.sliding_window + max_prefill_rows + 2 * block_size) // block_size)

    def allocate_kv_cache(self, num_blocks, block_size):
        """num_blocks physical blocks used as a ring: position p lives at slot p mod
        (num_blocks * block_size) (the decode page table is the identity over the blocks, and every
        update index the caller passes is taken modulo that capacity, `kv_capacity`)."""
        assert self.kv is None
        shape = [num_blocks, self.NKV, block_size, self.HD]
        self.kv_capacity = num_blocks * block_size

        def mk():
            return ttnn.as_tensor(
                torch.zeros(shape, dtype=torch.bfloat16),
                device=self.mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=_DRAM,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )

        self.kv = [(mk(), mk()) for _ in range(self.n_layers)]

    def ring_page_table(self, rows=1):
        """Identity page table over the ring's blocks, [rows, blocks] torch int32."""
        blocks = self.kv[0][0].shape[0]
        return torch.arange(blocks, dtype=torch.int32).reshape(1, blocks).repeat(rows, 1)

    def free_kv_cache(self):
        for k, v in self.kv or ():
            ttnn.deallocate(k)
            ttnn.deallocate(v)
        self.kv = None

    # ---- pieces -----------------------------------------------------------------------------------
    def _interleave(self, x):
        """x in DRAM-interleaved (x itself if it already is; otherwise a copy, and x is freed)."""
        mc = x.memory_config()
        if mc.buffer_type == ttnn.BufferType.DRAM and mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED:
            return x
        out = ttnn.to_memory_config(x, _DRAM)
        ttnn.deallocate(x)
        return out

    def _norm_decode(self, norm, x, cfg="attn"):
        nc = self.args.get_norm_config(cfg, Mode.DECODE)
        return self._interleave(norm(x, mode=Mode.DECODE, norm_config=nc))

    def _context(self, taps, mode):
        """g = hidden_norm(fc(cat(taps))), replicated [1, 1, rows, dim]. taps: fractured [1, 1, rows, dim/tp]."""
        cat = ttnn.concat(list(taps), dim=-1, memory_config=_DRAM)
        partial = ttnn.linear(cat, self.fc, compute_kernel_config=tpc.COMPUTE_HIFI2, memory_config=_DRAM)
        ttnn.deallocate(cat)
        frac = tt_all_reduce(
            partial, self.mesh, self.model.tt_ccl, cluster_axis=0, dim=3, topology=self.args.ccl_topology(), memory_config=_DRAM
        )
        if mode == Mode.DECODE:
            return self._norm_decode(self.hidden_norm, frac)
        g = self.hidden_norm(frac, mode=Mode.PREFILL)
        ttnn.deallocate(frac)
        return self._interleave(g)

    def _heads_decode(self, x, wqkv, R):
        """Fused q|k|v projection + nlp_create_qkv_heads_decode -> q [1,R,NH,HD], k/v [1,R,NKV,HD] (L1)."""
        qkv = ttnn.linear(x, wqkv, compute_kernel_config=tpc.COMPUTE_HIFI2, memory_config=_DRAM)
        qkv_l1 = ttnn.to_memory_config(qkv, _L1)
        ttnn.deallocate(qkv)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv_l1, num_heads=self.NH, num_kv_heads=self.NKV, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        )
        ttnn.deallocate(qkv_l1)
        return tuple(ttnn.sharded_to_interleaved(t, _L1) for t in (q, k, v))

    def _qk_norm(self, t, w):
        return ttnn.multiply(ttnn.rms_norm(t, epsilon=self.eps, memory_config=_L1), w, memory_config=_L1)

    def _kv_shard_cfg(self, R):
        cols = next(c for c in range(min(8, R), 0, -1) if R % c == 0)
        return ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.HD),
            core_grid=ttnn.CoreGrid(x=cols, y=R // cols),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def _write_kv(self, kv, k, v, R, update_idxs, page_table):
        """Per-row paged_update_cache of k/v [1, R, NKV, HD]."""
        cfg = self._kv_shard_cfg(R)
        for src, cache in ((k, kv[0]), (v, kv[1])):
            p = ttnn.pad(src, [1, R, 32, self.HD], [0, 0, 0, 0], 0.0, memory_config=_L1)
            sh = ttnn.to_memory_config(p, cfg)
            ttnn.deallocate(p)
            for idx in update_idxs:
                ttnn.experimental.paged_update_cache(cache, sh, update_idxs_tensor=idx, page_table=page_table)
            ttnn.deallocate(sh)

    def _conv_prepare(self, h, conv):
        """Replicated normed h [1,1,R,dim] -> (side-0 conv of h, dyn [1,1,R,4*groups] for side 1)."""
        dyn = ttnn.linear(h, conv["proj"], compute_kernel_config=tpc.COMPUTE_HIFI2, memory_config=_DRAM)
        return self._conv_apply(h, dyn, 0, conv["base0"], self.E_full), dyn

    def _conv_apply(self, h, dyn, side, bases, E):
        """out = w0 * h + w1 * shift(h), w_tap = expand(dyn[side, tap]) + base[side, tap]."""
        G = self.groups
        R = h.shape[-2]
        out = None
        for tap in range(2):
            lo = (side * 2 + tap) * G
            coef = ttnn.slice(dyn, (0, 0, 0, lo), (1, 1, R, lo + G), memory_config=_DRAM)
            w = ttnn.linear(coef, E, bias=bases[tap], compute_kernel_config=_EXACT, memory_config=_DRAM)
            ttnn.deallocate(coef)
            if tap == 0:
                values = h
            else:
                values = ttnn.matmul(self.shift, h, compute_kernel_config=_EXACT, memory_config=_DRAM)
            term = ttnn.multiply(w, values, memory_config=_DRAM)
            ttnn.deallocate(w)
            if tap:
                ttnn.deallocate(values)
            if out is None:
                out = term
            else:
                new = ttnn.add(out, term, memory_config=_DRAM)
                ttnn.deallocate(out)
                ttnn.deallocate(term)
                out = new
        return out

    # ---- prompt pass --------------------------------------------------------------------------
    def prefill(self, taps, chunk_start=0):
        """Write the context K/V for prompt positions chunk_start..chunk_start+rows-1 (one prefill
        chunk, eager; long prompts call it per chunk, in order, for the chunks inside the window).
        taps: fractured target taps [1, 1, rows, dim/tp] (Qwen36Model.prefill_taps); rows past the
        prompt carry padding features, written at positions >= the prompt length, which the first
        block overwrites before anything reads them (the next block rewrites p'..p'+7 each step).
        chunk_start is block-aligned; the chunk's blocks wrap around the ring."""
        S = taps[0].shape[-2]
        g = self._context(taps, Mode.PREFILL)
        blocks, block = self.kv[0][0].shape[0], self.kv[0][0].shape[2]
        assert chunk_start % block == 0, chunk_start
        blk0 = chunk_start // block
        nblk = (S + block - 1) // block
        assert nblk <= blocks, f"prefill chunk of {S} rows exceeds the {blocks}-block ring"
        ring_blocks = torch.tensor([[(blk0 + j) % blocks for j in range(nblk)]], dtype=torch.int32)
        pt = ttnn.from_torch(ring_blocks, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh)
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.HD, 2).float() / self.HD))
        positions = torch.arange(chunk_start, chunk_start + S, dtype=torch.float32)
        emb = torch.cat([torch.outer(positions, inv_freq)] * 2, dim=-1)
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        cos = ttnn.from_torch(emb.cos()[None, None].to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh, mesh_mapper=rep)
        sin = ttnn.from_torch(emb.sin()[None, None].to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh, mesh_mapper=rep)
        for layer, kv in zip(self.layers, self.kv):
            qkv = ttnn.linear(g, layer["wqkv"], compute_kernel_config=tpc.COMPUTE_HIFI2, memory_config=_DRAM)
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                qkv, num_heads=self.NH, num_kv_heads=self.NKV, transpose_k_heads=False, memory_config=_DRAM
            )
            ttnn.deallocate(qkv)
            ttnn.deallocate(q)
            kn = ttnn.multiply(ttnn.rms_norm(k, epsilon=self.eps, memory_config=_DRAM), layer["k_norm"], memory_config=_DRAM)
            ttnn.deallocate(k)
            kr = apply_partial_rope_prefill(kn, cos, sin, self.NKV, self.HD)
            ttnn.deallocate(kn)
            ttnn.experimental.paged_fill_cache(kv[0], kr, pt, batch_idx=0)
            ttnn.experimental.paged_fill_cache(kv[1], v, pt, batch_idx=0)
            ttnn.deallocate(kr)
            ttnn.deallocate(v)
        for t in (g, pt, cos, sin):
            ttnn.deallocate(t)

    # ---- decode (traceable) -------------------------------------------------------------------
    def decode(self, taps, inj_cos, inj_sin, inj_upd, tokens, cos, sin, upd, cur_pos, page_table, return_hidden=False):
        """One drafter call: inject the context rows, then denoise the block.

        taps: fractured target taps [1, 1, R, dim/tp] of the rows to inject (the last verify's rows);
        inj_cos/inj_sin [1, R, 1, HD] and inj_upd (R update-index tensors) for their positions;
        tokens [R, 1] uint32 = [anchor, mask x (R-1)] (or, with the host embedding table, their
        embedded rows [1, 1, R, dim/tp], left intact); cos/sin [1, R, 1, HD] and upd for the block
        positions p'..p'+R-1; cur_pos [R] int32 = p'+R-1 on every row (non-causal attention over
        the window ending at p'+R-1); page_table [R, ring blocks] (ring_page_table). Update indices
        are ring slots: positions modulo kv_capacity.

        Returns device tensors (top-k values [1,1,R,k], top-k local indices [1,1,R,k] uint32 per vocab
        shard, selector projection [1,1,R,rank] replicated)."""
        model, R = self.model, self.block_size
        g = self._context(taps, Mode.DECODE)
        embedded = model.host_embedding is not None
        if embedded:
            x = tokens
        else:
            x = model.embd(tokens)
            x = ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[1], x.shape[-1]))
        assert x.shape[-2] == R
        first = True
        sdpa_grid = self.mesh.compute_with_storage_grid_size()
        sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(sdpa_grid.x, sdpa_grid.y), exp_approx_mode=False, q_chunk_size=0, k_chunk_size=0
        )
        for layer, kv in zip(self.layers, self.kv):
            # Context rows: K/V of the injected target features at their positions.
            _, ki, vi = self._heads_decode(g, layer["wqkv"], R)
            ttnn.deallocate(_)
            ki = apply_partial_rope_decode(self._qk_norm(ki, layer["k_norm"]), inj_cos, inj_sin, self.NKV, R, self.HD)
            self._write_kv(kv, ki, vi, R, inj_upd, page_table)
            ttnn.deallocate(ki)
            ttnn.deallocate(vi)

            # Attention over [context; block], conv around it.
            h = self._norm_decode(layer["attn_norm"], x)
            hc, dyn = self._conv_prepare(h, layer["attention_conv"])
            ttnn.deallocate(h)
            q, k, v = self._heads_decode(hc, layer["wqkv"], R)
            ttnn.deallocate(hc)
            q = apply_partial_rope_decode(self._qk_norm(q, layer["q_norm"]), cos, sin, self.NH, R, self.HD)
            k = apply_partial_rope_decode(self._qk_norm(k, layer["k_norm"]), cos, sin, self.NKV, R, self.HD)
            self._write_kv(kv, k, v, R, upd, page_table)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q,
                kv[0],
                kv[1],
                page_table_tensor=page_table,
                cur_pos_tensor=cur_pos,
                scale=self.scale,
                sliding_window_size=min(self.sliding_window, self.kv_capacity) if self.sliding_window else None,
                cache_position_modulo=self.kv_capacity if self.sliding_window else None,
                program_config=sdpa_cfg,
                memory_config=_L1,
            )
            ttnn.deallocate(q)
            flat = self._concat_heads(attn, R)
            partial = ttnn.linear(flat, layer["wo"], compute_kernel_config=tpc.COMPUTE_HIFI2, memory_config=_DRAM)
            ttnn.deallocate(flat)
            partial = ttnn.reshape(partial, (1, 1, R, partial.shape[-1]))
            a = tt_all_reduce(
                partial, self.mesh, model.tt_ccl, cluster_axis=0, dim=3, topology=self.args.ccl_topology(), memory_config=_DRAM
            )
            a_c = self._conv_apply(a, dyn, 1, layer["attention_conv"]["base1"], self.E_local)
            ttnn.deallocate(a)
            ttnn.deallocate(dyn)
            x1 = ttnn.add(x, a_c, memory_config=_DRAM)
            ttnn.deallocate(a_c)
            if not (first and embedded):
                ttnn.deallocate(x)
            first = False

            # MLP, conv around it.
            f = self._norm_decode(layer["ffn_norm"], x1)
            fc_, dyn = self._conv_prepare(f, layer["mlp_conv"])
            ttnn.deallocate(f)
            m = layer["mlp"].forward(fc_, mode="decode")
            ttnn.deallocate(fc_)
            m_dram = self._interleave(m)
            m_c = self._conv_apply(m_dram, dyn, 1, layer["mlp_conv"]["base1"], self.E_local)
            ttnn.deallocate(m_dram)
            ttnn.deallocate(dyn)
            x = ttnn.add(x1, m_c, memory_config=_DRAM)
            ttnn.deallocate(m_c)
            ttnn.deallocate(x1)
        ttnn.deallocate(g)

        nc = dict(self.args.get_norm_config("lm_head", Mode.DECODE))
        nc["output_mem_config"] = _DRAM
        hidden = self._interleave(self.final_norm(x, mode=Mode.DECODE, norm_config=nc))
        ttnn.deallocate(x)
        logits = ttnn.linear(hidden, model.lm_head_weight)
        vals, idx = self.chunk_topk(logits)
        ttnn.deallocate(logits)
        sel = ttnn.linear(hidden, self.hproj, compute_kernel_config=_EXACT, memory_config=_DRAM)
        if return_hidden:
            return vals, idx, sel, hidden
        ttnn.deallocate(hidden)
        return vals, idx, sel

    def _concat_heads(self, attn, R):
        from models.tt_transformers.tt.model_config import num_to_corerange

        grid = self.mesh.compute_with_storage_grid_size()
        gx = min(R, grid.x)
        core_grid = ttnn.CoreRangeSet({num_to_corerange(R, grid_x=gx, grid_y=grid.y)})
        cfg = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.HD),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        sh = ttnn.to_memory_config(attn, cfg)
        ttnn.deallocate(attn)
        out_sh = ttnn.experimental.nlp_concat_heads_decode(sh, num_heads=self.NH)
        ttnn.deallocate(sh)
        out = ttnn.sharded_to_interleaved(out_sh, _L1)
        ttnn.deallocate(out_sh)
        if out.shape[-2] != R:
            out = ttnn.slice(out, (0, 0, 0, 0), (1, 1, R, self.NH * self.HD), memory_config=_L1)
        return out

    # ---- candidates and selector --------------------------------------------------------------
    # ttnn.topk runs multi-core only on power-of-two widths below 65535; over a whole 124160-wide
    # vocab shard it takes the single-core path (36 ms for 8 rows). Each row's shard is cut into
    # power-of-two chunks (one tile row of 32 chunk rows for R=8), each chunk keeps its top k on
    # device, and the host merges the chunk winners (the global top k is among them).
    _TOPK_CHUNK = 32768

    def chunk_topk(self, logits):
        """logits [1, 1, R, per_shard] -> per-chunk (values, local indices) [1, 1, R*C, k]."""
        W = self._TOPK_CHUNK
        C = -(-self.per_shard // W)
        R = logits.shape[-2]
        padded = ttnn.pad(logits, [(0, 0), (0, 0), (0, 0), (0, C * W - self.per_shard)], value=-1e30)
        grid = ttnn.reshape(padded, (1, 1, R * C, W))
        ttnn.deallocate(padded)
        vals, idx = ttnn.topk(grid, k=self.top_k, dim=-1, largest=True, sorted=True)
        ttnn.deallocate(grid)
        return vals, idx

    def read_candidates(self, vals, idx, sel):
        """Device outputs -> (cands [R, k] global ids, unary [R, k] fp32, s [R, rank] fp32) on host."""
        R, k, W = self.block_size, self.top_k, self._TOPK_CHUNK
        C = -(-self.per_shard // W)
        comp = ttnn.ConcatMeshToTensor(self.mesh, dim=0)
        v = ttnn.to_torch(vals, mesh_composer=comp).float().reshape(self.tp, R, C, k)
        i = ttnn.to_torch(idx, mesh_composer=comp).to(torch.int64).reshape(self.tp, R, C, k)
        i = i + torch.arange(self.tp).view(-1, 1, 1, 1) * self.per_shard + torch.arange(C).view(1, 1, -1, 1) * W
        v = v.permute(1, 0, 2, 3).reshape(R, -1)
        i = i.permute(1, 0, 2, 3).reshape(R, -1)
        top, pos = torch.topk(v, k, dim=-1)
        s = ttnn.to_torch(sel, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0)).float()
        s = s.reshape(self.tp, -1, s.shape[-1])[0, :R]
        return i.gather(-1, pos), top, s

    def select(self, cands, unary, s, anchor):
        """Greedy selector walk over block rows 1..R-1 (CandidateSelector.select, temperature 0)."""
        pred = int(anchor)
        drafts = []
        for row in range(1, self.block_size):
            scores = unary[row] + (self.succ_cb[cands[row]] @ (self.prev_cb[pred] * s[row]))
            pred = int(cands[row, int(torch.argmax(scores))])
            drafts.append(pred)
        return drafts
