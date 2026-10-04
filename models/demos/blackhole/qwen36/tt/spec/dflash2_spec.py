# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DFlash2 speculative greedy decoding for a TP Qwen36Model (single sequence).

Step (z-lab dflash_generate, temperature 0; llama.cpp common/speculative.cpp draft-dflash). The
committed tokens end at x_p, not yet in the target state; d_1..d_7 are the drafts for p+1..p+7.
  1. Verify rows [x_p, d_1..d_7] at p..p+7 (TargetVerifier, 8 rows), keeping the residual stream
     after the drafter's target layers [5, 19, 33, 47, 61] of every row, and argmaxes y_0..y_7.
  2. n = the longest prefix with y_i == d_{i+1}. Commit y_0..y_n, p' = p+n+1. If n < 7, roll the
     DeltaNet state back to its snapshot after row n.
  3. Drafter call: inject the 8 verify rows' features at p..p+7 (rows 0..n are the committed tokens;
     rows past n land at p'..p+7 and the block overwrites them), then denoise the block
     [y_n, mask x 7] at p'..p'+7; the host selector walks rows 1..7 to the next 7 drafts.
Prompt (any length; Qwen36Model.prefill_chunked_collect): the target prefill keeps the taps of every
prompt row, chunk by chunk; the drafter writes the context K/V of the positions its sliding window
reaches from the first block (>= T+8-W), and the first block is [y_0, mask x 7] at T..T+7.

Every committed token is the target's argmax given the committed prefix, so the output equals
plain greedy decoding token for token.
"""
import time

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.generator_interface import unpack_rope
from models.demos.blackhole.qwen36.tt.spec.argmax import combine_argmax_host, sharded_argmax_rows
from models.demos.blackhole.qwen36.tt.spec.dflash2 import rope_host
from models.demos.blackhole.qwen36.tt.spec.verify import TargetVerifier
from models.tt_transformers.tt.common import copy_host_to_device


def _host_i32(values):
    return ttnn.from_torch(torch.tensor(values, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)


def _host_u32_col(values):
    return ttnn.from_torch(torch.tensor([[v] for v in values], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)


def _upd_host(pos, R, modulo=None):
    """R update-index tensors: tensor r holds pos+r (mod `modulo`, a ring's capacity) at row r and -1 elsewhere."""
    slot = [(pos + r) % modulo if modulo else pos + r for r in range(R)]
    return [_host_i32([slot[r] if i == r else -1 for i in range(R)]) for r in range(R)]


def _memory_view(mesh, buffer_type):
    view = ttnn.get_memory_view(mesh, buffer_type)
    return {
        "num_banks": view.num_banks,
        "total_bytes_per_bank": view.total_bytes_per_bank,
        "free_bytes_per_bank": view.total_bytes_free_per_bank,
        "largest_contiguous_free_per_bank": view.largest_contiguous_bytes_free_per_bank,
    }


class DFlash2Decoder:
    """Greedy DFlash2 speculative decoding: one 8-token block draft and one 8-row verify per step."""

    def __init__(self, model, drafter, host_embedding=False):
        """host_embedding=True moves the target's token embedding table to the host first
        (tt/spec/host_embedding.py; frees 1.21 GiB per N300 ASIC; irreversible for this model
        object, whose plain `_run_tp_generation` then embeds on the host too)."""
        self.model = model
        self.drafter = drafter
        self.R = drafter.block_size
        self.k = self.R - 1
        self.mesh = model.mesh_device
        self.vocab = model.args.vocab_size
        self.per_shard = self.vocab // model.num_devices
        self.profile_iters = 0  # > 0: time back-to-back trace replays once after capture (stats["profile_ms"])
        if host_embedding:
            from models.demos.blackhole.qwen36.tt.spec.host_embedding import enable_host_embedding

            enable_host_embedding(model)

    def _tokens_host(self, tokens):
        emb = self.model.host_embedding
        return emb.host(tokens) if emb is not None else _host_u32_col(tokens)

    def _verify_inputs_host(self, tokens, pos):
        R = len(tokens)
        # Token ids, or the embedded rows with the host embedding table (prepare_decode_inputs_host).
        tok, cur, rope, _ = self.model.prepare_decode_inputs_host(
            torch.tensor([[t] for t in tokens], dtype=torch.int32), torch.arange(pos, pos + R, dtype=torch.int32)
        )
        return [tok, cur, rope, *_upd_host(pos, R)]

    def _draft_inputs_host(self, inj_pos, anchor, pos):
        """inj rope at inj_pos.., block tokens [anchor, mask..], block rope at pos.., cur_pos, then the
        block's and the injected rows' update indices (drafter ring slots)."""
        dr, R = self.drafter, self.R
        return [
            rope_host(range(inj_pos, inj_pos + R), dr.HD, dr.theta),
            self._tokens_host([anchor] + [dr.mask_token_id] * (R - 1)),
            rope_host(range(pos, pos + R), dr.HD, dr.theta),
            _host_i32([pos + R - 1] * R),
            *_upd_host(pos, R, dr.kv_capacity),
            *_upd_host(inj_pos, R, dr.kv_capacity),
        ]

    def _snapshot_gdn_host(self):
        comp = ttnn.ConcatMeshToTensor(self.mesh, dim=0)
        return [
            (ttnn.to_torch(dn.rec_state, mesh_composer=comp), [ttnn.to_torch(c, mesh_composer=comp) for c in dn.conv_states])
            for dn in self._gdn
        ]

    def _restore_gdn_host(self, snap):
        mapper = ttnn.ShardTensorToMesh(self.mesh, dim=0)
        for dn, (rec, convs) in zip(self._gdn, snap):
            for src, dst in [(rec, dn.rec_state), *zip(convs, dn.conv_states)]:
                t = ttnn.from_torch(src, dtype=dst.dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh, mesh_mapper=mapper)
                ttnn.copy(t, dst)
                ttnn.deallocate(t)

    def generate(self, token_ids, max_new_tokens, num_blocks, block_size=64, use_trace=True, draft_override=None):
        """Greedy-generate max_new_tokens tokens after token_ids [1, T] (any T that fits the KV).

        draft_override(index, slot, draft) -> draft (tests only): replaces the slot-th draft of a
        step (slot 0..6), which proposes generated token `index`.
        Returns (generated token ids, stats dict)."""
        model, dr, mesh = self.model, self.drafter, self.mesh
        num_blocks = ((num_blocks + 31) // 32) * 32
        model.allocate_kv_caches([num_blocks, model.args.n_local_kv_heads, block_size, model.args.head_dim], ttnn.bfloat16, batch_size=1)
        verifier = TargetVerifier(model, self.R)
        trace_ids = []
        try:  # everything allocated from here on is freed on the way out, also when an allocation fails
            # The drafter attends over its sliding window only, so its KV is a ring (DFlash2Drafter
            # docstring) of at most num_blocks blocks.
            ring = min(dr.kv_ring_blocks(block_size, max(model._PREFILL_MASK_BUCKETS)) or num_blocks, num_blocks)
            dr.allocate_kv_cache(ring, block_size)
            verifier.allocate()
            self._gdn = [layer.attention for layer in model.layers if not layer.is_full_attention]
            page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
            return self._generate(token_ids, max_new_tokens, page_table, verifier, use_trace, draft_override, trace_ids)
        finally:
            for tid in trace_ids:
                ttnn.release_trace(mesh, tid)
            verifier.free()
            dr.free_kv_cache()
            model.free_kv_caches()

    def _generate(self, token_ids, max_new_tokens, page_table, verifier, use_trace, draft_override, trace_ids):
        model, dr, mesh = self.model, self.drafter, self.mesh
        R, k, vocab, per_shard = self.R, self.k, self.vocab, self.per_shard
        taps_ids = tuple(dr.target_layer_ids)
        T = token_ids.shape[1]

        # Prefill: the target's chunked path (prefill_chunked_collect) keeping the drafter's taps of
        # every prompt row; the drafter writes the context K/V of the chunks its window reaches
        # (the first block, at T..T+7, reads positions >= T+8-W).
        t0 = time.time()
        window = dr.sliding_window or T + R

        def draft_chunk(cs, n, _hidden, taps):
            if cs + n > T + 8 - window:
                dr.prefill(taps, chunk_start=cs)

        model._keep_prefill_taps = taps_ids
        try:
            logits_dev, last_cs, _ = model.prefill_chunked_collect(token_ids[:, :T], page_table, T, on_chunk=draft_chunk)
        finally:
            model._keep_prefill_taps = ()
        prompt_taps, model.prefill_taps = model.prefill_taps, None
        lt = ttnn.to_torch(logits_dev, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        y = int(torch.argmax(lt.reshape(-1, vocab)[0].float()).item())
        generated = [y]
        ttft = time.time() - t0
        dr.prefill(prompt_taps, chunk_start=last_cs)
        for t in prompt_taps:
            ttnn.deallocate(t)
        ttnn.synchronize_device(mesh)
        prompt_s = time.time() - t0

        # Persistent step inputs (allocated before any trace; refreshed with host->device copies).
        pos = T
        v_dev = copy_host_to_device(self._verify_inputs_host([y] * R, pos), mesh_device=mesh)
        v_tok, v_pos, v_rope, v_upd = v_dev[0], v_dev[1], v_dev[2], v_dev[3:]
        d_dev = copy_host_to_device(self._draft_inputs_host(pos, y, pos), mesh_device=mesh)
        d_inj_rope, d_tok, d_rope, d_cur = d_dev[:4]
        d_upd, d_inj_upd = d_dev[4 : 4 + R], d_dev[4 + R :]
        dev_pt = ttnn.to_device(ttnn.from_torch(page_table.repeat(R, 1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), device=mesh)
        dr_pt = ttnn.to_device(ttnn.from_torch(dr.ring_page_table(R), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), device=mesh)

        def verify_fwd():
            cos, sin = unpack_rope(v_rope)
            logits, hidden, taps = verifier.forward(v_tok, cos, sin, v_pos, dev_pt, v_upd, tap_layers=taps_ids)
            ttnn.deallocate(hidden)
            idx, val = sharded_argmax_rows(logits, R, per_shard)
            ttnn.deallocate(logits)
            return idx, val, taps

        def draft_fwd(taps):
            # The injected rows sit at the verify positions (their ring slots in d_inj_upd).
            icos, isin = unpack_rope(d_inj_rope)
            cos, sin = unpack_rope(d_rope)
            return dr.decode(taps, icos, isin, d_inj_upd, d_tok, cos, sin, d_upd, d_cur, dr_pt)

        # Compile (and capture) on the first step's inputs. The GDN state is restored afterwards; the
        # target and drafter KV entries written at T.. are rewritten before they are read.
        gdn_snap = self._snapshot_gdn_host()
        t_c = time.time()
        vi, vv, vtaps = verify_fwd()
        outs = [vi, vv, *draft_fwd(vtaps)]
        for j in range(k):
            verifier.rollback(j)
        for t in outs + vtaps:
            ttnn.deallocate(t)
        trace_rollback = []
        if use_trace:

            def capture(fn):
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                out = fn()
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                trace_ids.append(tid)
                return tid, out

            trace_verify, (tv_idx, tv_val, tv_taps) = capture(verify_fwd)
            trace_draft, (td_vals, td_idx, td_sel) = capture(lambda: draft_fwd(tv_taps))
            for j in range(k):
                trace_rollback.append(capture(lambda j=j: verifier.rollback(j))[0])
        else:
            # Eager: the first block's injected rows are dummies (written at T..T+7, then
            # overwritten by the block itself).
            zero_taps = [
                ttnn.from_torch(
                    torch.zeros(1, 1, R, model.args.dim, dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
                )
                for _ in taps_ids
            ]
        self._restore_gdn_host(gdn_snap)
        ttnn.synchronize_device(mesh)
        logger.info(f"[DFlash2] compile{' + trace capture' if use_trace else ''}: {time.time() - t_c:.1f}s")
        dram, trace_region = _memory_view(mesh, ttnn.BufferType.DRAM), _memory_view(mesh, ttnn.BufferType.TRACE)
        logger.info(f"[DFlash2] DRAM per device: {dram}; trace region: {trace_region}")
        # Optional device-bound profile (self.profile_iters > 0): back-to-back replays without
        # readback. Replays write KV entries at T.. (rewritten later); the GDN state is restored.
        profile = {}
        if use_trace and self.profile_iters:
            for name, tid in (("verify", trace_verify), ("draft", trace_draft), ("rollback", trace_rollback[0])):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                t_p = time.time()
                for _ in range(self.profile_iters):
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                profile[name] = 1000 * (time.time() - t_p) / self.profile_iters
            self._restore_gdn_host(gdn_snap)
            logger.info(f"[DFlash2] replay ms: {profile}")

        def run_draft(inj_pos, anchor, block_pos, taps):
            copy_host_to_device(self._draft_inputs_host(inj_pos, anchor, block_pos), device_tensors=d_dev)
            if use_trace:
                ttnn.execute_trace(mesh, trace_draft, cq_id=0, blocking=False)
                outs = (td_vals, td_idx, td_sel)
            else:
                outs = draft_fwd(taps)
            cands, unary, s = dr.read_candidates(*outs)
            if not use_trace:
                for t in outs:
                    ttnn.deallocate(t)
            drafts = dr.select(cands, unary, s, anchor)
            if draft_override is not None:
                drafts = [draft_override(len(generated) + i, i, d) for i, d in enumerate(drafts)]
            return drafts

        # First block: the injected rows are the (unused) verify tap buffers at T..T+7, which the
        # block overwrites; the verify update indices already hold T..T+7.
        drafts = run_draft(pos, y, pos, None if use_trace else zero_taps)
        if not use_trace:
            for t in zero_taps:
                ttnn.deallocate(t)
        logger.info(f"[DFlash2] prefill T={T}: first token {y}, first drafts {drafts}, ttft {ttft * 1000:.1f} ms")

        accepted_at = [0] * k
        step_times, step_tokens, verify_times, draft_times = [], [], [], []
        cur = y
        while len(generated) < max_new_tokens:
            t_step = time.time()
            copy_host_to_device(self._verify_inputs_host([cur, *drafts], pos), device_tensors=v_dev)
            if use_trace:
                ttnn.execute_trace(mesh, trace_verify, cq_id=0, blocking=False)
                v_idx, v_val, v_taps = tv_idx, tv_val, tv_taps
            else:
                v_idx, v_val, v_taps = verify_fwd()
            ys = combine_argmax_host(v_idx, v_val, mesh, R, per_shard)
            if not use_trace:
                ttnn.deallocate(v_idx)
                ttnn.deallocate(v_val)
            t_verify = time.time()

            n = 0
            while n < k and ys[n] == drafts[n]:
                n += 1
            for i in range(n):
                accepted_at[i] += 1
            generated += ys[: n + 1]
            if n < k:
                if use_trace:
                    ttnn.execute_trace(mesh, trace_rollback[n], cq_id=0, blocking=False)
                else:
                    verifier.rollback(n)
            new_pos = pos + n + 1
            drafts = run_draft(pos, ys[n], new_pos, v_taps)
            if not use_trace:
                for t in v_taps:
                    ttnn.deallocate(t)
            pos, cur = new_pos, ys[n]
            t_end = time.time()
            step_times.append(t_end - t_step)
            verify_times.append(t_verify - t_step)
            draft_times.append(t_end - t_verify)
            step_tokens.append(n + 1)

        generated = generated[:max_new_tokens]
        steps = len(step_times)
        accepted = sum(accepted_at)
        steady_t, steady_n = sum(step_times[1:]), sum(step_tokens[1:])

        def mean_ms(ts):
            return 1000 * sum(ts[1:]) / max(1, len(ts) - 1)

        stats = {
            "method": "dflash2",
            "block_size": R,
            "k": k,
            "ttft_s": ttft,
            # Prompt length and the time to the first draft (target prefill + drafter prompt pass).
            "prompt_tokens": T,
            "prompt_s": prompt_s,
            "drafts": k * steps,
            "accepted": accepted,
            "acceptance": accepted / (k * steps) if steps else 0.0,
            "steps": steps,
            "mean_accepted_length": sum(step_tokens) / steps if steps else 0.0,
            "accepted_per_position": accepted_at,
            "acceptance_per_position": [a / steps if steps else 0.0 for a in accepted_at],
            "conditional_acceptance_per_position": [
                accepted_at[i] / (accepted_at[i - 1] if i else steps) if (accepted_at[i - 1] if i else steps) else 0.0
                for i in range(k)
            ],
            "decode_tok_s": steady_n / steady_t if steady_t > 0 else 0.0,
            # Steady step split: verify (inputs, trace, argmax readback); draft (rollback on reject,
            # drafter inputs, trace, candidate readback, host selector).
            "step_ms_mean": mean_ms(step_times),
            "verify_ms_mean": mean_ms(verify_times),
            "draft_ms_mean": mean_ms(draft_times),
            "dram": dram,
            "trace_region": trace_region,
            "profile_ms": profile,
        }
        return generated, stats
