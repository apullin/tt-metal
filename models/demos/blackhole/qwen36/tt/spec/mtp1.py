# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MTP-1 speculative greedy decoding for a TP Qwen36Model (single sequence).

Positions (evidence SPECULATIVE.md "MTP-1 step"): committed tokens end at x_p, which is not yet in
the target state, and d is the MTP draft for x_{p+1}.
  1. verify rows [x_p, d] at [p, p+1] -> post-norm h_p, h_{p+1}, argmax y0 (for p+1), y1 (for p+2);
  2. y0 == d: commit d, y1; p += 2.  Otherwise commit y0, p += 1, roll GDN back to after row 0
     (row 1's KV entry at p+1 is rewritten by the next verify);
  3. MTP rows [(h_p, y0) at p, (h_{p+1}, y1) at p+1]; the next draft is row 1's argmax on accept,
     row 0's on reject (a rejected row 1's MTP KV entry is rewritten by the next step).

Greedy speculative output equals plain greedy output token for token: every committed token is the
target's own argmax given the committed prefix.
"""
import time

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.generator_interface import unpack_rope
from models.demos.blackhole.qwen36.tt.spec.argmax import combine_argmax_host, sharded_argmax_rows
from models.demos.blackhole.qwen36.tt.spec.verify import TargetVerifier
from models.tt_transformers.tt.common import copy_host_to_device

ROWS = 2  # verify rows per step: [x_p, draft]


class MTP1Decoder:
    def __init__(self, model, mtp):
        self.model = model
        self.mtp = mtp
        self.mesh = model.mesh_device
        self.vocab = model.args.vocab_size
        self.per_shard = self.vocab // model.num_devices

    # ---- host <-> device input staging ---------------------------------------------------------
    def _verify_inputs_host(self, tokens, pos):
        """(tokens [2,1], cur_pos [2], rope, kv update idxs row 0, row 1) host tensors."""
        tok_tt, pos_tt, rope_tt, _ = self.model.prepare_decode_inputs_host(
            torch.tensor([[tokens[0]], [tokens[1]]], dtype=torch.int32),
            torch.tensor([pos, pos + 1], dtype=torch.int32),
        )
        upd = [
            ttnn.from_torch(torch.tensor(v, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            for v in ([pos, -1], [-1, pos + 1])
        ]
        return [tok_tt, pos_tt, rope_tt, *upd]

    @staticmethod
    def _tokens_host(tokens):
        return ttnn.from_torch(
            torch.tensor([[t] for t in tokens], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )

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

    # ---- generation ----------------------------------------------------------------------------
    def generate(self, token_ids, max_new_tokens, num_blocks, block_size=64, use_trace=True, draft_override=None):
        """Greedy-generate max_new_tokens tokens after token_ids [1, T] (T <= one prefill chunk).

        draft_override(index, mtp_draft) -> draft (tests only): replaces the draft for generated
        token `index`, e.g. to force accepts (oracle) or rejects.
        Returns (generated token ids, stats dict)."""
        model, mtp, mesh = self.model, self.mtp, self.mesh
        T = token_ids.shape[1]
        num_blocks = ((num_blocks + 31) // 32) * 32
        kv_cache_shape = [num_blocks, model.args.n_local_kv_heads, block_size, model.args.head_dim]
        model.allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=1)
        mtp.allocate_kv_cache(kv_cache_shape)
        verifier = TargetVerifier(model, ROWS)
        verifier.allocate()
        self._gdn = [layer.attention for layer in model.layers if not layer.is_full_attention]
        page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
        trace_ids = []
        try:
            return self._generate(token_ids, T, max_new_tokens, page_table, verifier, use_trace, draft_override, trace_ids)
        finally:
            for tid in trace_ids:
                ttnn.release_trace(mesh, tid)
            verifier.free()
            mtp.free_kv_cache()
            model.free_kv_caches()

    def _generate(self, token_ids, T, max_new_tokens, page_table, verifier, use_trace, draft_override, trace_ids):
        model, mtp, mesh = self.model, self.mtp, self.mesh
        vocab, per_shard = self.vocab, self.per_shard

        # Prefill: the target's own masked-bucket path, keeping every row's post-norm hidden.
        t0 = time.time()
        model._keep_prefill_hidden = True
        try:
            logits_dev = model.prefill_traced_chunked(token_ids[:, :T], page_table, actual_len=T)
        finally:
            model._keep_prefill_hidden = False
        hidden_all, model.prefill_hidden = model.prefill_hidden, None
        lt = ttnn.to_torch(logits_dev, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        y = int(torch.argmax(lt.reshape(-1, vocab)[0].float()).item())
        generated = [y]
        ttft = time.time() - t0

        # MTP prompt pass: rows (h_i, x_{i+1}) at positions 0..T-1 -> draft for x_{T+1}.
        next_tokens = torch.cat([token_ids[:, 1:T].to(torch.int32), torch.tensor([[y]], dtype=torch.int32)], dim=1)
        mtp_logits = mtp.prefill(next_tokens, hidden_all, T, page_table)
        ttnn.deallocate(hidden_all)
        mt = ttnn.to_torch(mtp_logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        draft = int(torch.argmax(mt.reshape(-1, vocab)[0].float()).item())
        if draft_override is not None:
            draft = draft_override(len(generated), draft)
        logger.info(f"[MTP1] prefill T={T}: first token {y}, first draft {draft}, ttft {ttft * 1000:.1f} ms")

        # Persistent decode inputs (allocated before any trace; refreshed with host->device copies).
        pos = T
        page_table_rows = page_table.repeat(ROWS, 1)
        host = self._verify_inputs_host([y, draft], pos)
        dev = copy_host_to_device(host, mesh_device=mesh)
        dev_pt = ttnn.to_device(
            ttnn.from_torch(page_table_rows, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), device=mesh
        )
        dev_mtp_tok = ttnn.to_device(self._tokens_host([y, y]), device=mesh)
        d_tok, d_pos, d_rope, d_upd0, d_upd1 = dev
        d_upd = [d_upd0, d_upd1]

        def verify_fwd():
            cos, sin = unpack_rope(d_rope)
            logits, hidden = verifier.forward(d_tok, cos, sin, d_pos, dev_pt, d_upd)
            idx, val = sharded_argmax_rows(logits, ROWS, per_shard)
            ttnn.deallocate(logits)
            return idx, val, hidden

        def mtp_fwd(hidden):
            cos, sin = unpack_rope(d_rope)
            logits = mtp.decode(dev_mtp_tok, hidden, cos, sin, d_pos, dev_pt, d_upd)
            idx, val = sharded_argmax_rows(logits, ROWS, per_shard)
            ttnn.deallocate(logits)
            return idx, val

        # Compile (and capture) on the real first-step inputs; the GDN state is restored afterwards
        # and the KV entries written at p, p+1 are rewritten by the first real step.
        gdn_snap = self._snapshot_gdn_host()
        t_c = time.time()
        vi, vv, vh = verify_fwd()
        mi, mv = mtp_fwd(vh)
        verifier.rollback(0)
        for t in (vi, vv, vh, mi, mv):
            ttnn.deallocate(t)
        if use_trace:
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            tv_idx, tv_val, tv_hidden = verify_fwd()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            trace_ids.append(tid)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            tm_idx, tm_val = mtp_fwd(tv_hidden)
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            trace_ids.append(tid)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            verifier.rollback(0)
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            trace_ids.append(tid)
            trace_verify, trace_mtp, trace_rollback = trace_ids
        self._restore_gdn_host(gdn_snap)
        ttnn.synchronize_device(mesh)
        logger.info(f"[MTP1] compile{' + trace capture' if use_trace else ''}: {time.time() - t_c:.1f}s")

        drafts = accepted = 0
        step_times, step_tokens = [], []
        cur = y
        while len(generated) < max_new_tokens:
            t_step = time.time()
            copy_host_to_device(self._verify_inputs_host([cur, draft], pos), device_tensors=dev)
            if use_trace:
                ttnn.execute_trace(mesh, trace_verify, cq_id=0, blocking=False)
                v_idx, v_val = tv_idx, tv_val
            else:
                v_idx, v_val, v_hidden = verify_fwd()
            y0, y1 = combine_argmax_host(v_idx, v_val, mesh, ROWS, per_shard)
            if not use_trace:
                ttnn.deallocate(v_idx)
                ttnn.deallocate(v_val)
            drafts += 1
            ok = y0 == draft
            if ok:
                accepted += 1
                generated += [y0, y1]
                cur, pos_next = y1, pos + 2
            else:
                generated.append(y0)
                cur, pos_next = y0, pos + 1
                if use_trace:
                    ttnn.execute_trace(mesh, trace_rollback, cq_id=0, blocking=False)
                else:
                    verifier.rollback(0)
            copy_host_to_device([self._tokens_host([y0, y1])], device_tensors=[dev_mtp_tok])
            if use_trace:
                ttnn.execute_trace(mesh, trace_mtp, cq_id=0, blocking=False)
                m_idx, m_val = tm_idx, tm_val
            else:
                m_idx, m_val = mtp_fwd(v_hidden)
                ttnn.deallocate(v_hidden)
            m0, m1 = combine_argmax_host(m_idx, m_val, mesh, ROWS, per_shard)
            if not use_trace:
                ttnn.deallocate(m_idx)
                ttnn.deallocate(m_val)
            draft = m1 if ok else m0
            if draft_override is not None:
                draft = draft_override(len(generated), draft)
            pos = pos_next
            step_times.append(time.time() - t_step)
            step_tokens.append(2 if ok else 1)

        generated = generated[:max_new_tokens]
        # Steady state: drop the first step (as the plain demo drops its first decode step).
        steady_t, steady_n = sum(step_times[1:]), sum(step_tokens[1:])
        stats = {
            "ttft_s": ttft,
            "drafts": drafts,
            "accepted": accepted,
            "acceptance": accepted / drafts if drafts else 0.0,
            "steps": len(step_times),
            "decode_tok_s": steady_n / steady_t if steady_t > 0 else 0.0,
            "step_ms_mean": 1000 * steady_t / max(1, len(step_times) - 1),
        }
        return generated, stats
