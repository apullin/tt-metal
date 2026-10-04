# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MTP-k speculative greedy decoding for a TP Qwen36Model (single sequence).

Chained drafts. The checkpoint has one MTP layer, reused for every draft. Draft j+1 reads draft j's
token, one position later, and draft j's MTP output hidden after `mtp.norm`, the same tensor the
LM head reads. References:
  - vLLM qwen3_5_mtp.py: forward returns the hidden after `self.norm`; `spec_step_idx %
    num_mtp_layers` picks the layer, so a single layer serves every step.
  - vLLM v1/spec_decode/llm_base_proposer.py, the draft loop: next input_ids = the last draft,
    positions + 1, hidden_states = the previous MTP output.
  - llama.cpp src/models/qwen35.cpp graph_mtp: t_h_nextn is the post-norm hidden.
    common/speculative.cpp draft(), single-head path: next row = (draft id, pos0+i+1, that
    hidden). process() re-runs the accepted tokens through the MTP with the target's hiddens,
    rewriting the MTP KV that the drafts wrote.

Step. Committed tokens end at x_p, which is not yet in the target state. d_1..d_k are the drafts
for positions p+1..p+k.
  1. Verify rows [x_p, d_1..d_k] at p..p+k. This gives post-norm target hiddens h_p..h_{p+k} and
     argmaxes y_0..y_k, where y_i is the target's token for p+i+1.
  2. n = the longest prefix with y_i == d_{i+1}. Commit y_0..y_n (the n accepted drafts plus the
     target's bonus token) and set p' = p+n+1. If n < k, roll the DeltaNet state back to its
     snapshot after verify row n. Target KV entries past p+n are rewritten by later verifies.
  3. Catch-up MTP rows (h_{p+i}, y_i) at p+i, i = 0..k. Rows 0..n are the exact MTP inputs for
     the committed tokens, since y_i == d_{i+1} for i < n. Row n gives d'_1, the draft for p'+1.
     Rows past n write MTP KV entries that later steps rewrite before anything reads them.
  4. Chained drafts j = 2..k: one MTP row (d'_{j-1}, hidden) at p'+j-2 gives d'_j. The hidden is
     catch-up row n's output for j = 2, and the previous chained row's output after that.

Every committed token is the target's own argmax given the committed prefix, so the output equals
plain greedy decoding token for token.
"""
import time

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.generator_interface import unpack_rope
from models.demos.blackhole.qwen36.tt.spec.argmax import combine_argmax_host, sharded_argmax_rows
from models.demos.blackhole.qwen36.tt.spec.verify import TargetVerifier
from models.tt_transformers.tt.common import copy_host_to_device


def _select_row(hidden, idx, out):
    """out[0] = hidden row idx (on device; ttnn.embedding copies the row exactly).
    hidden [1, 1, rows, dim]; idx [1, 1] uint32; out [1, 1, 1, dim]."""
    table = ttnn.reshape(ttnn.to_layout(hidden, ttnn.ROW_MAJOR_LAYOUT), (hidden.shape[-2], hidden.shape[-1]))
    row = ttnn.embedding(idx, table, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(table)
    ttnn.copy(ttnn.reshape(row, tuple(out.shape)), out)
    ttnn.deallocate(row)


def _host_u32(values):
    return ttnn.from_torch(torch.tensor([[v] for v in values], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)


class MTPDecoder:
    """Greedy MTP-k speculative decoding: k chained MTP drafts and one (k+1)-row verify per step."""

    def __init__(self, model, mtp, k=1, host_embedding=False):
        """host_embedding=True moves the target's token embedding table to the host first
        (tt/spec/host_embedding.py; frees 1.21 GiB per N300 ASIC; irreversible for this model object,
        whose plain `_run_tp_generation` then embeds on the host too)."""
        assert k >= 1
        self.model = model
        self.mtp = mtp
        self.k = k
        self.rows = k + 1
        self.mesh = model.mesh_device
        self.vocab = model.args.vocab_size
        self.per_shard = self.vocab // model.num_devices
        if host_embedding:
            from models.demos.blackhole.qwen36.tt.spec.host_embedding import enable_host_embedding

            enable_host_embedding(model)

    def _tokens_host(self, tokens):
        """MTP token inputs [R, 1] uint32, or their embedded rows with the host embedding table."""
        emb = self.model.host_embedding
        return emb.host(tokens) if emb is not None else _host_u32(tokens)

    # ---- host <-> device input staging ---------------------------------------------------------
    def _rows_inputs_host(self, tokens, pos):
        """(tokens [R,1] or their embedded rows, cur_pos [R] = pos..pos+R-1, rope, kv update idxs per row) host tensors."""
        R = len(tokens)
        tok_tt, pos_tt, rope_tt, _ = self.model.prepare_decode_inputs_host(
            torch.tensor([[t] for t in tokens], dtype=torch.int32), torch.arange(pos, pos + R, dtype=torch.int32)
        )
        upd = [
            ttnn.from_torch(
                torch.tensor([pos + r if i == r else -1 for i in range(R)], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            for r in range(R)
        ]
        return [tok_tt, pos_tt, rope_tt, *upd]

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
        """Greedy-generate max_new_tokens tokens after token_ids [1, T] (any T that fits the KV).

        draft_override(index, slot, draft) -> draft (tests only): replaces the slot-th draft of a
        step (slot 0..k-1), which proposes generated token `index`, e.g. to force accepts or rejects.
        Returns (generated token ids, stats dict)."""
        model, mtp, mesh = self.model, self.mtp, self.mesh
        T = token_ids.shape[1]
        num_blocks = ((num_blocks + 31) // 32) * 32
        kv_cache_shape = [num_blocks, model.args.n_local_kv_heads, block_size, model.args.head_dim]
        model.allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=1)
        verifier = TargetVerifier(model, self.rows)
        trace_ids = []
        try:  # everything allocated from here on is freed on the way out, also when an allocation fails
            # The MTP layer's KV in the target's KV dtype (bfloat8_b under QWEN_SDPA_BF8=1).
            mtp.allocate_kv_cache(kv_cache_shape, dtype=model._paged_kv_caches[0][0].dtype)
            verifier.allocate()
            self._gdn = [layer.attention for layer in model.layers if not layer.is_full_attention]
            page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
            return self._generate(token_ids, T, max_new_tokens, page_table, verifier, use_trace, draft_override, trace_ids)
        finally:
            for tid in trace_ids:
                ttnn.release_trace(mesh, tid)
            verifier.free()
            mtp.free_kv_cache()
            model.free_kv_caches()

    def _generate(self, token_ids, T, max_new_tokens, page_table, verifier, use_trace, draft_override, trace_ids):
        model, mtp, mesh = self.model, self.mtp, self.mesh
        k, R, vocab, per_shard = self.k, self.rows, self.vocab, self.per_shard
        chained = k > 1

        # Prefill: the target's own chunked path (prefill_chunked_collect) keeping every row's
        # post-norm hidden. The MTP prompt pass, rows (h_i, x_{i+1}) at positions 0..T-1, follows
        # chunk by chunk; the last row's draft is the one for x_{T+1}.
        t0 = time.time()

        def mtp_chunk(cs, n, hidden, _taps):
            mtp.prefill(token_ids[:, cs + 1 : cs + n + 1], hidden, n, page_table, chunk_start=cs, want_logits=False)

        model._keep_prefill_hidden = True
        try:
            logits_dev, last_cs, last_n = model.prefill_chunked_collect(token_ids[:, :T], page_table, T, on_chunk=mtp_chunk)
        finally:
            model._keep_prefill_hidden = False
        hidden_all, model.prefill_hidden = model.prefill_hidden, None
        lt = ttnn.to_torch(logits_dev, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        y = int(torch.argmax(lt.reshape(-1, vocab)[0].float()).item())
        generated = [y]
        ttft = time.time() - t0

        next_tokens = torch.cat([token_ids[:, last_cs + 1 : T].to(torch.int32), torch.tensor([[y]], dtype=torch.int32)], dim=1)
        mtp_out = mtp.prefill(next_tokens, hidden_all, last_n, page_table, chunk_start=last_cs, return_hidden=chained)
        mtp_logits, prompt_hidden = mtp_out if chained else (mtp_out, None)
        ttnn.deallocate(hidden_all)
        mt = ttnn.to_torch(mtp_logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        first_draft = int(torch.argmax(mt.reshape(-1, vocab)[0].float()).item())
        prompt_s = time.time() - t0

        # Persistent step inputs (allocated before any trace; refreshed with host->device copies).
        pos = T
        dev = copy_host_to_device(self._rows_inputs_host([y] * R, pos), mesh_device=mesh)
        d_tok, d_pos, d_rope, d_upd = dev[0], dev[1], dev[2], dev[3:]
        dev_pt = ttnn.to_device(
            ttnn.from_torch(page_table.repeat(R, 1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), device=mesh
        )
        dev_mtp_tok = ttnn.to_device(self._tokens_host([y] * R), device=mesh)
        if chained:
            dev_sel = ttnn.to_device(_host_u32([0]), device=mesh)
            c_tok, c_pos, c_rope, _ = model.prepare_decode_inputs_host(
                torch.tensor([[y]], dtype=torch.int32), torch.tensor([pos], dtype=torch.int32)
            )
            chain_dev = copy_host_to_device([c_tok, c_pos, c_rope], mesh_device=mesh)
            c_pt = ttnn.to_device(ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), device=mesh)
            chain_hidden = ttnn.from_torch(
                torch.zeros(1, 1, 1, model.args.dim, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        def verify_fwd():
            cos, sin = unpack_rope(d_rope)
            logits, hidden = verifier.forward(d_tok, cos, sin, d_pos, dev_pt, d_upd)
            idx, val = sharded_argmax_rows(logits, R, per_shard)
            ttnn.deallocate(logits)
            return idx, val, hidden

        def catchup_fwd(hidden):
            cos, sin = unpack_rope(d_rope)
            if chained:
                logits, mtp_hidden = mtp.decode(dev_mtp_tok, hidden, cos, sin, d_pos, dev_pt, d_upd, return_hidden=True)
                _select_row(mtp_hidden, dev_sel, chain_hidden)
                ttnn.deallocate(mtp_hidden)
            else:
                logits = mtp.decode(dev_mtp_tok, hidden, cos, sin, d_pos, dev_pt, d_upd)
            idx, val = sharded_argmax_rows(logits, R, per_shard)
            ttnn.deallocate(logits)
            return idx, val

        def chain_fwd():
            cos, sin = unpack_rope(chain_dev[2])
            logits, mtp_hidden = mtp.decode(chain_dev[0], chain_hidden, cos, sin, chain_dev[1], c_pt, None, return_hidden=True)
            ttnn.copy(mtp_hidden, chain_hidden)
            ttnn.deallocate(mtp_hidden)
            idx, val = sharded_argmax_rows(logits, 1, per_shard)
            ttnn.deallocate(logits)
            return idx, val

        # Compile (and capture) on the real first-step inputs. The GDN state is restored afterwards;
        # the target and MTP KV entries written at p.. are rewritten before they are read.
        gdn_snap = self._snapshot_gdn_host()
        t_c = time.time()
        vi, vv, vh = verify_fwd()
        outs = [vi, vv, vh, *catchup_fwd(vh)]
        if chained:
            outs += chain_fwd()
        for j in range(k):
            verifier.rollback(j)
        for t in outs:
            ttnn.deallocate(t)
        trace_rollback = []
        if use_trace:

            def capture(fn):
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                out = fn()
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                trace_ids.append(tid)
                return tid, out

            trace_verify, (tv_idx, tv_val, tv_hidden) = capture(verify_fwd)
            trace_mtp, (tm_idx, tm_val) = capture(lambda: catchup_fwd(tv_hidden))
            if chained:
                trace_chain, (tc_idx, tc_val) = capture(chain_fwd)
            for j in range(k):
                trace_rollback.append(capture(lambda j=j: verifier.rollback(j))[0])
        self._restore_gdn_host(gdn_snap)
        ttnn.synchronize_device(mesh)
        logger.info(f"[MTP-{k}] compile{' + trace capture' if use_trace else ''}: {time.time() - t_c:.1f}s")

        # DRAM headroom with every speculative buffer (and trace) in place.
        def memory_view(buffer_type):
            view = ttnn.get_memory_view(mesh, buffer_type)
            return {
                "num_banks": view.num_banks,
                "total_bytes_per_bank": view.total_bytes_per_bank,
                "free_bytes_per_bank": view.total_bytes_free_per_bank,
                "largest_contiguous_free_per_bank": view.largest_contiguous_bytes_free_per_bank,
            }

        dram, trace_region = memory_view(ttnn.BufferType.DRAM), memory_view(ttnn.BufferType.TRACE)
        logger.info(f"[MTP-{k}] DRAM per device: {dram}; trace region: {trace_region}")

        def run_chain(drafts, base_pos):
            """Chained drafts 2..k after catch-up (or prefill) left the chain hidden in place."""
            while len(drafts) < k:
                copy_host_to_device(
                    list(
                        model.prepare_decode_inputs_host(
                            torch.tensor([[drafts[-1]]], dtype=torch.int32),
                            torch.tensor([base_pos + len(drafts) - 1], dtype=torch.int32),
                        )[:3]
                    ),
                    device_tensors=chain_dev,
                )
                if use_trace:
                    ttnn.execute_trace(mesh, trace_chain, cq_id=0, blocking=False)
                    c_idx, c_val = tc_idx, tc_val
                else:
                    c_idx, c_val = chain_fwd()
                d = combine_argmax_host(c_idx, c_val, mesh, 1, per_shard)[0]
                if not use_trace:
                    ttnn.deallocate(c_idx)
                    ttnn.deallocate(c_val)
                if draft_override is not None:
                    d = draft_override(len(generated) + len(drafts), len(drafts), d)
                drafts.append(d)
            return drafts

        drafts = [first_draft if draft_override is None else draft_override(len(generated), 0, first_draft)]
        if chained:
            ttnn.copy(prompt_hidden, chain_hidden)
            ttnn.deallocate(prompt_hidden)
            drafts = run_chain(drafts, pos)
        logger.info(f"[MTP-{k}] prefill T={T}: first token {y}, first drafts {drafts}, ttft {ttft * 1000:.1f} ms")

        accepted_at = [0] * k
        step_times, step_tokens, verify_times, mtp_times, chain_times = [], [], [], [], []
        cur = y
        while len(generated) < max_new_tokens:
            t_step = time.time()
            copy_host_to_device(self._rows_inputs_host([cur, *drafts], pos), device_tensors=dev)
            if use_trace:
                ttnn.execute_trace(mesh, trace_verify, cq_id=0, blocking=False)
                v_idx, v_val = tv_idx, tv_val
            else:
                v_idx, v_val, v_hidden = verify_fwd()
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
            upd = [self._tokens_host(ys)] + ([_host_u32([n])] if chained else [])
            copy_host_to_device(upd, device_tensors=[dev_mtp_tok] + ([dev_sel] if chained else []))
            if use_trace:
                ttnn.execute_trace(mesh, trace_mtp, cq_id=0, blocking=False)
                m_idx, m_val = tm_idx, tm_val
            else:
                m_idx, m_val = catchup_fwd(v_hidden)
                ttnn.deallocate(v_hidden)
            ms = combine_argmax_host(m_idx, m_val, mesh, R, per_shard)
            if not use_trace:
                ttnn.deallocate(m_idx)
                ttnn.deallocate(m_val)
            pos += n + 1
            drafts = [ms[n] if draft_override is None else draft_override(len(generated), 0, ms[n])]
            t_mtp = time.time()
            if chained:
                drafts = run_chain(drafts, pos)
            cur = ys[n]
            t_end = time.time()
            step_times.append(t_end - t_step)
            verify_times.append(t_verify - t_step)
            mtp_times.append(t_mtp - t_verify)
            chain_times.append(t_end - t_mtp)
            step_tokens.append(n + 1)

        generated = generated[:max_new_tokens]
        steps = len(step_times)
        accepted = sum(accepted_at)
        # Steady state: drop the first step (as the plain demo drops its first decode step).
        steady_t, steady_n = sum(step_times[1:]), sum(step_tokens[1:])

        def mean_ms(ts):
            return 1000 * sum(ts[1:]) / max(1, len(ts) - 1)

        stats = {
            "k": k,
            "ttft_s": ttft,
            # Prompt length and the time to the first draft (target prefill + MTP prompt pass).
            "prompt_tokens": T,
            "prompt_s": prompt_s,
            "drafts": k * steps,
            "accepted": accepted,
            "acceptance": accepted / (k * steps) if steps else 0.0,
            "steps": steps,
            # Tokens committed per step (accepted drafts + the target's token).
            "mean_accepted_length": sum(step_tokens) / steps if steps else 0.0,
            # Draft slot i is accepted in accepted_per_position[i] of `steps` steps; conditional:
            # among the steps whose slots < i were all accepted.
            "accepted_per_position": accepted_at,
            "acceptance_per_position": [a / steps if steps else 0.0 for a in accepted_at],
            "conditional_acceptance_per_position": [
                accepted_at[i] / (accepted_at[i - 1] if i else steps) if (accepted_at[i - 1] if i else steps) else 0.0
                for i in range(k)
            ],
            "decode_tok_s": steady_n / steady_t if steady_t > 0 else 0.0,
            # Split of the steady step: verify (inputs, trace, argmax readback); MTP catch-up
            # (rollback on reject, tokens, trace, readback); chained drafts (k-1 MTP rows).
            "step_ms_mean": mean_ms(step_times),
            "verify_ms_mean": mean_ms(verify_times),
            "mtp_ms_mean": mean_ms(mtp_times),
            "chain_ms_mean": mean_ms(chain_times),
            "dram": dram,
            "trace_region": trace_region,
        }
        return generated, stats
