# SPDX-License-Identifier: Apache-2.0

"""Speculative verification for tt_transformers decode under vllm-tt-plugin.

Implements the verify call of the plugin's SPEC_DECODE_CONTRACT.md (``argmax_ids`` mode) for
a draft method whose proposer lives outside the model, such as vLLM's n-gram proposer.

A verify step checks a ``[B, 1+K]`` candidate block in one decode forward. Every candidate is
its own decode row: row ``r = b * W + j`` (``W = 1 + K``) carries request b's column-j token
at its position and reads request b's page-table row. The model is built with
``max_batch_size`` R, the request rows rounded up to a batch tt_transformers supports, so the
verify is the ordinary decode graph: matmuls, norms, CCLs, LM head and argmax treat rows
independently, and R rows pad to the same 32-row tile as one. Two ops are swapped while a
decode graph is built (the compile pass and the trace capture; replays run what was recorded):

- KV write. ``paged_fused_update_cache`` gives each row a core that reads the row's 32-row
  cache tile, patches one row and writes the whole tile back, so rows of one request that
  share a tile race inside one call. A verify writes once per column, every other row at -1.
- Attention. Paged SDPA decode splits a row's context over cores / B, so a row of a B-row call
  reduces in a different order than a B=1 step (with the stock call, Qwen3-8B greedy text left
  the single-stream text within 20 tokens of a 257-token prompt). Each live row runs its own
  B=1 SDPA, the core split of a single-stream decode, and the rows are concatenated. Plain
  decode steps of the R-row model run their live rows the same way, so plain steps, verify rows
  and a server without speculation compute the same numbers. ``TT_SPEC_VERIFY_SDPA=batched``
  keeps the stock B-row call: a cheaper verify, but no longer that server's text.

The verify has its own trace and persistent inputs: staged (compiled, inputs allocated) in
warmup phase 1 and recorded in phase 2, after the plain decode traces. Its argmax is
TTSampling's greedy path, so a verify row picks what a greedy decode step picks.

Enable with ``TT_SPEC_VERIFY_K=<K>`` and route the architecture here, e.g.
``TT_MODEL_CLASS_OVERRIDES=TTQwen3ForCausalLM=models.tt_transformers.tt.spec_verify:QwenSpecForCausalLM``
with ``--speculative-config '{"method": "ngram", "num_speculative_tokens": <K>, ...}'``.
"""

import os
from contextlib import contextmanager

import torch
from loguru import logger
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.tt_transformers.tt.common import Mode, copy_host_to_device
from models.tt_transformers.tt.generator_vllm import QwenForCausalLM

SPEC_K_ENV = "TT_SPEC_VERIFY_K"
VERIFY_SDPA_ENV = "TT_SPEC_VERIFY_SDPA"
# ModelArgs accepts these decode batch sizes only.
SUPPORTED_ROWS = (1, 2, 4, 8, 16, 32)


def configured_k() -> int:
    """The draft length the verify rows are built for (0 when speculation is off)."""
    raw = os.environ.get(SPEC_K_ENV, "").strip()
    if not raw:
        return 0
    k = int(raw)
    if k < 1:
        raise ValueError(f"{SPEC_K_ENV}={raw} must be a positive draft length")
    return k


def rows_for(requests: int, k: int) -> int:
    """Decode rows for ``requests`` requests of ``1 + k`` candidates each."""
    need = requests * (1 + k)
    for rows in SUPPORTED_ROWS:
        if rows >= need:
            return rows
    raise ValueError(
        f"{requests} requests x {1 + k} candidate rows = {need} decode rows, "
        f"above the largest supported decode batch {SUPPORTED_ROWS[-1]}"
    )


def _sdpa_mode() -> bool:
    """True for one B=1 SDPA per live row, False for the stock batched call."""
    mode = os.environ.get(VERIFY_SDPA_ENV, "rows").strip()
    if mode not in ("rows", "batched"):
        raise ValueError(f"{VERIFY_SDPA_ENV}={mode!r} must be 'rows' or 'batched'")
    return mode == "rows"


@contextmanager
def _row_attention(current_pos, page_table, *, live_rows, page_rows, kv_columns, zero_pad):
    """Swap the paged KV write and paged SDPA decode while one decode graph is built.

    live_rows: rows that get their own B=1 SDPA, or None to keep the stock batched call.
    page_rows: for each live row, the page-table row its SDPA reads.
    kv_columns: per-column update-index tensors (verify), or None for the stock write.
    zero_pad: zeros completing the R-row SDPA output after the live rows, or None.
    """
    sdpa_stock = ttnn.transformer.paged_scaled_dot_product_attention_decode
    fused_stock = ttnn.experimental.paged_fused_update_cache
    update_stock = ttnn.experimental.paged_update_cache
    owned = []
    try:
        if live_rows is not None:
            # Sliced once per graph, not per layer: every layer reads the same rows.
            row_pos = [ttnn.slice(current_pos, (r,), (r + 1,)) for r in live_rows]
            width = page_table.shape[-1]
            pages = {p: ttnn.slice(page_table, (p, 0), (p + 1, width)) for p in sorted(set(page_rows))}
            owned = row_pos + list(pages.values())

            def sdpa(q, keys, values, *, page_table_tensor, cur_pos_tensor, memory_config=None, **kwargs):
                del page_table_tensor, cur_pos_tensor  # replaced by the per-row slices
                q_rows = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG) if q.is_sharded() else q
                heads, head_dim = q.shape[2], q.shape[3]
                # Slice every row first so the SDPA launches run back to back as one program.
                q_slices = [ttnn.slice(q_rows, (0, r, 0, 0), (1, r + 1, heads, head_dim)) for r in live_rows]
                if q_rows is not q:
                    ttnn.deallocate(q_rows)
                outs = [
                    sdpa_stock(
                        q_r,
                        keys,
                        values,
                        page_table_tensor=pages[page_rows[i]],
                        cur_pos_tensor=row_pos[i],
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        **kwargs,
                    )
                    for i, q_r in enumerate(q_slices)
                ]
                for q_r in q_slices:
                    ttnn.deallocate(q_r)
                pieces = outs + ([zero_pad] if zero_pad is not None else [])
                if len(pieces) == 1:
                    return pieces[0]
                out = ttnn.concat(pieces, dim=1, memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG)
                for t in outs:
                    ttnn.deallocate(t)
                return out

            ttnn.transformer.paged_scaled_dot_product_attention_decode = sdpa

        if kv_columns is not None:

            def per_column(stock):
                def write(*args, update_idxs_tensor, **kwargs):
                    del update_idxs_tensor  # one call per column instead
                    for idxs in kv_columns:
                        stock(*args, update_idxs_tensor=idxs, **kwargs)

                return write

            ttnn.experimental.paged_fused_update_cache = per_column(fused_stock)
            ttnn.experimental.paged_update_cache = per_column(update_stock)
        yield
    finally:
        ttnn.transformer.paged_scaled_dot_product_attention_decode = sdpa_stock
        ttnn.experimental.paged_fused_update_cache = fused_stock
        ttnn.experimental.paged_update_cache = update_stock
        for t in owned:
            ttnn.deallocate(t)


class SpecVerifyMixin:
    """Verify call, row-exact attention and verify trace for a tt_transformers vLLM generator."""

    spec_k = 0
    spec_requests = None
    spec_rows = None

    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs, requested_k):
        from vllm_tt_plugin.spec_decode import ACCEPT_MODE_ARGMAX_IDS, DRAFTER_STATE_INTERNAL, SpecPlan, SpecReject

        del vllm_config  # the verify rows depend only on the request count and K
        k = configured_k()
        if k < 1:
            return SpecReject(reason=f"{SPEC_K_ENV} is unset; the verify rows are built when the model loads")
        if requested_k < 1:
            return SpecReject(reason=f"a draft length of {requested_k} speculates nothing", supported_k=(k,))
        try:
            rows_for(max_num_seqs, k)
        except ValueError as e:
            return SpecReject(reason=str(e))
        return SpecPlan(
            effective_k=min(requested_k, k),
            lanes_per_request=1 + k,
            # Verify KV lands in the request's own paged blocks; the verify trace's inputs
            # are a few hundred bytes per row.
            extra_bytes_per_seq=0,
            extra_bytes_per_token=0,
            accept_modes=(ACCEPT_MODE_ARGMAX_IDS,),
            drafter_state=DRAFTER_STATE_INTERNAL,
            # A step without drafts is a plain decode: its sampled rows keep their
            # temperature, and it costs one row of attention, not 1+K.
            supports_narrow_decode=True,
        )

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len, *args, **kwargs):
        k = configured_k()
        rows = rows_for(max_batch_size, k) if k else max_batch_size
        generator = super().initialize_vllm_model(hf_config, mesh_device, rows, max_seq_len, *args, **kwargs)
        generator._init_spec(requests=max_batch_size, k=k)
        return generator

    def _init_spec(self, requests, k):
        self.spec_k = k
        self.spec_requests = requests
        self.spec_rows = rows_for(requests, k) if k else requests
        self._per_row_sdpa = _sdpa_mode()
        self._verify_inputs = None
        self._verify_trace = None
        self._verify_kv_columns = None
        if not k:
            return
        if self.data_parallel != 1:
            raise NotImplementedError("speculative verify supports one data-parallel replica")
        model = self.model[0]
        if model.sampling is None:
            raise NotImplementedError("speculative verify needs the on-device sampler for its argmax")
        attention = model.layers[0].attention
        width = k + 1
        self._plain_pad = self._zero_rows(self.spec_rows - requests, attention.n_local_heads, attention.head_dim)
        self._verify_pad = self._zero_rows(
            self.spec_rows - requests * width, attention.n_local_heads, attention.head_dim
        )
        if requests == 1:
            # One request slot never prefills a batch. Left alone, the R decode rows would
            # warm batched prefill traces up to R users, spending trace region on shapes
            # no request reaches.
            self.model_args[0].disable_batched_prefill = True
        stock = model.ttnn_decode_forward

        def ttnn_decode_forward(
            x, current_pos, rot_mat_idxs=None, page_table=None, kv_cache=None, on_device_logits=False, **kwargs
        ):
            plan = self._attention_plan()
            call = dict(
                rot_mat_idxs=rot_mat_idxs,
                page_table=page_table,
                kv_cache=kv_cache,
                on_device_logits=on_device_logits,
                **kwargs,
            )
            if plan is None or page_table is None:
                return stock(x, current_pos, **call)
            with _row_attention(current_pos, page_table, **plan):
                return stock(x, current_pos, **call)

        model.ttnn_decode_forward = ttnn_decode_forward
        logger.info(
            f"Speculative verify: K={k}, {requests} request(s) x {width} candidate rows in a "
            f"{self.spec_rows}-row decode, SDPA {'per row' if self._per_row_sdpa else 'batched'}"
        )

    def _zero_rows(self, rows, heads, head_dim):
        """Persistent zeros for the SDPA output rows no live row fills."""
        if rows <= 0:
            return None
        mesh_device = self.model_args[0].mesh_device
        return ttnn.from_torch(
            torch.zeros(1, rows, heads, head_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def _attention_plan(self):
        """How the decode graph being built swaps its attention ops, or None for stock."""
        if self._verify_kv_columns is not None:
            width = self.spec_k + 1
            live = self.spec_requests * width
            return dict(
                live_rows=list(range(live)) if self._per_row_sdpa else None,
                page_rows=[(r // width) * width for r in range(live)],
                kv_columns=self._verify_kv_columns,
                zero_pad=self._verify_pad,
            )
        if not self._per_row_sdpa:
            return None
        live = self.spec_requests
        return dict(
            live_rows=list(range(live)),
            page_rows=list(range(live)),
            kv_columns=None,
            zero_pad=self._plain_pad,
        )

    # -- decode -------------------------------------------------------------
    def decode_forward(self, *args, **kwargs):
        if not self.spec_k:
            return super().decode_forward(*args, **kwargs)
        kwargs.update(zip(("tokens", "start_pos", "page_table", "kv_cache"), args))
        spec_mode = kwargs.pop("spec_mode", None)
        num_valid = kwargs.pop("num_valid_drafts", None)
        accepted = kwargs.pop("accepted_counts", None)
        if spec_mode is None:
            return super().decode_forward(**self._pad_plain(kwargs))
        return self._verify(spec_mode, num_valid, accepted, **kwargs)

    def _pad_plain(self, kwargs):
        """Pad a plain decode's request rows to the model's R rows (position -1, null page)."""
        tokens, start_pos = kwargs["tokens"], kwargs["start_pos"]
        pad = self.spec_rows - tokens.shape[0]
        if pad > 0:
            kwargs["tokens"] = torch.cat([tokens, tokens.new_zeros((pad, *tokens.shape[1:]))])
            kwargs["start_pos"] = torch.cat([start_pos, start_pos.new_full((pad, *start_pos.shape[1:]), -1)])
            page_table = kwargs.get("page_table")
            if page_table is not None:
                kwargs["page_table"] = torch.cat([page_table, page_table.new_zeros((pad, page_table.shape[1]))])
        return kwargs

    def read_decode_output(self, tt_out, async_read=False, *args, **kwargs):
        # A verify answers with host ids; nothing to read.
        if isinstance(tt_out, torch.Tensor):
            return (tt_out, []) if async_read else tt_out
        return super().read_decode_output(tt_out, async_read, *args, **kwargs)

    def _verify(
        self,
        spec_mode,
        num_valid,
        accepted,
        *,
        tokens,
        start_pos,
        page_table,
        kv_cache,
        enable_trace=True,
        slot_remap=None,
        **_ignored,  # sampling, readback and reload commands: a verify reloads every input
    ):
        from vllm_tt_plugin.spec_decode import (
            ACCEPT_MODE_ARGMAX_IDS,
            PLACEHOLDER_TOKEN_ID,
            VerifyOutput,
            check_spec_side_tensors,
        )

        if spec_mode != ACCEPT_MODE_ARGMAX_IDS:
            raise NotImplementedError(f"{type(self).__name__} verifies in {ACCEPT_MODE_ARGMAX_IDS!r}, not {spec_mode!r}")
        requests, cols = tokens.shape
        check_spec_side_tensors(num_valid, accepted, int(requests), int(cols) - 1)
        if cols > self.spec_k + 1 or requests > self.spec_requests:
            raise ValueError(
                f"a [{requests}, {cols}] candidate block exceeds the verify rows built for "
                f"{self.spec_requests} request(s) x {self.spec_k + 1} columns"
            )
        self._enter_decode_mode()
        host = self._verify_host_inputs(tokens, start_pos, page_table, num_valid)
        row_ids = self._run_verify(host, kv_cache, enable_trace)
        width = self.spec_k + 1
        ids = torch.full((requests, cols), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
        for b in range(requests):
            if int(start_pos[b, 0]) < 0:
                continue  # padding request
            n = int(num_valid[b])
            ids[b, : n + 1] = row_ids[b * width : b * width + n + 1].to(torch.int32)
        self._apply_sampling_slot_remap(slot_remap)
        return VerifyOutput(spec_mode=spec_mode, argmax_ids=ids, hidden=None)

    def _enter_decode_mode(self):
        if self.mode != Mode.DECODE:
            self.mode = Mode.DECODE
        for model in self.model:
            model.switch_mode(Mode.DECODE)

    def _index_tensor(self, values):
        """A host int32 index vector laid out as the model's decode positions are."""
        args = self.model_args[0]
        return ttnn.from_torch(
            values,
            device=None,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(args.mesh_device, dims=(None, None), mesh_shape=args.cluster_shape),
        )

    def _verify_host_inputs(self, tokens, start_pos, page_table, num_valid):
        """Host tensors for one verify: the decode inputs of the R rows, then one KV index per column."""
        width = self.spec_k + 1
        rows = self.spec_rows
        requests, cols = tokens.shape
        row_tokens = torch.zeros(rows, dtype=torch.int64)
        row_pos = torch.full((rows,), -1, dtype=torch.int64)
        row_pages = torch.zeros(rows, page_table.shape[1], dtype=torch.int32)
        for b in range(requests):
            if int(start_pos[b, 0]) < 0:
                continue  # padding request: no position, null page
            n = min(int(num_valid[b]), cols - 1)
            r0 = b * width
            row_tokens[r0 : r0 + n + 1] = tokens[b, : n + 1]
            row_pos[r0 : r0 + n + 1] = start_pos[b, : n + 1]
            row_pages[r0 : r0 + width] = page_table[b]
        host = list(self.model[0].prepare_decode_inputs_host(row_tokens, row_pos, page_table=row_pages))
        for j in range(width):
            column = torch.full((rows,), -1, dtype=torch.int64)
            column[j : requests * width : width] = row_pos[j : requests * width : width]
            host.append(self._index_tensor(column))
        return host

    def _verify_graph(self, device_inputs, kv_cache):
        """One verify forward on device inputs; returns the per-row argmax ids (device)."""
        tokens, current_pos, rope_idxs, page_table, *kv_columns = device_inputs
        model = self.model[0]
        self._verify_kv_columns = kv_columns
        try:
            logits = model.ttnn_decode_forward(
                tokens,
                current_pos,
                rot_mat_idxs=rope_idxs,
                page_table=page_table,
                kv_cache=kv_cache[0] if kv_cache is not None else None,
                on_device_logits=True,
            )
        finally:
            self._verify_kv_columns = None
        return model.sampling.tt_sampling.argmax(logits)

    def _read_ids(self, out):
        host = out.cpu(blocking=True)
        return ttnn.to_torch(ttnn.get_device_tensors(host)[0]).reshape(-1)[: self.spec_rows].to(torch.int64)

    def _stage_verify(self, kv_cache, num_blocks):
        """Warmup phase 1: allocate the verify trace's inputs and compile its programs."""
        if self._verify_inputs is not None:
            return
        self._enter_decode_mode()
        width = self.spec_k + 1
        requests = self.spec_requests
        host = self._verify_host_inputs(
            torch.zeros(requests, width, dtype=torch.int32),
            torch.arange(width, dtype=torch.int32).repeat(requests, 1),
            torch.zeros(requests, num_blocks, dtype=torch.int32),
            torch.full((requests,), width - 1, dtype=torch.int32),
        )
        self._verify_inputs = copy_host_to_device(host, mesh_device=self.model_args[0].mesh_device)
        out = self._verify_graph(self._verify_inputs, kv_cache)
        self._read_ids(out)
        ttnn.deallocate(out)
        logger.info("Compiled the speculative verify")

    def _record_verify_trace(self, kv_cache):
        """Warmup phase 2: capture the verify trace on the staged inputs."""
        if self._verify_trace is not None:
            return
        if self._verify_inputs is None:
            raise RuntimeError("verify inputs were not staged in warmup phase 1 (warmup_model_decode without trace)")
        self._enter_decode_mode()
        mesh_device = self.model_args[0].mesh_device
        with trace_allocation_tracker.corruptible_allocation_scope(mesh_device):
            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            out = self._verify_graph(self._verify_inputs, kv_cache)
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        self._verify_trace = (trace_id, out)
        logger.info("Captured the speculative verify trace")

    def _run_verify(self, host, kv_cache, enable_trace):
        mesh_device = self.model_args[0].mesh_device
        if not enable_trace:
            device_inputs = copy_host_to_device(host, mesh_device=mesh_device)
            out = self._verify_graph(device_inputs, kv_cache)
            ids = self._read_ids(out)
            ttnn.deallocate(out)
            for t in device_inputs:
                ttnn.deallocate(t)
            return ids
        if self._verify_trace is None:
            # Unwarmed fallback, as Generator._capture_decode_trace_text: safe only while no
            # other trace is live.
            logger.warning("Speculative verify trace was not captured in warmup; capturing on first use")
            if self._verify_inputs is None:
                self._verify_inputs = copy_host_to_device(host, mesh_device=mesh_device)
                ttnn.deallocate(self._verify_graph(self._verify_inputs, kv_cache))
            self._record_verify_trace(kv_cache)
        copy_host_to_device(host, device_tensors=self._verify_inputs)
        trace_id, out = self._verify_trace
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        return self._read_ids(out)

    # -- warmup -------------------------------------------------------------
    def warmup_model_decode(self, kv_cache, enable_trace, max_batch_size, num_blocks, can_sample_on_device, **kwargs):
        if not self.spec_k:
            return super().warmup_model_decode(
                kv_cache, enable_trace, max_batch_size, num_blocks, can_sample_on_device, **kwargs
            )
        # Plain decode runs R rows (see _pad_plain), so its traces are R rows wide.
        super().warmup_model_decode(kv_cache, enable_trace, self.spec_rows, num_blocks, can_sample_on_device, **kwargs)
        if enable_trace:
            self._record_verify_trace(kv_cache)
        else:
            self._stage_verify(kv_cache, num_blocks)

    def release_persistent_capture(self) -> None:
        trace = getattr(self, "_verify_trace", None)
        self._verify_trace = None
        self._verify_inputs = None
        if trace is not None:
            try:
                ttnn.release_trace(self.model_args[0].mesh_device, trace[0])
            except Exception:
                pass  # as the base class: cleanup errors are ignored
        super().release_persistent_capture()


class QwenSpecForCausalLM(SpecVerifyMixin, QwenForCausalLM):
    """Qwen (tt_transformers) with the speculative verify call; see the module docstring."""

    model_capabilities = {
        **QwenForCausalLM.model_capabilities,
        "supports_spec_decode": True,
        # Host proposers (n-gram) need nothing from the model.
        "spec_requirements": (),
        "spec_hidden_handoff": (),
        # The verify returns host ids, which read_decode_output passes through, and no
        # hidden handle.
        "supports_async_spec_decode": True,
    }
