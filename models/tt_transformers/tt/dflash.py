# SPDX-License-Identifier: Apache-2.0

"""DFlash speculative decoding for tt_transformers Qwen3 under vllm-tt-plugin.

The drafter is z-lab's DFlash block-diffusion model (``z-lab/Qwen3-8B-DFlash-b16``): five Qwen3
decoder layers fed by the target's residual stream after decoder layers 1, 9, 17, 25 and 33 (the
taps). For every committed target row it keeps context K/V in its own cache,
``K = RoPE(k_norm(k_proj(g)))`` and ``V = v_proj(g)`` per layer with
``g = hidden_norm(fc(concat(taps)))``. A draft is one non-causal pass over the 16-row block
``[last token, MASK x 15]`` at the next 16 positions, attending to the cached context and to
itself; the target's LM head on block rows 1..15 gives 15 drafts.

On the N300 the drafter reuses tt_transformers' tensor-parallel pieces in the target's layout:
``TransformerBlock`` for its layers (fractured residual, heads split across the two ASICs),
``DistributedNorm`` for ``hidden_norm`` and the final norm, and the target's embedding, RoPE
tables, LM head and greedy argmax. ``fc`` is row-parallel over each ASIC's half of every tap,
followed by the same reduce-scatter the decoder blocks use. Context K/V live in a paged cache
with an identity page table.

A propose call is one trace: inject the 16 tap rows of the last target forward (a verify's
candidates or a plain step's row) at that forward's positions, then run the block. Tap rows past
the accepted ones land at positions the block rewrites before anything reads them, so every call
injects all 16 rows. Block rows attend non-causally because every row's SDPA position is the
block's last position, while RoPE and the KV writes use each row's own position; both K/V writes
go one column at a time, as the verify's do (``spec_verify._row_attention``). Prefill injects the
prompt's taps chunk by chunk, eagerly, so the server runs with ``trace_mode: decode_only``.

The verify path is ``spec_verify.SpecVerifyMixin`` with K = 15 (16 decode rows). Enable with
``TT_SPEC_VERIFY_K=15``,
``TT_MODEL_CLASS_OVERRIDES=TTQwen3ForCausalLM=models.tt_transformers.tt.dflash:QwenDFlashForCausalLM``
and ``--speculative-config '{"method": "custom_class", "model": "vllm_tt_plugin.model_owned_drafter",
"num_speculative_tokens": 15}'``; ``TT_DFLASH_DRAFTER`` names another drafter checkpoint.
"""

import json
import math
import os
from contextlib import contextmanager
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from loguru import logger
from safetensors.torch import load_file
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.common.rmsnorm import RMSNorm
from models.tt_transformers.tt.ccl import tt_all_reduce
from models.tt_transformers.tt.common import Mode, copy_host_to_device
from models.tt_transformers.tt.decoder import TransformerBlock
from models.tt_transformers.tt.distributed_norm import DistributedNorm
from models.tt_transformers.tt.generator_vllm import QwenForCausalLM
from models.tt_transformers.tt.load_checkpoints import convert_hf_to_meta, convert_hf_to_meta_no_qkv_permute
from models.tt_transformers.tt.model_config import TensorGroup
from models.tt_transformers.tt.spec_verify import SpecVerifyMixin, _row_attention

DRAFTER_ENV = "TT_DFLASH_DRAFTER"
DEFAULT_DRAFTER = "z-lab/Qwen3-8B-DFlash-b16"


@contextmanager
def _context_only_attention(kv_columns):
    """Make attention layers only write context K/V: SDPA is skipped and its output unused.

    ``kv_columns`` (decode) writes the rows one column at a time; prefill fills the chunk.
    """
    stock = (
        ttnn.transformer.paged_scaled_dot_product_attention_decode,
        ttnn.transformer.scaled_dot_product_attention,
        ttnn.transformer.chunked_scaled_dot_product_attention,
    )

    def skip(*args, **kwargs):
        q = args[0] if args else kwargs["input_tensor_q"]
        return ttnn.clone(q, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    ttnn.transformer.paged_scaled_dot_product_attention_decode = skip
    ttnn.transformer.scaled_dot_product_attention = skip
    ttnn.transformer.chunked_scaled_dot_product_attention = skip
    try:
        if kv_columns is None:
            yield
        else:
            with _row_attention(None, None, live_rows=None, page_rows=None, kv_columns=kv_columns, zero_pad=None):
                yield
    finally:
        (
            ttnn.transformer.paged_scaled_dot_product_attention_decode,
            ttnn.transformer.scaled_dot_product_attention,
            ttnn.transformer.chunked_scaled_dot_product_attention,
        ) = stock


class DFlashDrafter:
    """The DFlash drafter on the target's mesh, layout and shared modules (see the module docstring)."""

    def __init__(self, model, args, repo_id, rows, dtype=ttnn.bfloat8_b):
        self.model = model
        self.args = args
        self.mesh_device = model.mesh_device
        self.rows = rows
        snapshot = Path(
            snapshot_download(repo_id, local_files_only=True, allow_patterns=["config.json", "model.safetensors"])
        )
        config = json.loads((snapshot / "config.json").read_text())
        self._check_config(config, repo_id)
        self.tap_layers = tuple(config["dflash_config"]["target_layer_ids"])
        self.mask_token_id = int(config["dflash_config"]["mask_token_id"])
        self.block_size = int(config["block_size"])
        if self.block_size != rows:
            raise ValueError(f"{repo_id} drafts {self.block_size}-row blocks; the verify was built with {rows} rows")

        raw = load_file(str(snapshot / "model.safetensors"))
        hf = {"model." + k: v for k, v in raw.items() if k.startswith("layers.") or k == "norm.weight"}
        convert = convert_hf_to_meta_no_qkv_permute if args.use_hf_rope else convert_hf_to_meta
        state_dict = convert(hf, args.head_dim, args.n_heads, args.n_kv_heads)
        state_dict["hidden_norm.weight"] = raw["hidden_norm.weight"]
        self.cache_dir = cache = Path(args.weight_cache_path(dtype)) / f"dflash_{snapshot.name}"
        cache.mkdir(parents=True, exist_ok=True)

        self.layers = [
            TransformerBlock(
                args=args,
                mesh_device=self.mesh_device,
                tt_ccl=model.tt_ccl,
                dtype=dtype,
                state_dict=state_dict,
                layer_num=i,
                weight_cache_path=cache,
                transformation_mats=model.trans_mats_dict,
                use_paged_kv_cache=True,
            )
            for i in range(config["num_hidden_layers"])
        ]
        self.hidden_norm = self._norm(state_dict, cache, "hidden_norm")
        self.norm = self._norm(state_dict, cache, "norm")
        self.fc = self._fc(raw["fc.weight"], cache, dtype)
        self.kv = None
        self.taps = None
        logger.info(f"DFlash drafter {repo_id}: {len(self.layers)} layers, taps {self.tap_layers}")

    def _check_config(self, config, repo_id):
        args = self.args
        expected = {
            "hidden_size": args.dim,
            "num_attention_heads": args.n_heads,
            "num_key_value_heads": args.n_kv_heads,
            "head_dim": args.head_dim,
            "rope_theta": args.rope_theta,
            "vocab_size": args.vocab_size,
        }
        mismatched = {k: (config.get(k), v) for k, v in expected.items() if config.get(k) != v}
        if mismatched:
            raise ValueError(f"{repo_id} does not match the target model (drafter, target): {mismatched}")

    def _norm(self, state_dict, cache, key):
        args = self.args
        return DistributedNorm(
            RMSNorm(
                device=self.mesh_device,
                dim=args.dim,
                eps=args.norm_eps,
                state_dict=state_dict,
                weight_cache_path=cache,
                weight_dtype=ttnn.bfloat16,
                weight_key=key,
                is_distributed=args.is_distributed_norm,
                add_unit_offset=args.rms_norm_add_unit_offset,
                ccl_topology=args.ccl_topology(),
                tt_ccl=self.model.tt_ccl,
            ),
            args,
            tt_ccl=self.model.tt_ccl,
            TG=args.is_galaxy,
        )

    def _fc(self, weight, cache, dtype):
        """fc as row-parallel matmuls: each ASIC multiplies its half of every tap's hidden dims."""
        dim, devices = self.args.dim, self.args.num_devices
        taps = len(self.tap_layers)
        w = weight.T.reshape(taps, devices, dim // devices, dim)  # [tap, device, rows, out]
        per_device = w.transpose(0, 1).reshape(devices, taps * dim // devices, dim)
        return ttnn.as_tensor(
            torch.cat(list(per_device), dim=-1).unsqueeze(0).unsqueeze(0),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, 3), mesh_shape=self.args.cluster_shape),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache / f"fc_tp{devices}",
        )

    def allocate_cache(self, target_kv, max_positions):
        """Context K/V per layer, shaped like the target's paged blocks."""
        _, kv_heads, block, head_dim = target_kv.shape
        self.kv_block_size = int(block)
        self.num_blocks = math.ceil((max_positions + 2 * self.block_size) / self.kv_block_size)
        shape = (self.num_blocks, int(kv_heads), self.kv_block_size, int(head_dim))
        self.kv = [
            [
                ttnn.as_tensor(
                    torch.zeros(shape),
                    device=self.mesh_device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=target_kv.dtype,
                    cache_file_name=self.cache_dir / f"empty_{kv}cache{shape}",
                )
                for kv in ("k", "v")
            ]
            for _ in self.layers
        ]
        logger.info(f"DFlash drafter cache: {self.num_blocks} blocks of {self.kv_block_size} positions per layer")

    def identity_pages(self, rows):
        return torch.arange(self.num_blocks, dtype=torch.int32).repeat(rows, 1)

    def _context(self, taps, mode):
        """g = hidden_norm(fc(concat(taps))), replicated like an attention input."""
        args = self.args
        cat = ttnn.concat(taps, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        g = ttnn.linear(cat, self.fc, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(cat)
        g = tt_all_reduce(
            g,
            self.mesh_device,
            self.model.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if mode == Mode.DECODE:
            g = ttnn.to_memory_config(g, args.get_residual_mem_config(Mode.DECODE, None))
        gn = self.hidden_norm(g, mode, norm_config=args.get_norm_config("attn", mode, None))
        ttnn.deallocate(g)
        return gn

    def _inject(self, gn, mode, current_pos, rot_mats, page_table, kv_columns=None, chunk_page_table=None, chunk_start=None):
        with _context_only_attention(kv_columns):
            for i, layer in enumerate(self.layers):
                out = layer.attention.forward(
                    ttnn.clone(gn),
                    current_pos,
                    rot_mats,
                    0,
                    mode,
                    page_table=page_table,
                    chunk_page_table=chunk_page_table,
                    chunk_start_idx=chunk_start,
                    kv_cache=self.kv[i],
                )
                ttnn.deallocate(out)
        ttnn.deallocate(gn)

    def inject_prefill(self, taps, rot_mats, chunk_start):
        """Write context K/V for one eager prefill chunk; ``taps`` maps layer -> [1, 1, S, dim/devices]."""
        gn = self._context([taps[i] for i in self.tap_layers], Mode.PREFILL)
        for t in taps.values():
            ttnn.deallocate(t)
        seq_len = int(gn.shape[-2])
        to_device = lambda t: ttnn.from_torch(
            t,
            device=self.mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        pages = self.identity_pages(1)
        chunk_pages = None
        if chunk_start:
            first = chunk_start // self.kv_block_size
            chunk_pages = to_device(pages[:, first : first + math.ceil(seq_len / self.kv_block_size)].contiguous())
        page_table = to_device(pages)
        self._inject(
            gn,
            Mode.PREFILL,
            None,
            rot_mats,
            page_table,
            chunk_page_table=chunk_pages,
            chunk_start=chunk_start or None,
        )
        for t in (page_table, chunk_pages):
            if t is not None:
                ttnn.deallocate(t)

    def graph(self, inp):
        """Inject the last forward's tap rows, then draft one block; returns the per-row argmax ids."""
        model, args = self.model, self.args
        gn = self._context([self.taps[i] for i in self.tap_layers], Mode.DECODE)
        self._inject(
            gn,
            Mode.DECODE,
            inp["inj_pos"],
            model.rope_setup.get_rot_mats(inp["inj_rope"]),
            inp["page_table"],
            kv_columns=inp["inj_cols"],
        )
        x = model._transform_decode_inputs_device(inp["blk_tokens"])
        rot_mats = model.rope_setup.get_rot_mats(inp["blk_rope"])
        residual_cfg = args.get_residual_mem_config(Mode.DECODE, None)
        with _row_attention(None, None, live_rows=None, page_rows=None, kv_columns=inp["blk_cols"], zero_pad=None):
            for i, layer in enumerate(self.layers):
                dtype = args.decoders_optimizations.get_tensor_dtype(decoder_id=i, tensor=TensorGroup.ACTIVATION)
                x = ttnn.to_memory_config(x, residual_cfg, dtype)
                x = layer(
                    x,
                    inp["blk_pos"],
                    rot_mats_global=rot_mats,
                    mode=Mode.DECODE,
                    page_table=inp["page_table"],
                    kv_cache=self.kv[i],
                )
        x = self.norm(x, Mode.DECODE, norm_config=args.get_norm_config("lm_head", Mode.DECODE, None))
        return model.sampling.tt_sampling.argmax(model.lm_head(x))


class QwenDFlashForCausalLM(SpecVerifyMixin, QwenForCausalLM):
    """Qwen3 with the DFlash drafter as the plugin's model-owned proposer (see the module docstring)."""

    model_capabilities = {
        **QwenForCausalLM.model_capabilities,
        "supports_spec_decode": True,
        "spec_requirements": ("device_propose", "hidden_feed"),
        # The taps stay in device buffers that the next target forward overwrites, so a
        # proposal must run before it: synchronous scheduling only.
        "spec_hidden_handoff": ("on_device",),
        "supports_async_spec_decode": False,
    }

    @classmethod
    def initialize_vllm_model(cls, *args, **kwargs):
        generator = super().initialize_vllm_model(*args, **kwargs)
        generator._init_dflash()
        return generator

    def _init_dflash(self):
        if self.spec_requests != 1:
            raise NotImplementedError("DFlash drafting is single-stream: launch with max_num_seqs=1")
        model, args = self.model[0], self.model_args[0]
        self._drafter = DFlashDrafter(model, args, os.environ.get(DRAFTER_ENV, DEFAULT_DRAFTER), self.spec_rows)
        self._propose_inputs = None
        self._propose_trace = None
        # Persistent tap buffers, one per tapped layer, in the decode residual layout.
        self._drafter.taps = {
            i: ttnn.from_torch(
                torch.zeros(1, 1, args.tile_padded_batch_rows, args.dim),
                device=model.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(model.mesh_device, dims=(None, 3), mesh_shape=args.cluster_shape),
            )
            for i in self._drafter.tap_layers
        }
        self._prefill_taps = None
        for i in self._drafter.tap_layers:
            self._hook_tap(model.layers[i], i)
        stock_prefill = model.ttnn_prefill_forward

        def ttnn_prefill_forward(x, *pargs, **kwargs):
            self._prefill_taps = {}
            try:
                out = stock_prefill(x, *pargs, **kwargs)
                taps, self._prefill_taps = self._prefill_taps, None
            except BaseException:
                self._prefill_taps = None
                raise
            # Every tap layer's call saw the same rotation tables and chunk start.
            meta = taps.pop("meta")
            self._drafter.inject_prefill(taps, meta["rot_mats_global"], meta.get("chunk_start_idx"))
            return out

        model.ttnn_prefill_forward = ttnn_prefill_forward

    def _hook_tap(self, layer, index):
        stock = layer.forward
        drafter = self._drafter

        def forward(x, current_pos, *args, **kwargs):
            out = stock(x, current_pos, *args, **kwargs)
            if kwargs.get("mode") == Mode.DECODE:
                staged = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG, ttnn.bfloat16)
                ttnn.copy(staged, drafter.taps[index])
                ttnn.deallocate(staged)
            elif self._prefill_taps is not None:
                if isinstance(kwargs.get("chunk_start_idx"), ttnn.Tensor):
                    raise NotImplementedError("DFlash taps need eager prefill (trace_mode decode_only)")
                self._prefill_taps[index] = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG, ttnn.bfloat16)
                self._prefill_taps["meta"] = kwargs
            return out

        layer.forward = forward

    def _verify_hidden(self):
        return self._drafter.taps

    # -- KV cache and warmup -----------------------------------------------
    def allocate_kv_cache(self, *args, **kwargs):
        kv_cache = super().allocate_kv_cache(*args, **kwargs)
        self._drafter.allocate_cache(kv_cache[0][0][0], self.model_args[0].max_seq_len)
        return kv_cache

    def warmup_model_decode(self, kv_cache, enable_trace, max_batch_size, num_blocks, can_sample_on_device, **kwargs):
        super().warmup_model_decode(kv_cache, enable_trace, max_batch_size, num_blocks, can_sample_on_device, **kwargs)
        if enable_trace:
            self._record_propose_trace()
        else:
            self._stage_propose()

    def release_persistent_capture(self) -> None:
        trace = getattr(self, "_propose_trace", None)
        self._propose_trace = None
        self._propose_inputs = None
        if trace is not None:
            try:
                ttnn.release_trace(self.model_args[0].mesh_device, trace[0])
            except Exception:
                pass  # as the base class: cleanup errors are ignored
        super().release_persistent_capture()

    # -- propose ---------------------------------------------------------------
    def _propose_host_inputs(self, last_token, block_start, inject_start):
        """Host tensors for one propose call (see DFlashDrafter.graph for their roles)."""
        model, drafter, rows = self.model[0], self._drafter, self.spec_rows
        block_pos = torch.arange(block_start, block_start + rows, dtype=torch.int64)
        inject_pos = torch.arange(inject_start, inject_start + rows, dtype=torch.int64)
        tokens = torch.full((rows,), drafter.mask_token_id, dtype=torch.int64)
        tokens[0] = last_token
        blk_tokens, _, blk_rope, page_table = model.prepare_decode_inputs_host(
            tokens, block_pos, page_table=drafter.identity_pages(rows)
        )

        def columns(pos):
            out = []
            for j in range(rows):
                col = torch.full((rows,), -1, dtype=torch.int64)
                col[j] = pos[j]
                out.append(self._index_tensor(col))
            return out

        return {
            "blk_tokens": blk_tokens,
            "blk_rope": blk_rope,
            # Every block row attends to the whole block: SDPA position = the block's last.
            "blk_pos": self._index_tensor(torch.full((rows,), block_start + rows - 1, dtype=torch.int64)),
            "blk_cols": columns(block_pos),
            "inj_rope": model.rope_setup.get_rot_idxs(inject_pos, on_host=True),
            "inj_pos": self._index_tensor(inject_pos),
            "inj_cols": columns(inject_pos),
            "page_table": page_table,
        }

    @staticmethod
    def _flat(inputs):
        return [t for k in sorted(inputs) for t in (inputs[k] if isinstance(inputs[k], list) else [inputs[k]])]

    def _unflat(self, flat, like):
        out, i = {}, 0
        for k in sorted(like):
            n = len(like[k]) if isinstance(like[k], list) else 1
            out[k] = flat[i : i + n] if isinstance(like[k], list) else flat[i]
            i += n
        return out

    def _stage_propose(self):
        """Warmup phase 1: allocate the propose trace's inputs and compile its programs."""
        if self._propose_inputs is not None:
            return
        self._enter_decode_mode()
        host = self._propose_host_inputs(0, 1, 0)
        flat = copy_host_to_device(self._flat(host), mesh_device=self.model_args[0].mesh_device)
        self._propose_inputs = self._unflat(flat, host)
        out = self._drafter.graph(self._propose_inputs)
        self._read_ids(out)
        ttnn.deallocate(out)
        logger.info("Compiled the DFlash propose graph")

    def _record_propose_trace(self):
        if self._propose_trace is not None:
            return
        if self._propose_inputs is None:
            raise RuntimeError("DFlash propose inputs were not staged in warmup phase 1")
        self._enter_decode_mode()
        mesh_device = self.model_args[0].mesh_device
        with trace_allocation_tracker.corruptible_allocation_scope(mesh_device):
            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            out = self._drafter.graph(self._propose_inputs)
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        self._propose_trace = (trace_id, out)
        logger.info("Captured the DFlash propose trace")

    def propose_draft_tokens(self, num_drafts, committed_tokens, committed_positions, accepted_counts, hidden=None):
        from vllm_tt_plugin.spec_decode import PLACEHOLDER_TOKEN_ID, DraftOutput

        del hidden  # the taps are the drafter's own device buffers
        rows = int(committed_tokens.shape[0])
        drafts = torch.full((rows, num_drafts), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
        num_valid = torch.zeros(rows, dtype=torch.int32)
        live = [b for b in range(rows) if int(committed_positions[b, 0]) >= 0]
        if live != [0]:
            return DraftOutput(draft_token_ids=drafts, num_valid=num_valid)
        accepted = int(accepted_counts[0])
        block_start = int(committed_positions[0, accepted - 1])
        if block_start + self.spec_rows > self.model_args[0].max_seq_len:
            return DraftOutput(draft_token_ids=drafts, num_valid=num_valid)
        # The last forward's row 0 sat one position before this step's first commit.
        host = self._propose_host_inputs(
            int(committed_tokens[0, accepted - 1]), block_start, int(committed_positions[0, 0]) - 1
        )
        self._enter_decode_mode()
        mesh_device = self.model_args[0].mesh_device
        if self._propose_trace is None:
            if self._propose_inputs is None:
                self._stage_propose()
            logger.warning("DFlash propose trace was not captured in warmup; capturing on first use")
            self._record_propose_trace()
        copy_host_to_device(self._flat(host), device_tensors=self._flat(self._propose_inputs))
        trace_id, out = self._propose_trace
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ids = self._read_ids(out)
        drafts[0] = ids[1 : 1 + num_drafts].to(torch.int32)
        num_valid[0] = num_drafts
        return DraftOutput(draft_token_ids=drafts, num_valid=num_valid)
