"""Written by localmaxxing-n300/tt_bench.py: canonical-prompt Qwen3.5/3.6 batches through _run_tp_generation."""
import functools
import json
import os
from pathlib import Path

import pytest
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.demos.blackhole.qwen36.demo.text_demo import BLOCK_SIZE, _blocks_for, _run_tp_generation
from models.demos.blackhole.qwen36.tt import tp_common
from models.demos.blackhole.qwen36.tt.gdn import tp as gdn_tp
from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
from models.tt_transformers.tt.common import Mode

# Wormhole adaptations of the Blackhole-tuned TP path (the N300 compute grid is 8x8; P150 is 11x10).
# All of them touch prefill only; decode runs the upstream code.
# - the fused prefill all-gather + matmul runs on 8x7 instead of 8x9, leaving the device grid's bottom
#   row to the op's fabric mux cores, as the tenth row does on Blackhole, and on one fabric link
#   instead of two: the N300's fabric exposes a single link between its chips (get_num_links == 1),
#   and with two the op hangs. Its worker grid stays 8 wide (grid.x = links x workers per link);
# - the DeltaNet prefill out-projection uses the unfused matmul + all-reduce instead of the fused
#   matmul + reduce-scatter, which hangs on the N300;
# - prefill norms stay distributed (hidden-fractured output) for the fused in-projections to gather.
#   Upstream enables that only for dim > 4096 or MoE models; with the 9B's dim 4096 the norm gathers
#   and the in-projection gathers again (K mismatch);
# - the DeltaNet prefill keeps its in-projection and FIR-conv outputs in DRAM instead of L1 (the
#   in-projection output alone is ~400 KB per core at TP=2 on 64 Wormhole cores) and runs its depthwise
#   conv1d in 4 channel chunks (exact; upstream chunks only the MoE model), so the conv's L1 shards fit
#   below its circular buffers;
# - the full-attention chunked prefill SDPA uses 64-token Q/K chunks instead of 128 so its circular
#   buffers (head_dim 256) fit in Wormhole L1;
# - prefill matmuls that ask for an interleaved L1 output (attention out-projection, MLP down
#   projection) write to DRAM: a [2048, 4096] output is 256 KB per core on 64 cores and collides with
#   the matmul's own circular buffers.
tp_common.all_gather_matmul_prefill = functools.partial(tp_common.all_gather_matmul_prefill, grid=(8, 7))
tp_common.all_gather_swiglu_prefill = functools.partial(tp_common.all_gather_swiglu_prefill, grid=(8, 7))
_upstream_agmm = ttnn.experimental.all_gather_minimal_matmul_async
ttnn.experimental.all_gather_minimal_matmul_async = lambda *args, num_links, num_workers_per_link, **kwargs: (
    _upstream_agmm(*args, num_links=1, num_workers_per_link=num_links * num_workers_per_link, **kwargs)
)
_upstream_chunked_sdpa = ttnn.transformer.chunked_scaled_dot_product_attention
ttnn.transformer.chunked_scaled_dot_product_attention = lambda *args, program_config, **kwargs: _upstream_chunked_sdpa(
    *args,
    program_config=ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=program_config.compute_with_storage_grid_size,
        exp_approx_mode=False,
        q_chunk_size=64,
        k_chunk_size=64,
    ),
    **kwargs,
)
_upstream_linear = ttnn.linear


def _linear(input_tensor, *args, memory_config=None, **kwargs):
    if memory_config == ttnn.L1_MEMORY_CONFIG and input_tensor.shape[-2] > ttnn.TILE_SIZE:
        memory_config = ttnn.DRAM_MEMORY_CONFIG
    return _upstream_linear(input_tensor, *args, memory_config=memory_config, **kwargs)


ttnn.linear = _linear
_upstream_is_distributed_norm = Qwen36ModelArgs.is_distributed_norm
Qwen36ModelArgs.is_distributed_norm = lambda self, mode: (
    (self.is_multichip and mode == Mode.PREFILL) or _upstream_is_distributed_norm(self, mode)
)
_upstream_project_qkvzab = TPGatedDeltaNet._project_qkvzab
TPGatedDeltaNet._project_qkvzab = lambda self, x, S, out_mc=None: (
    _upstream_project_qkvzab(self, x, S, out_mc=None if S > ttnn.TILE_SIZE else out_mc)
)
# Masked (shorter-than-chunk) prefill runs the FIR conv instead of conv1d.
_upstream_causal_conv1d_fir = gdn_tp._causal_conv1d_fir
gdn_tp._causal_conv1d_fir = lambda *args, memory_config=None, **kwargs: (
    _upstream_causal_conv1d_fir(*args, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kwargs)
)
_upstream_gdn_init = TPGatedDeltaNet.__init__
_upstream_init_tp_config = Qwen36ModelArgs._init_tp_config


def _gdn_init(self, *args, **kwargs):
    _upstream_gdn_init(self, *args, **kwargs)
    self._fuse_out_mmrs_prefill = False


def _init_tp_config(self, mesh_device):
    _upstream_init_tp_config(self, mesh_device)
    self.gdn_conv_channel_chunks = 4


TPGatedDeltaNet.__init__ = _gdn_init
Qwen36ModelArgs._init_tp_config = _init_tp_config

# Knobs for checkpoints bigger than the 9B (set per model in tt_bench's MODEL_ENV; all default off):
# - QWEN36_GATEUP_AGMM=0 drops the prefill-only packed [gate|up] copy of every MLP layer (~w1+w3 of
#   extra DRAM); prefill then gathers after ff_norm and runs the unfused gate/up matmuls;
# - QWEN36_DTYPE_OVERRIDES="<cache-name substring>=<ttnn dtype>,..." changes the format of matching
#   tp_common.shard_w weights (the cache file name carries the dtype, so caches stay consistent);
# - QWEN36_TRACE_REGION_MB sets the per-device trace region (default 1024);
# - QWEN36_PREFILL_CHUNK=<n> captures and replays the chunked prefill trace with n-token chunks
#   instead of 2048, and warms only the masked tail buckets below n. Upstream sizes prefill for
#   2048-token chunks on 4-8 Blackhole devices; with the 27B at TP=2 on Wormhole's smaller L1, the
#   2048-row DeltaNet norm outputs, attention out-projection, and MLP matmul blocks overflow it;
# - QWEN36_MLP_PREFILL_ROWS=<n> runs the TP MLP prefill in n-row slices. Without the packed gate/up
#   copy, prefill takes the unfused 2D matmul path, whose program config gives each Wormhole core a
#   68-tile-wide gate/up block (4 of the 8 grid columns), so its circular buffers scale with the rows.
if os.environ.get("QWEN36_GATEUP_AGMM") == "0":
    tp_common.mlp_gateup_agmm_enabled = lambda *args, **kwargs: False
_DTYPE_OVERRIDES = [
    (key, getattr(ttnn, value))
    for key, value in (item.split("=", 1) for item in os.environ.get("QWEN36_DTYPE_OVERRIDES", "").split(",") if item)
]
if _DTYPE_OVERRIDES:
    _upstream_shard_w = tp_common.shard_w

    def _shard_w(torch_tensor, mesh, dim, memory_config, cache_path, dtype=ttnn.bfloat8_b):
        dtype = next((value for key, value in _DTYPE_OVERRIDES if cache_path and key in str(cache_path)), dtype)
        return _upstream_shard_w(torch_tensor, mesh, dim, memory_config, cache_path, dtype=dtype)

    tp_common.shard_w = _shard_w

_PREFILL_CHUNK = int(os.environ.get("QWEN36_PREFILL_CHUNK", "0"))
if _PREFILL_CHUNK:
    _upstream_capture_prefill = Qwen36Model.capture_prefill_trace_chunked

    def _capture_prefill(self, device, page_table, chunk_size=2048, **kwargs):
        return _upstream_capture_prefill(self, device, page_table, chunk_size=_PREFILL_CHUNK, **kwargs)

    Qwen36Model.capture_prefill_trace_chunked = _capture_prefill
    Qwen36Model._PREFILL_MASK_BUCKETS = tuple(b for b in Qwen36Model._PREFILL_MASK_BUCKETS if b <= _PREFILL_CHUNK)

_MLP_PREFILL_ROWS = int(os.environ.get("QWEN36_MLP_PREFILL_ROWS", "0"))
if _MLP_PREFILL_ROWS:
    from models.demos.blackhole.qwen36.tt.mlp import Qwen36MLP

    _upstream_mlp_forward_tp = Qwen36MLP._forward_tp

    def _mlp_forward_tp(self, x):
        rows = x.shape[-2]
        if rows <= _MLP_PREFILL_ROWS:
            return _upstream_mlp_forward_tp(self, x)
        lead = tuple(x.shape)[:-2]
        pieces = []
        for start in range(0, rows, _MLP_PREFILL_ROWS):
            stop = min(start + _MLP_PREFILL_ROWS, rows)
            piece = ttnn.slice(x, (0,) * len(lead) + (start, 0), lead + (stop, x.shape[-1]),
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
            pieces.append(_upstream_mlp_forward_tp(self, piece))
            ttnn.deallocate(piece)
        out = ttnn.concat(pieces, dim=-2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for piece in pieces:
            ttnn.deallocate(piece)
        return out

    Qwen36MLP._forward_tp = _mlp_forward_tp

DEVICE_PARAMS = {
    "l1_small_size": 24576,
    "num_command_queues": 2,
    "fabric_config": ttnn.FabricConfig.FABRIC_1D,
    "trace_region_size": int(os.environ.get("QWEN36_TRACE_REGION_MB", "1024")) << 20,
}


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_canonical_prompt(mesh_device):
    prompt = json.loads(Path(os.environ["TT_BENCH_PROMPT_JSON"]).read_text())[0]["prompt"]
    max_new_tokens = int(os.environ["TT_BENCH_MAX_TOKENS"])
    tokenizer = AutoTokenizer.from_pretrained(os.environ["HF_MODEL"])
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    token_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    logger.info(f"Encoded prompt lengths:{token_ids.shape[1]}")
    mesh_device.enable_program_cache()
    spec = os.environ.get("QWEN36_SPEC", "")
    # DFlash2's 8-row verify can write KV up to 7 positions past the last generated token.
    num_blocks = _blocks_for(token_ids.shape[1], max_new_tokens + (16 if spec == "dflash2" else 0))
    # QWEN36_CONTEXT_TOKENS: allocate the paged KV for this context window (LocalMaxxing contextLength)
    # instead of just prompt + output; rounded up to 32 blocks like the decode paths do.
    context = int(os.environ.get("QWEN36_CONTEXT_TOKENS", "0"))
    num_blocks = ((max(num_blocks, -(-context // BLOCK_SIZE)) + 31) // 32) * 32
    logger.info(f"TT_BENCH context: tokens={num_blocks * BLOCK_SIZE}")
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=num_blocks * BLOCK_SIZE)
    # QWEN36_HOST_EMBEDDING=1 (tt-metal branch qwen36-n300-mtp): the token embedding table lives on the host
    # (frees 1.21 GiB per ASIC; needed for a 128K window). Every TP path then takes embedded rows.
    host_embedding = os.environ.get("QWEN36_HOST_EMBEDDING") == "1"
    decoder = None
    if spec.startswith("mtp"):
        # tt-metal branch qwen36-n300-mtp: the checkpoint's MTP head drafts k chained tokens per step and a
        # (k+1)-row verify step accepts the longest prefix equal to the target's greedy tokens (exact
        # greedy; the DeltaNet state rolls back to the last accepted row).
        from models.demos.blackhole.qwen36.tt.spec.mtp import MTPHead, load_mtp_state_dict
        from models.demos.blackhole.qwen36.tt.spec.mtpk import MTPDecoder

        head = MTPHead(model, load_mtp_state_dict(model.args.CKPT_DIR), model.args.weight_cache_path())
        decoder = MTPDecoder(model, head, int(spec[3:]), host_embedding=host_embedding)
    elif spec == "dflash2":
        # Same branch: the z-lab DFlash2 drafter (block 8, target hidden taps) drafts 7 tokens per step and an
        # 8-row verify step accepts the longest matching prefix. Needs QWEN36_TRACE_REGION_MB=128 (set by tt_bench).
        from huggingface_hub import snapshot_download
        from models.demos.blackhole.qwen36.tt.spec.dflash2 import DFlash2Drafter, load_dflash2_checkpoint
        from models.demos.blackhole.qwen36.tt.spec.dflash2_spec import DFlash2Decoder

        drafter_dir = snapshot_download(os.environ["QWEN36_SPEC_DRAFTER"], local_files_only=True)
        config, state_dict = load_dflash2_checkpoint(drafter_dir)
        drafter = DFlash2Drafter(model, config, state_dict, model.args.weight_cache_path())
        decoder = DFlash2Decoder(model, drafter, host_embedding=host_embedding)
    elif host_embedding:
        from models.demos.blackhole.qwen36.tt.spec.host_embedding import enable_host_embedding

        enable_host_embedding(model)
    for batch in range(int(os.environ["TT_BENCH_BATCHES"])):
        logger.info(f"Processing batch {batch}")
        if decoder is None:
            generated, perf = _run_tp_generation(model, tokenizer, token_ids, max_new_tokens, num_blocks)
            model.free_kv_caches()
        else:
            generated, perf = decoder.generate(token_ids, max_new_tokens, num_blocks)  # frees its KV caches
            logger.info(f"TT_BENCH spec: drafts={perf['drafts']} accepted={perf['accepted']} steps={perf['steps']}")
        logger.info(
            f"TT_BENCH batch summary: ttft_ms={perf['ttft_s'] * 1000:.2f} "
            f"decode_tok_s={perf['decode_tok_s']:.3f} output_tokens={len(generated)}"
        )
        output = tokenizer.decode(generated, skip_special_tokens=True)
        logger.info(f"\n==REPEAT BATCH {batch}\n==USER 0 - PROMPT\n(canonical reasoning-v1)\n==USER 0 - OUTPUT\n{output.strip()}\n==END OUTPUT\n")
