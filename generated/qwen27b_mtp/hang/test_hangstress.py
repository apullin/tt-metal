"""Hang stress (throwaway, gitignored): repeated long-prompt prefills in one process.

Loads the 27B with tt_bench's QWEN36_TEST patches (via test_spec.py), host embedding, the paged KV
for HS_CONTEXT tokens (bfp8 under QWEN_SDPA_BF8=1) and, with HS_MTP=1, the MTP head plus its KV
(the memory layout of the MTP-3 runs that hung). Then runs Qwen36Model.prefill_chunked_collect over
the first HS_TOKENS tokens of HS_PROMPT, HS_REPS times (HS_MTP=1: with the MTP prompt pass per chunk,
as MTPDecoder does). HS_PATH=traced replays the plain path's chunk trace instead
(prefill_traced_chunked). HS_FREE_MB=<n> then allocates DRAM filler until about n MiB stay free
(the MTP decoder's verify buffers leave ~50-110 MiB at a 128K window). Logs per-rep time; writes
hang/results-<HS_TAG>.json. The MMIO-chip telemetry sampler (telem.py) runs during the test body:
hang/telem-<HS_TAG>.jsonl.
"""
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("qwen27b_test_spec", HERE.parent / "test_spec.py")
ts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ts)

import ttnn  # noqa: E402
from models.demos.blackhole.qwen36.demo.text_demo import BLOCK_SIZE, _blocks_for  # noqa: E402
from models.demos.blackhole.qwen36.tt.model import Qwen36Model  # noqa: E402
from models.demos.blackhole.qwen36.tt.spec.host_embedding import enable_host_embedding  # noqa: E402


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("device_params", [ts._ns["DEVICE_PARAMS"]], indirect=True)
def test_hangstress(mesh_device):
    context = int(os.environ.get("HS_CONTEXT", "131072"))
    prompt = os.environ.get("HS_PROMPT", "longprompt-65536.pt")
    n_tok = int(os.environ.get("HS_TOKENS", "0"))
    reps = int(os.environ.get("HS_REPS", "4"))
    use_mtp = os.environ.get("HS_MTP", "1") == "1"
    path = os.environ.get("HS_PATH", "eager")
    tag = os.environ.get("HS_TAG", "hs")
    telem = subprocess.Popen(
        ["/home/pullin/.tenstorrent-venv/bin/python", str(HERE / "telem.py"), str(HERE / f"telem-{tag}.jsonl"), "1"],
        stdout=subprocess.DEVNULL,
        stderr=open(HERE / f"telem-{tag}.err", "w"),
    )
    ids = torch.load(HERE.parent / prompt, weights_only=False)["ids"]
    if n_tok:
        ids = ids[:, :n_tok]
    T = ids.shape[1]
    mesh_device.enable_program_cache()
    num_blocks = ((max(_blocks_for(T, 16), -(-context // BLOCK_SIZE)) + 31) // 32) * 32
    res = {"tag": tag, "T": T, "context_tokens": num_blocks * BLOCK_SIZE, "reps_planned": reps, "mtp": use_mtp,
           "path": path, "env": {k: v for k, v in os.environ.items() if k.startswith(("QWEN", "TT_MM", "HS_", "TT_METAL_"))}}
    logger.info(f"HANGSTRESS start {json.dumps(res)}")
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=num_blocks * BLOCK_SIZE)
    enable_host_embedding(model)
    mtp = None
    if use_mtp:
        from models.demos.blackhole.qwen36.tt.spec.mtp import MTPHead, load_mtp_state_dict

        mtp = MTPHead(model, load_mtp_state_dict(model.args.CKPT_DIR), model.args.weight_cache_path())
    kv_shape = [num_blocks, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    if mtp is not None:
        mtp.allocate_kv_cache(kv_shape, dtype=model._paged_kv_caches[0][0].dtype)
    verifier = None
    if os.environ.get("HS_VERIFIER", "0") == "1":  # MTPDecoder.generate allocates these before its prefill
        from models.demos.blackhole.qwen36.tt.spec.verify import TargetVerifier

        verifier = TargetVerifier(model, int(os.environ.get("HS_VERIFY_ROWS", "4")))
        verifier.allocate()
    page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
    if path == "traced":
        chunk = max(model._PREFILL_MASK_BUCKETS)
        model.capture_prefill_trace_chunked(mesh_device, page_table, chunk_size=chunk)
    free_mb = int(os.environ.get("HS_FREE_MB", "0"))
    filler = []
    if free_mb:
        # Fill DRAM in 16 MiB (per ASIC) slabs, then smaller ones, until about free_mb MiB stay free.
        for slab in (16, 4, 1):
            while ts._memory(mesh_device, "filler")["free_total_MiB"] - slab > free_mb:
                rows = slab * 2**20 // (2 * 1024)  # bf16 [rows, 1024]
                filler.append(ttnn.from_torch(torch.zeros(1, 1, rows, 1024), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                              device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG))
    res["filler_tensors"] = len(filler)
    res["dram_ready"] = ts._memory(mesh_device, "ready to prefill")

    def mtp_chunk(cs, n, hidden, _taps):
        mtp.prefill(ids[:, cs + 1 : cs + n + 1], hidden, n, page_table, chunk_start=cs, want_logits=False)

    res["reps"] = []
    for r in range(reps):
        t0 = time.time()
        if path == "traced":
            logits = model.prefill_traced_chunked(ids, page_table, actual_len=T)
        else:
            model._keep_prefill_hidden = mtp is not None
            try:
                logits, _, _ = model.prefill_chunked_collect(ids, page_table, T, on_chunk=mtp_chunk if mtp else None)
            finally:
                model._keep_prefill_hidden = False
            for t in ([model.prefill_hidden] if model.prefill_hidden is not None else []) + list(model.prefill_taps or ()):
                ttnn.deallocate(t)
            model.prefill_hidden, model.prefill_taps = None, None
        top = int(torch.argmax(ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0].flatten()))
        if ttnn.is_tensor_storage_on_device(logits):
            ttnn.deallocate(logits)
        dt = time.time() - t0
        res["reps"].append({"s": dt, "tok_s": T / dt, "top": top})
        logger.info(f"HANGSTRESS rep {r + 1}/{reps}: {dt:.1f} s ({T / dt:.1f} tok/s) top {top}")
    (HERE / f"results-{tag}.json").write_text(json.dumps(res, indent=1, default=str))
    logger.info(f"HANGSTRESS done {tag}")
    telem.terminate()
