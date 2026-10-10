"""Dev harness: host embedding, bf8 MTP KV, 128K windows and long prompts (throwaway, gitignored).

Reuses test_spec.py (tt_bench's QWEN36_TEST patches + knobs, helpers). Runs, in order, the methods in
LC_METHODS on one prompt and compares every text with the first plain run of the same embedding mode:
  plain_dev  plain greedy (_run_tp_generation) with the device embedding table (before LC_HOST_EMB flips)
  plain      plain greedy (_run_tp_generation), host embedding table when LC_HOST_EMB=1
  mtp<k>     MTPDecoder(k)            dflash2   DFlash2Decoder
Env: LC_METHODS (e.g. plain_dev,plain,mtp3,dflash2), LC_HOST_EMB (0|1), LC_CONTEXT (KV window tokens,
0 = prompt + output only), LC_PROMPT (canonical | longprompt-<N>.pt), LC_TOKENS, LC_TAG,
LC_WARMUP_TOKENS (> 0: a canonical-prompt run of that many tokens per method first, to compile).
KV dtype: QWEN_SDPA_BF8=1 (bfp8 KV for the target and the MTP head).
"""
import importlib.util
import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

_spec = importlib.util.spec_from_file_location("qwen27b_test_spec", Path(__file__).parent / "test_spec.py")
ts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ts)

import ttnn  # noqa: E402
from models.demos.blackhole.qwen36.demo.text_demo import BLOCK_SIZE, _blocks_for, _run_tp_generation  # noqa: E402
from models.demos.blackhole.qwen36.tt.model import Qwen36Model  # noqa: E402
from models.demos.blackhole.qwen36.tt.spec.host_embedding import enable_host_embedding  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

OUT = Path(__file__).parent
DFT = next(Path.home().glob(".cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/*"))


def _canonical_ids(tokenizer):
    text = tokenizer.apply_chat_template([{"role": "user", "content": ts.PROMPT}], add_generation_prompt=True, tokenize=False)
    return tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]


def _emb_bench(model, iters=50):
    """Host-embedding input costs: host tensor build and host->device copy, 1 and 4 rows, vs token ids."""
    from models.tt_transformers.tt.common import copy_host_to_device

    mesh, emb = model.mesh_device, model.host_embedding
    res = {}
    for rows in (1, 4):
        toks = list(range(1000, 1000 + rows))
        variants = {
            "embedded_tile": lambda: emb.host(toks),
            "embedded_rowmajor": lambda: ttnn.from_torch(
                emb.table[torch.tensor(toks)][None, None].contiguous(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=emb.mapper
            ),
            "token_ids": lambda: ttnn.from_torch(torch.tensor([[t] for t in toks], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
        }
        for name, build in variants.items():
            dev = copy_host_to_device([build()], mesh_device=mesh)
            t0 = time.time()
            for _ in range(iters):
                h = build()
            t1 = time.time()
            for _ in range(iters):
                copy_host_to_device([h], device_tensors=dev)
            ttnn.synchronize_device(mesh)
            t2 = time.time()
            res[f"{name}_rows{rows}"] = {"build_ms": 1000 * (t1 - t0) / iters, "copy_ms": 1000 * (t2 - t1) / iters}
            ttnn.deallocate(dev[0])
    logger.info(f"EMB_BENCH {json.dumps(res)}")
    return res


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("device_params", [ts._ns["DEVICE_PARAMS"]], indirect=True)
def test_longctx(mesh_device):
    methods = os.environ.get("LC_METHODS", "plain").split(",")
    host_emb = os.environ.get("LC_HOST_EMB", "0") == "1"
    context = int(os.environ.get("LC_CONTEXT", "0"))
    prompt_src = os.environ.get("LC_PROMPT", "canonical")
    n_tok = int(os.environ.get("LC_TOKENS", "256"))
    warm = int(os.environ.get("LC_WARMUP_TOKENS", "0"))
    tag = os.environ.get("LC_TAG", "lc")
    tokenizer = AutoTokenizer.from_pretrained(os.environ["HF_MODEL"])
    canonical = _canonical_ids(tokenizer)
    if prompt_src == "canonical":
        token_ids, meta = canonical, {"source": "canonical reasoning-v1"}
    else:
        d = torch.load(OUT / prompt_src, weights_only=False)
        token_ids, meta = d["ids"], d["meta"]
    T = token_ids.shape[1]
    results = {
        "tag": tag,
        "methods": methods,
        "host_embedding": host_emb,
        "kv_bf8": os.environ.get("QWEN_SDPA_BF8", "0") == "1",
        "trace_region_mb": os.environ.get("QWEN36_TRACE_REGION_MB", "1024"),
        "prompt": {**meta, "tokens": T},
        "output_tokens": n_tok,
    }
    logger.info(f"LONGCTX prompt {T} tokens, methods {methods}, context {context}")
    mesh_device.enable_program_cache()
    num_blocks = _blocks_for(T, n_tok + 16)
    num_blocks = ((max(num_blocks, -(-context // BLOCK_SIZE)) + 31) // 32) * 32
    results["context_tokens"] = num_blocks * BLOCK_SIZE
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=num_blocks * BLOCK_SIZE)
    results["mem_target"] = ts._memory(mesh_device, "after target load")

    def flip_host_emb():
        if host_emb and model.host_embedding is None:
            before = ts._memory(mesh_device, "before moving the embedding table to the host")
            enable_host_embedding(model)
            results["mem_host_embedding"] = ts._memory(mesh_device, "after moving the embedding table to the host")
            results["host_embedding_freed_MiB"] = results["mem_host_embedding"]["free_total_MiB"] - before["free_total_MiB"]
            if os.environ.get("LC_EMB_BENCH") == "1":
                results["emb_bench"] = _emb_bench(model)

    drafters = {}

    def decoder(method):
        if method.startswith("mtp"):
            from models.demos.blackhole.qwen36.tt.spec.mtp import MTPHead, load_mtp_state_dict
            from models.demos.blackhole.qwen36.tt.spec.mtpk import MTPDecoder

            if "mtp" not in drafters:
                drafters["mtp"] = MTPHead(model, load_mtp_state_dict(model.args.CKPT_DIR), model.args.weight_cache_path())
                results["mem_mtp_head"] = ts._memory(mesh_device, "after MTP head load")
            return MTPDecoder(model, drafters["mtp"], int(method[3:]))
        from models.demos.blackhole.qwen36.tt.spec.dflash2 import DFlash2Drafter, load_dflash2_checkpoint
        from models.demos.blackhole.qwen36.tt.spec.dflash2_spec import DFlash2Decoder

        if "dflash2" not in drafters:
            config, sd = load_dflash2_checkpoint(DFT)
            drafters["dflash2"] = DFlash2Drafter(model, config, sd, model.args.weight_cache_path())
            del sd
            results["mem_dflash2"] = ts._memory(mesh_device, "after DFlash2 drafter load")
        return DFlash2Decoder(model, drafters["dflash2"])

    # LC_SPEC_CONTEXTS: descending KV windows to try for the speculative methods (the first that
    # allocates, compiles and runs is measured; failures are recorded). Default: the plain window.
    spec_contexts = [int(c) for c in os.environ.get("LC_SPEC_CONTEXTS", "").split(",") if c]

    def blocks_for(ctx):
        return ((max(_blocks_for(T, n_tok + 16), -(-ctx // BLOCK_SIZE)) + 31) // 32) * 32

    def run(method, ids, tokens, blocks):
        t0 = time.time()
        if method.startswith("plain"):
            got, perf = _run_tp_generation(model, tokenizer, ids, tokens, blocks)
            mem = ts._memory(mesh_device, f"{method}: after the run, KV still allocated")
            model.free_kv_caches()
            st = {"ttft_s": perf["ttft_s"], "decode_tok_s": perf["decode_tok_s"], "dram_after_run": mem}
        else:
            got, st = decoder(method).generate(ids, tokens, blocks)
        st["wall_s"] = time.time() - t0
        st["context_tokens"] = blocks * BLOCK_SIZE
        return got, st

    ref = {}
    for method in methods:
        if method != "plain_dev":
            flip_host_emb()
        failures = []
        windows = [num_blocks] if method.startswith("plain") or not spec_contexts else [blocks_for(c) for c in spec_contexts]
        for i, blocks in enumerate(windows):
            try:
                if warm:  # same window (page-table widths key the programs), canonical prompt
                    _, wst = run(method, canonical, warm, blocks)
                    logger.info(f"WARMUP {method}: {wst.get('decode_tok_s')}")
                got, st = run(method, token_ids, n_tok, blocks)
                break
            except Exception as e:  # noqa: BLE001 - fit probe: record and try the next window
                if i == len(windows) - 1:
                    raise
                failures.append({"context_tokens": blocks * BLOCK_SIZE, "error": str(e)[:400]})
                logger.warning(f"FIT {method} context {blocks * BLOCK_SIZE} failed: {str(e)[:400]}")
        st["fit_failures"] = failures
        text = tokenizer.decode(got, skip_special_tokens=True)
        entry = {**{k: v for k, v in st.items() if k != "profiler"}, "tokens": got, "text": text}
        base = "plain" if "plain" in ref else ("plain_dev" if "plain_dev" in ref else None)
        if base is not None and method != base:
            entry["vs"] = base
            entry.update(ts._compare(f"{method} vs {base}", got, ref[base]["tokens"], n_tok))
            entry["text_equal"] = text == ref[base]["text"]
        if "plain_dev" in ref and method == "plain":
            entry["vs_plain_dev"] = ts._compare("plain (host emb) vs plain_dev", got, ref["plain_dev"]["tokens"], n_tok)
            entry["vs_plain_dev"]["text_equal"] = text == ref["plain_dev"]["text"]
        if method.startswith("plain"):
            ref[method] = entry
        results[method] = entry
        logger.info(
            f"LONGCTX {method}: T={T} ttft {st['ttft_s']:.2f}s ({T / st['ttft_s']:.1f} tok/s) decode {st['decode_tok_s']:.2f} tok/s "
            f"acc {st.get('accepted')}/{st.get('drafts')} equal {entry.get('equal')} dram {st.get('dram') or st.get('dram_after_run')}"
        )
    results["mem_end"] = ts._memory(mesh_device, "end")
    (OUT / f"results-{tag}.json").write_text(json.dumps(results, indent=1, default=str))
    logger.info(f"RESULTS written to {OUT / f'results-{tag}.json'}")
