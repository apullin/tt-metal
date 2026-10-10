"""Dev harness: batched (concurrent-stream) greedy decode of Qwen3.8-27B on the N300 (throwaway, gitignored).

Applies tt_bench's QWEN36_TEST Wormhole patches + knobs, then:
  BATCH_MODE=ref    plain B=1 greedy (text_demo._run_tp_generation, tt_bench's path) for every prompt in
                    BATCH_PROMPTS, one after another; saves tokens per prompt
  BATCH_MODE=batch  vLLM-style batched path: model built with max_batch_size=B (tt_transformers accepts
                    only 1/2/4/8/16/32, so a non-power-of-two B builds the next power of two and decodes
                    at width B: the attention's bucketed-width path; GDN state is allocated for B rows), one paged KV pool of
                    B * (BATCH_CTX / 64) blocks (row u of the page table = user u's blocks) and the [B,...]
                    DeltaNet state; prefill_paged_slots prefills each user into its slot through the
                    persistent B=1 GDN scratch; then one traced decode step for all B rows per iteration
                    with the per-row per-shard on-device argmax (tt/spec/argmax.py).
Env: BATCH_B, BATCH_TOKENS (128), BATCH_CTX (4096 per user), BATCH_PROMPTS (comma list of keys, cycled
over the B rows; default canon), BATCH_TAG, BATCH_CTX_PROBE (1: time 20 replays at position CTX-96).
"""
import importlib.util
import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer

_spec = importlib.util.spec_from_file_location("tt_bench", "/home/pullin/personal/serving/localmaxxing-n300/tt_bench.py")
_tt_bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tt_bench)
_ns = {"__name__": "tt_bench_qwen36_patches"}
exec(compile(_tt_bench.QWEN36_TEST, "QWEN36_TEST", "exec"), _ns)  # monkeypatches + DEVICE_PARAMS

import ttnn  # noqa: E402
from models.demos.blackhole.qwen36.demo.text_demo import BLOCK_SIZE, _blocks_for, _run_tp_generation  # noqa: E402
from models.demos.blackhole.qwen36.tt.model import Qwen36Model  # noqa: E402
from models.demos.blackhole.qwen36.tt.spec.argmax import combine_argmax_host, sharded_argmax_rows  # noqa: E402
from models.tt_transformers.tt.common import copy_host_to_device  # noqa: E402

OUT = Path(__file__).parent
REF_RUN = Path(
    "/home/pullin/personal/serving/localmaxxing-n300/runs-tt/20261003T162349Z-Qwen--Qwen3.8-27B-qwen36-default/parsed.json"
)
PROMPTS = {
    "canon": Path("/home/pullin/personal/serving/localmaxxing-n300/reasoning-v1.txt").read_text(encoding="utf-8").rstrip("\n"),
    "fib": "Write a Python function that returns the n-th Fibonacci number iteratively, with a short docstring.",
    "bash": "Give a bash one-liner that counts the lines of every .py file under the current directory, and explain it briefly.",
    "sql": "Write a SQL query that returns the five customers with the highest total order amount from tables customers(id, name) and orders(id, customer_id, amount).",
}


def _memory(mesh, label, buffer_type=ttnn.BufferType.DRAM):
    v = ttnn.get_memory_view(mesh, buffer_type)
    info = {
        "label": label,
        "num_banks": v.num_banks,
        "free_bytes_per_bank": v.total_bytes_free_per_bank,
        "largest_free_per_bank": v.largest_contiguous_bytes_free_per_bank,
        "free_total_MiB": v.num_banks * v.total_bytes_free_per_bank / 2**20,
        "total_MiB": v.num_banks * v.total_bytes_per_bank / 2**20,
        "used_MiB": v.num_banks * (v.total_bytes_per_bank - v.total_bytes_free_per_bank) / 2**20,
    }
    logger.info(f"MEMORY {'TRACE' if buffer_type == ttnn.BufferType.TRACE else 'DRAM'} {label}: {json.dumps(info)}")
    return info


def _encode(tokenizer, text):
    chat = tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)
    return tokenizer(chat, add_special_tokens=False, return_tensors="pt")["input_ids"]


def _compare(name, got, ref, n):
    n = min(n, len(ref))
    eq = next((i for i, (a, b) in enumerate(zip(got[:n], ref[:n])) if a != b), None)
    ok = eq is None and len(got) >= n
    logger.info(f"COMPARE {name}: {'EQUAL' if ok else 'DIVERGE'} first {n} tokens; first mismatch at {eq}")
    return {"equal": ok, "n": n, "first_mismatch": eq}


def _save(results, tag):
    (OUT / f"results-{tag}.json").write_text(json.dumps(results, indent=1))


def _run_ref(mesh, tokenizer, keys, n_tok, results, tag):
    encoded = {k: _encode(tokenizer, PROMPTS[k]) for k in keys}
    T_max = max(t.shape[1] for t in encoded.values())
    num_blocks = _blocks_for(T_max, n_tok)
    model = Qwen36Model.from_pretrained(mesh, max_batch_size=1, max_seq_len=num_blocks * BLOCK_SIZE)
    results["mem_load"] = _memory(mesh, "after load")
    results["ref"] = {}
    for k in keys:
        toks, perf = _run_tp_generation(model, tokenizer, encoded[k], n_tok, num_blocks)
        model.free_kv_caches()
        results["ref"][k] = {
            "prompt_tokens": encoded[k].shape[1],
            "tokens": toks,
            "text": tokenizer.decode(toks, skip_special_tokens=True),
            "ttft_s": perf["ttft_s"],
            "decode_tok_s": perf["decode_tok_s"],
        }
        logger.info(f"REF {k}: T={encoded[k].shape[1]} ttft={perf['ttft_s']*1e3:.0f} ms {perf['decode_tok_s']:.3f} tok/s")
        _save(results, tag)


def _run_batch(mesh, tokenizer, B, keys, n_tok, ctx, results, tag):
    row_keys = [keys[u % len(keys)] for u in range(B)]
    encoded = {k: _encode(tokenizer, PROMPTS[k]) for k in set(row_keys)}
    results["rows"] = row_keys
    max_b = 1 << max(0, (B - 1).bit_length())
    results["model_max_batch_size"] = max_b
    model = Qwen36Model.from_pretrained(mesh, max_batch_size=max_b, max_seq_len=ctx)
    results["mem_load"] = _memory(mesh, "after load")
    vocab = model.args.vocab_size
    per_shard = vocab // model.num_devices

    # ---- one KV pool for B users, ctx tokens each ----
    bpu = ctx // BLOCK_SIZE
    assert bpu % 8 == 0
    page_table = torch.stack([torch.arange(u * bpu, (u + 1) * bpu, dtype=torch.int32) for u in range(B)])
    kv_shape = [B * bpu, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=B)
    results["kv_dtype"] = str(model._paged_kv_caches[0][0].dtype)
    results["gdn_rec_dtype"] = str(next(l.attention.rec_state.dtype for l in model.layers if not l.is_full_attention))
    results["mem_kv"] = _memory(mesh, "after KV + GDN state alloc")
    _save(results, tag)

    # ---- prefill warmup (vLLM warmup_model_prefill, batched branch): masked-bucket programs on the
    # persistent B=1 GDN scratch. Prompts here are < one chunk, so no chunk trace is parked. ----
    prev = model._bind_gdn_prefill_scratch()
    try:
        model.capture_prefill_trace_chunked(mesh, page_table[0:1].contiguous(), capture_chunk_trace=False)
    finally:
        model._unbind_gdn_prefill_scratch(prev)
    results["mem_prefill_warm"] = _memory(mesh, "after prefill warmup (B=1 GDN scratch)")

    # ---- per-user prefill into its slot (requests all arrive at t=0, served in order) ----
    first = []
    prefill_s = []
    t_all = time.time()
    for u in range(B):
        t0 = time.time()
        hl = model.prefill_paged_slots([encoded[row_keys[u]]], page_table[u : u + 1].contiguous(), [u])
        first.append(int(torch.argmax(hl[0].reshape(-1)[:vocab].float())))
        prefill_s.append(time.time() - t0)
    ttft_cum = [sum(prefill_s[: u + 1]) for u in range(B)]
    logger.info(f"PREFILL per user ms {[round(s * 1e3) for s in prefill_s]}; TTFT (all arrive at once) ms {[round(s * 1e3) for s in ttft_cum]}")
    results["prefill_ms"] = [s * 1e3 for s in prefill_s]
    results["ttft_ms_cumulative"] = [s * 1e3 for s in ttft_cum]
    results["prompt_tokens"] = [encoded[k].shape[1] for k in row_keys]
    results["mem_prefilled"] = _memory(mesh, "after prefill")
    _save(results, tag)

    # ---- traced batched decode (text_demo._run_tp_generation_batched pattern, greedy per-shard argmax) ----
    model._ondev_argmax = True
    pos = [encoded[k].shape[1] for k in row_keys]
    dev = model.prepare_inputs_decode(
        torch.tensor(first, dtype=torch.int32).reshape(B, 1), torch.tensor(pos, dtype=torch.int32), page_table=page_table
    )
    gdn = [l.attention for l in model.layers if not l.is_full_attention]
    comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
    mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
    snap = [(ttnn.to_torch(dn.rec_state, mesh_composer=comp), [ttnn.to_torch(c, mesh_composer=comp) for c in dn.conv_states]) for dn in gdn]

    def _fwd():
        logits, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
        return sharded_argmax_rows(logits, B, per_shard)

    for t in _fwd():  # eager compile
        ttnn.deallocate(t)
    trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
    tt_idx, tt_val = _fwd()
    ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
    for dn, (rec, convs) in zip(gdn, snap):
        for src, dst in [(rec, dn.rec_state), *zip(convs, dn.conv_states)]:
            t = ttnn.from_torch(src, dtype=dst.dtype, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=mapper)
            ttnn.copy(t, dst)
            ttnn.deallocate(t)
    results["mem_traced"] = _memory(mesh, "after decode trace capture")
    results["trace_region"] = _memory(mesh, "decode trace", ttnn.BufferType.TRACE)
    _save(results, tag)

    def _update(tokens, positions):
        host = model.prepare_decode_inputs_host(
            torch.tensor(tokens, dtype=torch.int32).reshape(B, 1), torch.tensor(positions, dtype=torch.int32)
        )
        copy_host_to_device(host[:3], device_tensors=dev[:3])

    generated = [[first[u]] for u in range(B)]
    times, exec_times = [], []
    while len(generated[0]) < n_tok:
        t0 = time.time()
        _update([g[-1] for g in generated], pos)
        t1 = time.time()
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        t2 = time.time()
        nxt = combine_argmax_host(tt_idx, tt_val, mesh, B, per_shard)
        for u in range(B):
            generated[u].append(nxt[u])
        pos = [p + 1 for p in pos]
        times.append(time.time() - t0)
        exec_times.append(t2 - t1)
    steady = times[1:]
    step_ms = 1e3 * sum(steady) / len(steady)
    exec_ms = 1e3 * sum(exec_times[1:]) / len(exec_times[1:])
    results["decode"] = {
        "steps": len(times),
        "step_ms": step_ms,
        "step_ms_min": 1e3 * min(steady),
        "step_ms_max": 1e3 * max(steady),
        "exec_ms": exec_ms,
        "aggregate_tok_s": B * 1e3 / step_ms,
        "per_stream_tok_s": 1e3 / step_ms,
    }
    logger.info(f"DECODE B={B}: step {step_ms:.2f} ms (exec {exec_ms:.2f}), aggregate {B * 1e3 / step_ms:.2f} tok/s, per stream {1e3 / step_ms:.2f} tok/s")
    results["rows_out"] = [
        {"key": row_keys[u], "tokens": generated[u], "text": tokenizer.decode(generated[u], skip_special_tokens=True)} for u in range(B)
    ]
    results["mem_end"] = _memory(mesh, "end of decode")
    _save(results, tag)

    if os.environ.get("BATCH_CTX_PROBE", "1") == "1":
        # Step time near the per-user context limit: every row at position ctx-96 (outputs ignored).
        pp = [ctx - 96] * B
        ct = []
        for i in range(21):
            t0 = time.time()
            _update([g[-1] for g in generated], pp)
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            combine_argmax_host(tt_idx, tt_val, mesh, B, per_shard)
            ct.append(time.time() - t0)
            pp = [p + 1 for p in pp]
        results["decode"]["step_ms_at_ctx"] = 1e3 * sum(ct[1:]) / len(ct[1:])
        logger.info(f"DECODE B={B} at position {ctx - 96}: step {results['decode']['step_ms_at_ctx']:.2f} ms")
    ttnn.release_trace(mesh, trace_id)


def _check(results, tokenizer, n_tok):
    ref_text = json.loads(REF_RUN.read_text())["batches"][-1]["outputText"]
    ref_retok = tokenizer(ref_text, add_special_tokens=False)["input_ids"]
    own = {}
    for p in OUT.glob("results-ref-*.json"):
        own.update(json.loads(p.read_text()).get("ref", {}))
    checks = []
    for u, row in enumerate(results["rows_out"]):
        c = {"row": u, "key": row["key"]}
        if row["key"] == "canon":
            c["vs_parsed_retokenized"] = _compare(f"row {u} canon vs 162349Z (re-tokenized)", row["tokens"], ref_retok, n_tok)
            c["text_prefix_of_162349Z"] = ref_text.startswith(row["text"].strip()[: len(row["text"].strip())])
        if row["key"] in own:
            c["vs_own_b1"] = _compare(f"row {u} {row['key']} vs own B=1 plain run", row["tokens"], own[row["key"]]["tokens"], n_tok)
        checks.append(c)
    results["checks"] = checks
    results["all_equal"] = all(
        all(v["equal"] for k, v in c.items() if isinstance(v, dict)) and c.get("text_prefix_of_162349Z", True) for c in checks
    )
    logger.info(f"CHECK all rows equal: {results['all_equal']}")


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("device_params", [_ns["DEVICE_PARAMS"]], indirect=True)
def test_batch(mesh_device):
    mode = os.environ.get("BATCH_MODE", "batch")
    B = int(os.environ.get("BATCH_B", "1"))
    n_tok = int(os.environ.get("BATCH_TOKENS", "128"))
    ctx = int(os.environ.get("BATCH_CTX", "4096"))
    keys = os.environ.get("BATCH_PROMPTS", "canon").split(",")
    tag = os.environ.get("BATCH_TAG", f"{mode}-b{B}")
    env_knobs = {k: os.environ.get(k) for k in ("QWEN36_TRACE_REGION_MB", "QWEN_SDPA_BF8", "QWEN35_GDN_STATE_BF16")}
    results = {"mode": mode, "B": B, "tokens": n_tok, "ctx": ctx, "prompts": keys, "knobs": env_knobs}
    logger.info(f"BATCH harness: {results}")
    tokenizer = AutoTokenizer.from_pretrained(os.environ["HF_MODEL"])
    mesh_device.enable_program_cache()
    if mode == "ref":
        _run_ref(mesh_device, tokenizer, keys, n_tok, results, tag)
    else:
        _run_batch(mesh_device, tokenizer, B, keys, n_tok, ctx, results, tag)
        _check(results, tokenizer, n_tok)
    results["mem_final"] = _memory(mesh_device, "final")
    _save(results, tag)
    logger.info(f"RESULTS written to {OUT / f'results-{tag}.json'}")
