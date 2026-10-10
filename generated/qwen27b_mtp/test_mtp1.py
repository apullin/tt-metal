"""Dev harness: MTP-1 speculative greedy decode for Qwen3.8-27B on the N300 (throwaway, gitignored).

Applies tt_bench's QWEN36_TEST Wormhole patches + knobs, builds the target (optionally a subset of
checkpoint layers) plus the MTP head, and checks the gates:
  a  verify-step rows vs plain B=1 decode steps on the same inputs (argmax + max |dlogit|)
  b  oracle drafts (all accepted) and forced-wrong drafts (all rejected -> rollback) == plain greedy
  c  MTP drafts == plain greedy; acceptance rate; tok/s
Env: MTP_LAYERS ("0,1,2,3" | "all"), MTP_TOKENS, MTP_GATES ("a,b,c"), MTP_TRACE (0|1), MTP_TAG.
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
from models.demos.blackhole.qwen36.tt.generator_interface import unpack_rope  # noqa: E402
from models.demos.blackhole.qwen36.tt.model import Qwen36Model  # noqa: E402
from models.demos.blackhole.qwen36.tt.spec.argmax import combine_argmax_host, sharded_argmax_rows  # noqa: E402
from models.demos.blackhole.qwen36.tt.spec.mtp import MTPHead, load_mtp_state_dict  # noqa: E402
from models.demos.blackhole.qwen36.tt.spec.mtp1 import MTP1Decoder  # noqa: E402
from models.demos.blackhole.qwen36.tt.spec.verify import TargetVerifier  # noqa: E402

PROMPT = Path("/home/pullin/personal/serving/localmaxxing-n300/reasoning-v1.txt").read_text(encoding="utf-8").rstrip("\n")
OUT = Path(__file__).parent
PLAIN_RUN = "/home/pullin/personal/serving/localmaxxing-n300/runs-tt/20261003T162349Z-Qwen--Qwen3.8-27B-qwen36-default/parsed.json"
def _memory(mesh, label):
    v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    info = {
        "label": label,
        "num_banks": v.num_banks,
        "total_bytes_per_bank": v.total_bytes_per_bank,
        "free_bytes_per_bank": v.total_bytes_free_per_bank,
        "largest_free_per_bank": v.largest_contiguous_bytes_free_per_bank,
        "free_total_MiB": v.num_banks * v.total_bytes_free_per_bank / 2**20,
        "total_MiB": v.num_banks * v.total_bytes_per_bank / 2**20,
    }
    logger.info(f"MEMORY {label}: {json.dumps(info)}")
    return info


def _gate_a(model, token_ids, num_blocks):
    """Verify rows [y, y1] at [T, T+1] vs two plain B=1 decode steps from the same state."""
    mesh = model.mesh_device
    vocab, per_shard = model.args.vocab_size, model.args.vocab_size // model.num_devices
    T = token_ids.shape[1]
    nb = ((num_blocks + 31) // 32) * 32
    kv_shape = [nb, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    verifier = TargetVerifier(model, 2)
    verifier.allocate()
    page_table = torch.arange(nb, dtype=torch.int32).reshape(1, nb)
    comp_vocab = ttnn.ConcatMeshToTensor(mesh, dim=-1)
    try:
        lt = ttnn.to_torch(model.prefill_traced_chunked(token_ids, page_table, actual_len=T), mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        y = int(torch.argmax(lt.reshape(-1, vocab)[0].float()))
        dec = MTP1Decoder(model, None)
        dec._gdn = [l.attention for l in model.layers if not l.is_full_attention]
        snap = dec._snapshot_gdn_host()

        plain_logits, plain_tok = [], []
        tok = y
        for step in range(2):
            d = model.prepare_inputs_decode(torch.tensor([[tok]], dtype=torch.int32), torch.tensor([T + step], dtype=torch.int32), page_table=page_table)
            cos, sin = unpack_rope(d[2])
            logits = model._forward_decode(d[0], cos, sin, d[1], d[3], sharded_lm_head=True)
            idx, val = sharded_argmax_rows(logits, 1, per_shard)
            tok = combine_argmax_host(idx, val, mesh, 1, per_shard)[0]
            plain_tok.append(tok)
            plain_logits.append(ttnn.to_torch(logits, mesh_composer=comp_vocab).float().reshape(-1, vocab)[0])
        dec._restore_gdn_host(snap)

        host = dec._verify_inputs_host([y, plain_tok[0]], T)
        dv = [ttnn.to_device(h, mesh) for h in host]
        pt2 = ttnn.to_device(ttnn.from_torch(page_table.repeat(2, 1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), mesh)
        cos, sin = unpack_rope(dv[2])
        logits, hidden = verifier.forward(dv[0], cos, sin, dv[1], pt2, dv[3:5])
        idx, val = sharded_argmax_rows(logits, 2, per_shard)
        vtok = combine_argmax_host(idx, val, mesh, 2, per_shard)
        vlog = ttnn.to_torch(logits, mesh_composer=comp_vocab).float().reshape(-1, vocab)[:2]
        res = {
            "first_token": y,
            "plain_tokens": plain_tok,
            "verify_tokens": vtok,
            "row0_argmax_equal": vtok[0] == plain_tok[0],
            "row1_argmax_equal": vtok[1] == plain_tok[1],
            "row0_max_abs_dlogit": float((vlog[0] - plain_logits[0]).abs().max()),
            "row1_max_abs_dlogit": float((vlog[1] - plain_logits[1]).abs().max()),
            "row0_bitwise_equal": bool(torch.equal(vlog[0], plain_logits[0])),
            "row1_bitwise_equal": bool(torch.equal(vlog[1], plain_logits[1])),
        }
        # Rollback check: restoring row 0's snapshot must leave the GDN state a plain step leaves.
        verifier.rollback(0)
        after_rb = dec._snapshot_gdn_host()
        dec._restore_gdn_host(snap)
        d = model.prepare_inputs_decode(torch.tensor([[y]], dtype=torch.int32), torch.tensor([T], dtype=torch.int32), page_table=page_table)
        cos, sin = unpack_rope(d[2])
        model._forward_decode(d[0], cos, sin, d[1], d[3], sharded_lm_head=True)
        after_plain = dec._snapshot_gdn_host()
        res["rollback_state_bitwise_equal"] = all(
            torch.equal(a[0], b[0]) and all(torch.equal(x, z) for x, z in zip(a[1], b[1])) for a, b in zip(after_rb, after_plain)
        )
        logger.info(f"GATE a: {json.dumps(res)}")
        return res
    finally:
        verifier.free()
        model.free_kv_caches()


def _compare(name, got, ref, n):
    eq = next((i for i, (a, b) in enumerate(zip(got[:n], ref[:n])) if a != b), None)
    ok = eq is None and len(got) >= n and len(ref) >= n
    logger.info(f"COMPARE {name}: {'EQUAL' if ok else 'DIVERGE'} first {n} tokens; first mismatch at {eq}")
    return {"equal": ok, "n": n, "first_mismatch": eq}


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("device_params", [_ns["DEVICE_PARAMS"]], indirect=True)
def test_mtp1(mesh_device):
    layers = os.environ.get("MTP_LAYERS", "0,1,2,3")
    layer_indices = None if layers == "all" else [int(i) for i in layers.split(",")]
    n_tok = int(os.environ.get("MTP_TOKENS", "48"))
    gates = os.environ.get("MTP_GATES", "a,b,c").split(",")
    use_trace = os.environ.get("MTP_TRACE", "0") == "1"
    tag = os.environ.get("MTP_TAG", f"l{layers.replace(',', '-')}-t{int(use_trace)}")
    results = {"layers": layers, "tokens": n_tok, "trace": use_trace}

    tokenizer = AutoTokenizer.from_pretrained(os.environ["HF_MODEL"])
    text = tokenizer.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=False)
    token_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    T = token_ids.shape[1]
    logger.info(f"Encoded prompt lengths:{T}")
    mesh_device.enable_program_cache()
    num_blocks = _blocks_for(T, n_tok + 2)
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=num_blocks * BLOCK_SIZE, layer_indices=layer_indices)
    results["mem_target"] = _memory(mesh_device, "after target load")
    t = time.time()
    mtp = MTPHead(model, load_mtp_state_dict(model.args.CKPT_DIR), model.args.weight_cache_path())
    logger.info(f"MTP head loaded in {time.time() - t:.1f}s")
    results["mem_mtp"] = _memory(mesh_device, "after MTP head load")

    # Plain greedy reference: the tt_bench decode path (traced).
    ref, perf = _run_tp_generation(model, tokenizer, token_ids, n_tok, num_blocks)
    model.free_kv_caches()
    results["plain"] = {"tokens": ref, "decode_tok_s": perf["decode_tok_s"], "text": tokenizer.decode(ref, skip_special_tokens=True)}
    logger.info(f"PLAIN {perf['decode_tok_s']:.3f} tok/s: {ref}")

    if "a" in gates:
        results["gate_a"] = _gate_a(model, token_ids, num_blocks)

    dec = MTP1Decoder(model, mtp)
    vocab = model.args.vocab_size
    if "b" in gates:
        oracle = lambda i, d: ref[i] if i < len(ref) else d
        got, st = dec.generate(token_ids, n_tok, num_blocks, use_trace=use_trace, draft_override=oracle)
        results["gate_b_oracle"] = {**_compare("oracle", got, ref, n_tok), **st}
        logger.info(f"GATE b oracle stats {st}")
        wrong = lambda i, d: (ref[i] + 1) % vocab if i < len(ref) else d
        got, st = dec.generate(token_ids, n_tok, num_blocks, use_trace=use_trace, draft_override=wrong)
        results["gate_b_wrong"] = {**_compare("forced-wrong", got, ref, n_tok), **st}
        logger.info(f"GATE b forced-wrong stats {st}")
    if "c" in gates:
        got, st = dec.generate(token_ids, n_tok, num_blocks, use_trace=use_trace)
        results["gate_c"] = {**_compare("mtp", got, ref, n_tok), **st, "tokens": got, "text": tokenizer.decode(got, skip_special_tokens=True)}
        logger.info(f"GATE c stats {st}")
        ref_text = json.loads(Path(PLAIN_RUN).read_text())["batches"][-1]["outputText"]
        spec_text = tokenizer.decode(got, skip_special_tokens=True).strip()
        results["gate_c"]["text_equals_tt_bench_plain_run"] = spec_text == ref_text
        results["gate_c"]["plain_inprocess_text_equals_tt_bench_plain_run"] = results["plain"]["text"].strip() == ref_text
        logger.info(
            f"TEXT vs {PLAIN_RUN}: spec equal={spec_text == ref_text}, in-process plain equal={results['plain']['text'].strip() == ref_text}"
        )
    results["mem_end"] = _memory(mesh_device, "end")
    (OUT / f"results-{tag}.json").write_text(json.dumps(results, indent=1))
    logger.info(f"RESULTS written to {OUT / f'results-{tag}.json'}")
