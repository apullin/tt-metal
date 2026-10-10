"""Dev harness: DFlash2 speculative greedy decode for Qwen3.8-27B on the N300 (throwaway, gitignored).

Reuses test_spec.py (tt_bench's QWEN36_TEST patches + knobs, gate a, helpers). Gates (DF_GATES):
  a  8-row verify vs 8 plain B=1 steps (logits bitwise) and rollback to every row (test_spec gate a)
  d  TT drafter vs the torch reference (serving/generated/dflash2_ref: ref_gate_b.py's CPU BF16
     target features fed to both): final hidden PCC, top-16 overlap, selector projection, drafts
  b  forced acceptance (accept exactly j of 7 drafts every step) == plain greedy, j in DF_FORCED
  c  real DFlash2 drafts == plain greedy; per-position acceptance; tok/s; step split
Env: DF_GATES, DF_TOKENS, DF_TRACE (0|1), DF_FORCED (e.g. 0,3,7), DF_HOST_EMB (0|1), DF_TAG.
"""
import importlib.util
import json
import os
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
from models.demos.blackhole.qwen36.tt.spec.dflash2 import DFlash2Drafter, load_dflash2_checkpoint, rope_host  # noqa: E402
from models.demos.blackhole.qwen36.tt.spec.dflash2_spec import DFlash2Decoder, _host_i32, _host_u32_col, _upd_host  # noqa: E402
from models.demos.blackhole.qwen36.tt.generator_interface import unpack_rope  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

REF = Path("/home/pullin/personal/serving/generated/dflash2_ref")
DFT = next(Path.home().glob(".cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/*"))
OUT = Path(__file__).parent


def _pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def _gate_d(model, dr, num_blocks):
    """TT drafter on the CPU reference features vs the torch reference blocks."""
    mesh = model.mesh_device
    feats_d = torch.load(REF / "feats.pt", weights_only=False)
    ref = torch.load(REF / "ref.pt", weights_only=False)
    feats, T = feats_d["feats"], feats_d["T"]
    seq = feats_d["seq"]
    R, D = dr.block_size, model.args.dim
    nb = ((num_blocks + 31) // 32) * 32
    page_table = torch.arange(nb, dtype=torch.int32).reshape(1, nb)
    dr.allocate_kv_cache(nb, BLOCK_SIZE)
    shard = ttnn.ShardTensorToMesh(mesh, dim=-1)

    def taps_dev(rows):
        """CPU features of `rows` (list; -1 = zeros) -> 5 fractured device taps [1,1,len,D/2]."""
        f = torch.stack([feats[r] if r >= 0 else torch.zeros_like(feats[0]) for r in rows])
        return [
            ttnn.from_torch(f[:, j * D : (j + 1) * D][None, None].contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=shard)
            for j in range(len(dr.target_layer_ids))
        ]

    res = {"blocks": []}
    try:
        bucket = model._mask_bucket_for(T)
        ptaps = taps_dev(list(range(T)) + [-1] * (bucket - T))
        dr.prefill(ptaps)
        for t in ptaps:
            ttnn.deallocate(t)
        pt = ttnn.to_device(ttnn.from_torch(page_table.repeat(R, 1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), device=mesh)
        prev = None
        for b in ref["blocks"]:
            start = b["start"]
            inj_pos = start if prev is None else prev
            inj = taps_dev([-1] * R if prev is None else list(range(prev, prev + R)))
            host = [
                rope_host(range(inj_pos, inj_pos + R), dr.HD, dr.theta),
                _host_u32_col([b["anchor"]] + [dr.mask_token_id] * (R - 1)),
                rope_host(range(start, start + R), dr.HD, dr.theta),
                _host_i32([start + R - 1] * R),
                *_upd_host(start, R),
                *_upd_host(inj_pos, R),
            ]
            dev = [ttnn.to_device(t, mesh) for t in host]
            icos, isin = unpack_rope(dev[0])
            cos, sin = unpack_rope(dev[2])
            vals, idx, sel, hidden = dr.decode(inj, icos, isin, dev[4 + R :], dev[1], cos, sin, dev[4 : 4 + R], dev[3], pt, return_hidden=True)
            cands, unary, s = dr.read_candidates(vals, idx, sel)
            drafts = dr.select(cands, unary, s, b["anchor"])
            h = ttnn.to_torch(hidden, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float().reshape(2, -1, D)[0, :R]
            ref_top = b["logits_top16"]
            overlap = [len(set(cands[r].tolist()) & set(ref_top.indices[r - 1].tolist())) for r in range(1, R)]
            top1 = [int(cands[r][int(torch.argmax(unary[r]))]) == int(ref_top.indices[r - 1][0]) for r in range(1, R)]
            truth = seq[start + 1 : start + R].tolist()
            acc = next((i for i in range(R - 1) if drafts[i] != truth[i]), R - 1)
            row = {
                "start": start,
                "hidden_pcc": _pcc(h, b["hidden_all"]),
                "hidden_pcc_rows": [_pcc(h[r], b["hidden_all"][r]) for r in range(R)],
                "selector_proj_pcc": _pcc(s[1:], b["hproj"]),
                "top16_overlap": overlap,
                "top1_equal": top1,
                "drafts": drafts,
                "ref_drafts": b["drafts"],
                "drafts_equal": drafts == b["drafts"],
                "accepted_vs_greedy": acc,
                "ref_accepted": b["accepted"],
            }
            logger.info(f"GATE d block: {json.dumps(row)}")
            res["blocks"].append(row)
            for t in [vals, idx, sel, hidden, *inj, *dev]:
                ttnn.deallocate(t)
            prev = start
        ttnn.deallocate(pt)
        return res
    finally:
        dr.free_kv_cache()


def _gate_q(model, dr, iters=20):
    """Eager timing of single ops in the drafter's tail: LM head, top-k and argmax over a vocab shard."""
    import time

    mesh = model.mesh_device
    R = dr.block_size
    h = ttnn.from_torch(torch.randn(1, 1, R, model.args.dim, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    logits = ttnn.linear(h, model.lm_head_weight)
    ops = {
        "lm_head": lambda: ttnn.linear(h, model.lm_head_weight),
        "topk16_chunked": lambda: dr.chunk_topk(logits),
        "topk16": lambda: ttnn.topk(logits, k=dr.top_k, dim=-1, largest=True, sorted=True),
        "argmax_rows": lambda: ts.sharded_argmax_rows(logits, R, dr.per_shard),
        "selector_proj": lambda: ttnn.linear(h, dr.hproj),
    }
    res = {}
    for name, fn in ops.items():
        ts._dealloc(fn())
        ttnn.synchronize_device(mesh)
        t0 = time.time()
        for _ in range(iters):
            ts._dealloc(fn())
        ttnn.synchronize_device(mesh)
        res[name] = 1000 * (time.time() - t0) / iters
    logger.info(f"GATE q eager ms: {res}")
    # Correctness of the chunked top-k against torch on the same logits (read back).
    host = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float().reshape(R, -1)
    ref_v, ref_i = torch.topk(host, dr.top_k, dim=-1)
    vals, idx = dr.chunk_topk(logits)
    sel = ttnn.linear(h, dr.hproj)
    cands, unary, _ = dr.read_candidates(vals, idx, sel)
    res["chunked_set_equal_rows"] = [set(cands[r].tolist()) == set(ref_i[r].tolist()) for r in range(R)]
    res["chunked_values_max_abs_diff"] = float((unary - ref_v).abs().max())
    res["chunked_vs_torch_first_row"] = {"tt": cands[0].tolist(), "tt_vals": unary[0].tolist(), "torch": ref_i[0].tolist(), "torch_vals": ref_v[0].tolist()}
    v1, i1 = ttnn.topk(logits, k=dr.top_k, dim=-1, largest=True, sorted=True)
    comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
    sv = ttnn.to_torch(v1, mesh_composer=comp).float().reshape(2, R, -1)
    si = ttnn.to_torch(i1, mesh_composer=comp).to(torch.int64).reshape(2, R, -1) + torch.tensor([0, dr.per_shard]).view(2, 1, 1)
    sv, si = sv.permute(1, 0, 2).reshape(R, -1), si.permute(1, 0, 2).reshape(R, -1)
    top, pos = torch.topk(sv, dr.top_k, dim=-1)
    res["single_core_set_equal_rows"] = [set(si.gather(-1, pos)[r].tolist()) == set(ref_i[r].tolist()) for r in range(R)]
    logger.info(f"GATE q topk check: {res}")
    ts._dealloc((vals, idx, sel, v1, i1))
    ttnn.deallocate(logits)
    ttnn.deallocate(h)
    return res


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("device_params", [ts._ns["DEVICE_PARAMS"]], indirect=True)
def test_dflash2(mesh_device):
    n_tok = int(os.environ.get("DF_TOKENS", "256"))
    gates = os.environ.get("DF_GATES", "d,c").split(",")
    use_trace = os.environ.get("DF_TRACE", "1") == "1"
    tag = os.environ.get("DF_TAG", f"df-n{n_tok}")
    results = {"tokens": n_tok, "trace": use_trace, "gates": gates, "trace_region_mb": os.environ.get("QWEN36_TRACE_REGION_MB", "1024")}
    tokenizer = AutoTokenizer.from_pretrained(os.environ["HF_MODEL"])
    text = tokenizer.apply_chat_template([{"role": "user", "content": ts.PROMPT}], add_generation_prompt=True, tokenize=False)
    token_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    T = token_ids.shape[1]
    mesh_device.enable_program_cache()
    num_blocks = _blocks_for(T, n_tok + 16)
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=num_blocks * BLOCK_SIZE)
    results["mem_target"] = ts._memory(mesh_device, "after target load")
    if "a" in gates:
        results["gate_a"] = ts._gate_a(model, token_ids, num_blocks, 8)
    ref = None
    if set(gates) & {"b", "c"}:
        ref, perf = _run_tp_generation(model, tokenizer, token_ids, n_tok, num_blocks)
        model.free_kv_caches()
        results["plain"] = {"tokens": ref, "decode_tok_s": perf["decode_tok_s"], "text": tokenizer.decode(ref, skip_special_tokens=True)}
        logger.info(f"PLAIN {perf['decode_tok_s']:.3f} tok/s")
    config, sd = load_dflash2_checkpoint(DFT)
    dr = DFlash2Drafter(model, config, sd, model.args.weight_cache_path())
    del sd
    results["mem_drafter"] = ts._memory(mesh_device, "after DFlash2 drafter load")
    if "d" in gates:
        results["gate_d"] = _gate_d(model, dr, num_blocks)
    host_emb = os.environ.get("DF_HOST_EMB", "0") == "1"
    results["host_embedding"] = host_emb
    dec = DFlash2Decoder(model, dr, host_embedding=host_emb)
    if host_emb:
        results["mem_host_embedding"] = ts._memory(mesh_device, "after moving the embedding table to the host")
    dec.profile_iters = int(os.environ.get("DF_PROFILE", "0"))
    vocab = model.args.vocab_size
    if "q" in gates:
        results["gate_q"] = _gate_q(model, dr)
    if "b" in gates:
        for j in [int(x) for x in os.environ.get("DF_FORCED", "0,3,7").split(",")]:

            def forced(index, slot, draft, j=j):
                if index >= len(ref):
                    return draft
                return ref[index] if slot < j else (ref[index] + 1) % vocab

            got, st = dec.generate(token_ids, n_tok, num_blocks, use_trace=use_trace, draft_override=forced)
            results[f"gate_b_accept{j}"] = {**ts._compare(f"forced accept {j}/7", got, ref, n_tok), **st}
            logger.info(f"GATE b accept {j}/7 stats {st}")
    if "c" in gates:
        got, st = dec.generate(token_ids, n_tok, num_blocks, use_trace=use_trace)
        results["gate_c"] = {**ts._compare("dflash2", got, ref, n_tok), **st, "tokens": got, "text": tokenizer.decode(got, skip_special_tokens=True)}
        results["gate_c"]["text_equals_plain"] = results["gate_c"]["text"] == results["plain"]["text"]
        logger.info(f"GATE c stats {st}")
    results["mem_end"] = ts._memory(mesh_device, "end")
    (OUT / f"results-{tag}.json").write_text(json.dumps(results, indent=1, default=str))
    logger.info(f"RESULTS written to {OUT / f'results-{tag}.json'}")
