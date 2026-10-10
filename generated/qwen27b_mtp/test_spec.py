"""Dev harness: verify-step profile + MTP-k speculative greedy decode for Qwen3.8-27B on the N300
(throwaway, gitignored).

Applies tt_bench's QWEN36_TEST Wormhole patches + knobs, builds the target plus the MTP head, and
runs the gates selected by MTP_GATES:
  p  device-bound replay timing of the plain step, the (k+1)-row verify step and their parts
  a  (k+1)-row verify vs k+1 plain B=1 decode steps (logits bitwise) and rollback to every row j
     vs the GDN state after j+1 plain steps (bitwise)
  b  forced-acceptance patterns: accept exactly j of k drafts every step, j = 0..k == plain greedy
  c  MTP drafts == plain greedy; per-position acceptance; tok/s; step split
  m  MTP1Decoder (tt_bench's API) == plain greedy (k=1 regression)
Env: MTP_K (1..3), MTP_TOKENS, MTP_GATES, MTP_TRACE (0|1), MTP_TAG, MTP_PROFILE_ITERS.
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
from models.demos.blackhole.qwen36.tt.spec.gdn_rows import gdn_decode_rows  # noqa: E402
from models.demos.blackhole.qwen36.tt.spec.mtp import MTPHead, load_mtp_state_dict  # noqa: E402
from models.demos.blackhole.qwen36.tt.spec.verify import TargetVerifier, decode_layer  # noqa: E402
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (  # noqa: E402
    recurrent_gated_delta_rule_decode_ttnn,
)
from models.tt_transformers.tt.common import Mode  # noqa: E402

PROMPT = Path("/home/pullin/personal/serving/localmaxxing-n300/reasoning-v1.txt").read_text(encoding="utf-8").rstrip("\n")
OUT = Path(__file__).parent


def _memory(mesh, label, buffer_type=ttnn.BufferType.DRAM):
    v = ttnn.get_memory_view(mesh, buffer_type)
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


def _snapshot_gdn(model):
    comp = ttnn.ConcatMeshToTensor(model.mesh_device, dim=0)
    return [
        (ttnn.to_torch(l.attention.rec_state, mesh_composer=comp), [ttnn.to_torch(c, mesh_composer=comp) for c in l.attention.conv_states])
        for l in model.layers
        if not l.is_full_attention
    ]


def _restore_gdn(model, snap):
    mapper = ttnn.ShardTensorToMesh(model.mesh_device, dim=0)
    dns = [l.attention for l in model.layers if not l.is_full_attention]
    for dn, (rec, convs) in zip(dns, snap):
        for src, dst in [(rec, dn.rec_state), *zip(convs, dn.conv_states)]:
            t = ttnn.from_torch(src, dtype=dst.dtype, layout=ttnn.TILE_LAYOUT, device=model.mesh_device, mesh_mapper=mapper)
            ttnn.copy(t, dst)
            ttnn.deallocate(t)


def _states_equal(a, b):
    return all(torch.equal(x[0], y[0]) and all(torch.equal(p, q) for p, q in zip(x[1], y[1])) for x, y in zip(a, b))


def _verify_dev_inputs(model, tokens, pos, nb):
    R = len(tokens)
    tok, cur, rope, _ = model.prepare_decode_inputs_host(
        torch.tensor([[t] for t in tokens], dtype=torch.int32), torch.arange(pos, pos + R, dtype=torch.int32)
    )
    upd = [
        ttnn.from_torch(torch.tensor([pos + r if i == r else -1 for i in range(R)], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        for r in range(R)
    ]
    mesh = model.mesh_device
    pt = ttnn.from_torch(torch.arange(nb, dtype=torch.int32).reshape(1, nb).repeat(R, 1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    dev = [ttnn.to_device(t, mesh) for t in (tok, cur, rope, pt, *upd)]
    return dev[0], dev[1], dev[2], dev[3], dev[4:]


def _prefill(model, token_ids, nb):
    mesh = model.mesh_device
    vocab = model.args.vocab_size
    page_table = torch.arange(nb, dtype=torch.int32).reshape(1, nb)
    lt = ttnn.to_torch(
        model.prefill_traced_chunked(token_ids, page_table, actual_len=token_ids.shape[1]),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
    )
    return int(torch.argmax(lt.reshape(-1, vocab)[0].float())), page_table


def _alloc(model, num_blocks):
    nb = ((num_blocks + 31) // 32) * 32
    model.allocate_kv_caches([nb, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim], ttnn.bfloat16, batch_size=1)
    return nb


# ---- gate p: device-bound profile ----------------------------------------------------------------
def _dealloc(out):
    for t in out if isinstance(out, (tuple, list)) else (out,):
        if isinstance(t, ttnn.Tensor):
            ttnn.deallocate(t)


def _time_trace(mesh, fn, iters):
    """Eager compile, trace capture, then `iters` back-to-back replays (no readback): ms per replay."""
    _dealloc(fn())
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    t = time.time()
    for _ in range(iters):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    ms = 1000 * (time.time() - t) / iters
    ttnn.release_trace(mesh, tid)
    _dealloc(out)
    return ms


def _gate_p(model, token_ids, num_blocks, R, iters):
    mesh = model.mesh_device
    per_shard = model.args.vocab_size // model.num_devices
    T = token_ids.shape[1]
    nb = _alloc(model, num_blocks)
    verifier = TargetVerifier(model, R)
    verifier.allocate()
    res = {"rows": R, "iters": iters}
    try:
        y, page_table = _prefill(model, token_ids, nb)
        d1 = model.prepare_inputs_decode(torch.tensor([[y]], dtype=torch.int32), torch.tensor([T], dtype=torch.int32), page_table=page_table)
        cos1, sin1 = unpack_rope(d1[2])
        tokR, posR, ropeR, ptR, updR = _verify_dev_inputs(model, [y] * R, T, nb)
        cosR, sinR = unpack_rope(ropeR)

        def plain():
            logits = model._forward_decode(d1[0], cos1, sin1, d1[1], d1[3], sharded_lm_head=True)
            idx, val = sharded_argmax_rows(logits, 1, per_shard)
            ttnn.deallocate(logits)
            return idx, val

        def verify():
            logits, hidden = verifier.forward(tokR, cosR, sinR, posR, ptR, updR)
            idx, val = sharded_argmax_rows(logits, R, per_shard)
            ttnn.deallocate(logits)
            return idx, val, hidden

        res["plain_step_ms"] = _time_trace(mesh, plain, iters)
        res["verify_step_ms"] = _time_trace(mesh, verify, iters)

        def embd(tok):
            x = model.embd(tok)
            return ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[1], x.shape[-1]))

        x1, xR = embd(d1[0]), embd(tokR)
        gl = next(l for l in model.layers if not l.is_full_attention)
        res["rollback_ms"] = _time_trace(mesh, lambda: verifier.rollback(0), iters)
        al = next(l for l in model.layers if l.is_full_attention)
        state = verifier.gdn_states[model.layers.index(gl)]
        dn = gl.attention
        nc = gl.args.get_norm_config("attn", Mode.DECODE)
        a1 = gl.attention_norm(x1, mode=Mode.DECODE, norm_config=nc)
        aR = gl.attention_norm(xR, mode=Mode.DECODE, norm_config=nc)

        def attn_rows(a):
            al.attention.kv_update_idxs = updR
            try:
                return al.attention.forward_decode(a, posR, cosR, sinR, page_table=ptR)
            finally:
                al.attention.kv_update_idxs = None

        res["gdn_layer_1_ms"] = _time_trace(mesh, lambda: gl.forward(x1, mode="decode"), iters)
        res["gdn_layer_R_ms"] = _time_trace(mesh, lambda: decode_layer(gl, xR, lambda a: gdn_decode_rows(dn, a, state)), iters)
        res["attn_layer_1_ms"] = _time_trace(
            mesh, lambda: al.forward(x1, cos1, sin1, position_tensor=d1[1], page_table=d1[3], mode="decode"), iters
        )
        res["attn_layer_R_ms"] = _time_trace(mesh, lambda: decode_layer(al, xR, attn_rows), iters)
        res["gdn_mixer_1_ms"] = _time_trace(mesh, lambda: dn.forward_decode(a1), iters)
        res["gdn_mixer_R_ms"] = _time_trace(mesh, lambda: gdn_decode_rows(dn, aR, state), iters)
        from models.demos.blackhole.qwen36.tt.spec.gdn_rows import GDNRowState

        state1 = GDNRowState(dn, 1)
        res["gdn_mixer_rows1_ms"] = _time_trace(mesh, lambda: gdn_decode_rows(dn, a1, state1), iters)
        state1.free()
        a1_3 = ttnn.reshape(a1, (1, 1, a1.shape[-1]))
        aR_3 = ttnn.reshape(aR, (1, R, aR.shape[-1]))
        res["gdn_proj_1_ms"] = _time_trace(mesh, lambda: dn._project_qkvzab(a1_3, 1, out_mc=ttnn.L1_MEMORY_CONFIG), iters)
        res["gdn_proj_R_ms"] = _time_trace(mesh, lambda: dn._project_qkvzab(aR_3, R, out_mc=ttnn.L1_MEMORY_CONFIG), iters)

        def _rand(shape):
            return ttnn.from_torch(
                torch.randn(shape, dtype=torch.bfloat16) * 0.1,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        q = _rand((1, 1, dn.Nv, dn.Dk))
        k = _rand((1, 1, dn.Nv, dn.Dk))
        v = _rand((1, 1, dn.Nv, dn.Dv))
        beta = _rand((1, 1, dn.Nv))
        g = _rand((1, 1, dn.Nv))

        def rec():
            o, new_rec = recurrent_gated_delta_rule_decode_ttnn(
                q, k, v, beta, g, scale=dn.scale, initial_state=dn.rec_state, device=mesh, high_precision=True
            )
            ttnn.copy(new_rec, dn.rec_state)
            ttnn.deallocate(new_rec)
            return o

        res["gdn_recurrence_1_ms"] = _time_trace(mesh, rec, iters)
        hid1 = model._final_norm_decode(x1)
        hidR = model._final_norm_decode(xR)

        def tail(h, rows):
            logits = ttnn.linear(h, model.lm_head_weight)
            idx, val = sharded_argmax_rows(logits, rows, per_shard)
            ttnn.deallocate(logits)
            return idx, val

        res["lm_head_argmax_1_ms"] = _time_trace(mesh, lambda: tail(hid1, 1), iters)
        res["lm_head_argmax_R_ms"] = _time_trace(mesh, lambda: tail(hidR, R), iters)
        n_gdn = sum(1 for l in model.layers if not l.is_full_attention)
        n_attn = len(model.layers) - n_gdn
        res["n_gdn"], res["n_attn"] = n_gdn, n_attn
        res["est_gdn_extra_ms"] = n_gdn * (res["gdn_layer_R_ms"] - res["gdn_layer_1_ms"])
        res["est_attn_extra_ms"] = n_attn * (res["attn_layer_R_ms"] - res["attn_layer_1_ms"])
        res["est_gdn_row_prep_ms"] = (
            res["gdn_mixer_R_ms"] - res["gdn_mixer_1_ms"] - (res["gdn_proj_R_ms"] - res["gdn_proj_1_ms"]) - (R - 1) * res["gdn_recurrence_1_ms"]
        )
        res["trace_region"] = _memory(mesh, "trace region (all profile traces released)", ttnn.BufferType.TRACE)
        logger.info(f"GATE p: {json.dumps(res)}")
        return res
    finally:
        verifier.free()
        model.free_kv_caches()


# ---- gate q: per-op profile of one GDN layer's verify ops ----------------------------------------
def _gate_q(model, R, iters):
    import models.demos.blackhole.qwen36.tt.spec.gdn_rows as gr

    mesh = model.mesh_device
    L1 = ttnn.L1_MEMORY_CONFIG
    gl = next(l for l in model.layers if not l.is_full_attention)
    dn = gl.attention
    dn.reset_state()
    Nv, Dk, Dv = dn.Nv, dn.Dk, dn.Dv

    def rnd(shape, dtype=ttnn.float32, mc=L1):
        return ttnn.from_torch(
            torch.randn(shape, dtype=torch.float32) * 0.1, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh,
            memory_config=mc, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    h = rnd((1, Nv, Dk, Dv))
    h_dram = rnd((1, Nv, Dk, Dv), mc=ttnn.DRAM_MEMORY_CONFIG)
    g = rnd((1, Nv, 1, 1))
    k_row = rnd((1, Nv, 1, Dk))
    k_col = rnd((1, Nv, Dk, 1))
    v_row = rnd((1, Nv, 1, Dv))
    gR = rnd((R, Nv, 1, 1))
    pc = gr._recurrent_read_query_program_config(mesh, Dk, Dv)
    vr = ttnn.matmul(k_row, h, memory_config=L1, program_config=pc, compute_kernel_config=gr._READ_QUERY_CFG)
    outer0 = ttnn.matmul(k_col, v_row, memory_config=L1, compute_kernel_config=gr._WRITE_CFG)
    ops = {
        "to_l1_state": lambda: ttnn.to_memory_config(h_dram, L1),
        "copy_state_to_dram": lambda: [ttnn.copy(h, h_dram)] and None,
        "mul_decay_exp": lambda: ttnn.multiply(h, g, input_tensor_b_activations=[ttnn.UnaryOpType.EXP], memory_config=L1),
        "matmul_read": lambda: ttnn.matmul(k_row, h, memory_config=L1, program_config=pc, compute_kernel_config=gr._READ_QUERY_CFG),
        "subtract": lambda: ttnn.subtract(v_row, vr, memory_config=L1),
        "matmul_outer": lambda: ttnn.matmul(k_col, v_row, memory_config=L1, compute_kernel_config=gr._WRITE_CFG),
        "mul_beta": lambda: ttnn.multiply(outer0, g, memory_config=L1),
        "add_state": lambda: ttnn.add(h, outer0, memory_config=L1),
        "row_slice_small": lambda: gr._row(gR, R - 1),
    }
    D = dn.qkv_dim_tp
    bf = ttnn.bfloat16
    cs = [rnd((1, 1, D), bf, ttnn.DRAM_MEMORY_CONFIG) for _ in range(4)]
    qkv = rnd((1, R, D), bf)
    window = ttnn.concat([*cs[1:], qkv], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ops.update(
        {
            "concat_window": lambda: ttnn.concat([*cs[1:], qkv], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            "slice_window_to_l1": lambda: ttnn.slice(window, (0, 1, 0), (1, 1 + R, D), memory_config=L1),
            "slice_state_out": lambda: [ttnn.slice(window, (0, R, 0), (1, R + 1, D), output_tensor=cs[0])] and None,
            "shift_copy": lambda: [ttnn.copy(cs[1], cs[0])] and None,
        }
    )
    res = {name: _time_trace(mesh, fn, iters) for name, fn in ops.items()}
    grid = mesh.compute_with_storage_grid_size()
    res["grid"] = [grid.x, grid.y]
    comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
    ref = ttnn.to_torch(ttnn.matmul(k_col, v_row, memory_config=L1, compute_kernel_config=gr._WRITE_CFG), mesh_composer=comp)
    for pm, pn, sh, sw in ((4, 4, 1, 4), (1, 4, 1, 4), (2, 2, 2, 2), (4, 2, 2, 2), (1, 2, 1, 2), (2, 4, 1, 4)):
        name = f"matmul_outer_reuse_m{pm}n{pn}"
        try:
            pc_o = ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=(grid.x, grid.y),
                in0_block_w=1,
                out_subblock_h=sh,
                out_subblock_w=sw,
                per_core_M=pm,
                per_core_N=pn,
            )
            fn = lambda pc_o=pc_o: ttnn.matmul(k_col, v_row, memory_config=L1, program_config=pc_o, compute_kernel_config=gr._WRITE_CFG)
            out = ttnn.to_torch(fn(), mesh_composer=comp)
            res[name] = {"ms": _time_trace(mesh, fn, iters), "bitwise_equal_default": bool(torch.equal(out, ref))}
        except Exception as e:  # noqa: BLE001 - exploring configs
            res[name] = repr(e)[:200]
    logger.info(f"GATE q: {json.dumps(res)}")
    return res




# ---- gate a: (k+1)-row verify vs plain steps, rollback to every row -----------------------------
def _gate_a(model, token_ids, num_blocks, R):
    mesh = model.mesh_device
    vocab, per_shard = model.args.vocab_size, model.args.vocab_size // model.num_devices
    T = token_ids.shape[1]
    nb = _alloc(model, num_blocks)
    verifier = TargetVerifier(model, R)
    verifier.allocate()
    comp_vocab = ttnn.ConcatMeshToTensor(mesh, dim=-1)
    try:
        y, page_table = _prefill(model, token_ids, nb)
        snap0 = _snapshot_gdn(model)
        plain_logits, plain_tok, plain_states = [], [], []
        tok = y
        for step in range(R):
            d = model.prepare_inputs_decode(torch.tensor([[tok]], dtype=torch.int32), torch.tensor([T + step], dtype=torch.int32), page_table=page_table)
            cos, sin = unpack_rope(d[2])
            logits = model._forward_decode(d[0], cos, sin, d[1], d[3], sharded_lm_head=True)
            idx, val = sharded_argmax_rows(logits, 1, per_shard)
            tok = combine_argmax_host(idx, val, mesh, 1, per_shard)[0]
            plain_tok.append(tok)
            plain_logits.append(ttnn.to_torch(logits, mesh_composer=comp_vocab).float().reshape(-1, vocab)[0])
            plain_states.append(_snapshot_gdn(model))
        _restore_gdn(model, snap0)

        tokR, posR, ropeR, ptR, updR = _verify_dev_inputs(model, [y] + plain_tok[:-1], T, nb)
        cos, sin = unpack_rope(ropeR)
        logits, hidden = verifier.forward(tokR, cos, sin, posR, ptR, updR)
        idx, val = sharded_argmax_rows(logits, R, per_shard)
        vtok = combine_argmax_host(idx, val, mesh, R, per_shard)
        vlog = ttnn.to_torch(logits, mesh_composer=comp_vocab).float().reshape(-1, vocab)[:R]
        res = {"rows": R, "first_token": y, "plain_tokens": plain_tok, "verify_tokens": vtok}
        res["rows_bitwise_equal"] = [bool(torch.equal(vlog[r], plain_logits[r])) for r in range(R)]
        res["rows_max_abs_dlogit"] = [float((vlog[r] - plain_logits[r]).abs().max()) for r in range(R)]
        after_verify = _snapshot_gdn(model)
        res["last_row_state_bitwise_equal"] = _states_equal(after_verify, plain_states[R - 1])
        rb = []
        for j in range(R - 1):
            verifier.rollback(j)
            rb.append(_states_equal(_snapshot_gdn(model), plain_states[j]))
        res["rollback_state_bitwise_equal"] = rb
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
def test_spec(mesh_device):
    k = int(os.environ.get("MTP_K", "1"))
    n_tok = int(os.environ.get("MTP_TOKENS", "256"))
    gates = os.environ.get("MTP_GATES", "a,b,c").split(",")
    use_trace = os.environ.get("MTP_TRACE", "1") == "1"
    iters = int(os.environ.get("MTP_PROFILE_ITERS", "20"))
    tag = os.environ.get("MTP_TAG", f"k{k}-n{n_tok}")
    results = {"k": k, "tokens": n_tok, "trace": use_trace, "gates": gates}

    tokenizer = AutoTokenizer.from_pretrained(os.environ["HF_MODEL"])
    text = tokenizer.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=False)
    token_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    T = token_ids.shape[1]
    logger.info(f"Encoded prompt lengths:{T}")
    mesh_device.enable_program_cache()
    num_blocks = _blocks_for(T, n_tok + k + 2)
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=num_blocks * BLOCK_SIZE)
    results["mem_target"] = _memory(mesh_device, "after target load")
    mtp = None
    if set(gates) & {"b", "c", "m"}:
        mtp = MTPHead(model, load_mtp_state_dict(model.args.CKPT_DIR), model.args.weight_cache_path())
        results["mem_mtp"] = _memory(mesh_device, "after MTP head load")

    if "q" in gates:  # run alone: resets one GDN layer's state buffers
        results["gate_q"] = _gate_q(model, k + 1, iters)
    if "p" in gates:
        results["gate_p"] = _gate_p(model, token_ids, num_blocks, k + 1, iters)
    if "a" in gates:
        results["gate_a"] = _gate_a(model, token_ids, num_blocks, k + 1)

    ref = None
    if set(gates) & {"b", "c", "m"}:
        ref, perf = _run_tp_generation(model, tokenizer, token_ids, n_tok, num_blocks)
        model.free_kv_caches()
        results["plain"] = {"tokens": ref, "decode_tok_s": perf["decode_tok_s"], "text": tokenizer.decode(ref, skip_special_tokens=True)}
        logger.info(f"PLAIN {perf['decode_tok_s']:.3f} tok/s")
    vocab = model.args.vocab_size

    if "m" in gates:
        from models.demos.blackhole.qwen36.tt.spec.mtp1 import MTP1Decoder

        got, st = MTP1Decoder(model, mtp).generate(token_ids, n_tok, num_blocks)
        results["gate_m"] = {**_compare("mtp1-api", got, ref, n_tok), **st}
        logger.info(f"GATE m stats {st}")
    if "b" in gates or "c" in gates:
        from models.demos.blackhole.qwen36.tt.spec.mtpk import MTPDecoder

        dec = MTPDecoder(model, mtp, k)
    if "b" in gates:
        for j in range(k + 1):
            # Drafts for slots < j are the plain greedy tokens, slot j (if any) is wrong, later slots
            # are wrong too: every step accepts exactly j drafts.
            def forced(index, slot, draft, j=j):
                if index >= len(ref):
                    return draft
                return ref[index] if slot < j else (ref[index] + 1) % vocab

            got, st = dec.generate(token_ids, n_tok, num_blocks, use_trace=use_trace, draft_override=forced)
            results[f"gate_b_accept{j}"] = {**_compare(f"forced accept {j}/{k}", got, ref, n_tok), **st}
            logger.info(f"GATE b accept {j}/{k} stats {st}")
    if "c" in gates:
        got, st = dec.generate(token_ids, n_tok, num_blocks, use_trace=use_trace)
        results["gate_c"] = {**_compare("mtp", got, ref, n_tok), **st, "tokens": got, "text": tokenizer.decode(got, skip_special_tokens=True)}
        results["gate_c"]["text_equals_plain"] = results["gate_c"]["text"] == results["plain"]["text"]
        logger.info(f"GATE c stats {st}")
    results["mem_end"] = _memory(mesh_device, "end")
    (OUT / f"results-{tag}.json").write_text(json.dumps(results, indent=1))
    logger.info(f"RESULTS written to {OUT / f'results-{tag}.json'}")
