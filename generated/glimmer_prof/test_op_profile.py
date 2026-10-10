"""Throwaway: per-op wall time of eager Glimmer decode steps (ttnn op hooks, device sync around each op).

Run with TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false}' so ops go through the hooked path.
"""
import collections
import os
import time

import pytest
import torch

import ttnn
from models.tt_transformers.tt.common import PagedAttentionConfig, create_tt_model
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import DecodersPrecision

NUM_LAYERS = int(os.environ.get("PROF_LAYERS", "4"))
WARM_STEPS = 3
PROF_STEPS = 5


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_op_profile(mesh_device):
    paged = PagedAttentionConfig(block_size=32, max_num_blocks=64)
    model_args, tt_model, tt_kv_cache, _ = create_tt_model(
        mesh_device,
        instruct=True,
        max_batch_size=1,
        optimizations=lambda args: DecodersPrecision.performance(args.n_layers, args.model_name),
        max_seq_len=2048,
        paged_attention_config=paged,
        num_layers=NUM_LAYERS,
    )
    generator = Generator([tt_model], [model_args], mesh_device, tokenizer=model_args.tokenizer)
    tokens = list(range(1000, 1300))
    page_table = torch.arange(paged.max_num_blocks).reshape(1, -1)
    logits = generator.prefill_forward_text(torch.tensor([tokens]), page_table=page_table, kv_cache=[tt_kv_cache], prompt_lens=[len(tokens)])
    token = int(logits.reshape(-1).argmax())

    def step(pos, tok):
        out = generator.decode_forward(torch.tensor([[tok]]), torch.tensor([pos]), page_table=page_table,
                                       kv_cache=[tt_kv_cache], enable_trace=False, reset_batch=pos == len(tokens))
        return int((out[0] if isinstance(out, tuple) else out).reshape(-1).argmax())

    pos = len(tokens)
    for _ in range(WARM_STEPS):
        token = step(pos, token)
        pos += 1

    totals = collections.defaultdict(float)
    counts = collections.Counter()
    stack = []

    def pre(operation, args, kwargs):
        if not stack:
            ttnn.synchronize_device(mesh_device)
        stack.append(time.perf_counter())

    def post(operation, args, kwargs, output):
        start = stack.pop()
        if not stack:
            ttnn.synchronize_device(mesh_device)
            totals[operation.python_fully_qualified_name] += time.perf_counter() - start
            counts[operation.python_fully_qualified_name] += 1

    step_times = []
    with ttnn.register_pre_operation_hook(pre), ttnn.register_post_operation_hook(post):
        for _ in range(PROF_STEPS):
            start = time.perf_counter()
            token = step(pos, token)
            step_times.append(time.perf_counter() - start)
            pos += 1

    # Baseline: the same sync-wrapped timing around a trivial op.
    tiny = ttnn.from_torch(torch.ones(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                           mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
    base = []
    for _ in range(50):
        ttnn.synchronize_device(mesh_device)
        start = time.perf_counter()
        ttnn.add(tiny, tiny)
        ttnn.synchronize_device(mesh_device)
        base.append(time.perf_counter() - start)
    baseline_ms = sorted(base)[len(base) // 2] * 1000

    print(f"PROF layers={NUM_LAYERS} eager step ms {[round(t * 1000, 1) for t in step_times]} baseline op {baseline_ms:.3f} ms")
    for name, total in sorted(totals.items(), key=lambda kv: -kv[1]):
        per_step_ms = total / PROF_STEPS * 1000
        calls = counts[name] / PROF_STEPS
        print(f"PROF {name:60s} {per_step_ms:8.3f} ms/step {calls:6.1f} calls/step {per_step_ms / calls:7.3f} ms/call")
