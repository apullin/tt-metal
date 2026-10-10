"""Throwaway: traced Glimmer decode step time with parts of the attention removed (timing only, outputs are wrong).

PROF_VARIANT: comma-separated subset of
  nogate     - drop the sigmoid output gate (matmul, reshard, multiply)
  noqknorm   - drop the per-head Q/K norms (two reshard round trips + two norms)
"""
import os
import statistics
import time

import pytest
import torch

import ttnn
from models.tt_transformers.tt import attention as tt_attention
from models.tt_transformers.tt.common import PagedAttentionConfig, create_tt_model
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import DecodersPrecision

NUM_LAYERS = int(os.environ.get("PROF_LAYERS", "13"))
VARIANT = set(filter(None, os.environ.get("PROF_VARIANT", "").split(",")))
STEPS = 60

_init = tt_attention.Attention.__init__


def _patched_init(self, *args, **kwargs):
    _init(self, *args, **kwargs)
    if "nogate" in VARIANT:
        self.wg = None
    if "noqknorm" in VARIANT:
        self.q_norm = lambda x, mode, norm_config: x
        self.k_norm = lambda x, mode, norm_config: x


tt_attention.Attention.__init__ = _patched_init


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": True, "trace_region_size": 50000000}], indirect=True)
def test_trace_ablation(mesh_device):
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
    generator.prefill_forward_text(torch.tensor([tokens]), page_table=page_table, kv_cache=[tt_kv_cache], prompt_lens=[len(tokens)])
    times = []
    token = 1000
    for step in range(STEPS):
        start = time.perf_counter()
        out = generator.decode_forward(torch.tensor([[token]]), torch.tensor([len(tokens) + step]), page_table=page_table,
                                       kv_cache=[tt_kv_cache], enable_trace=True, reset_batch=step == 0)
        logits = out[0] if isinstance(out, tuple) else out
        token = int(logits.reshape(-1).argmax()) % 200000
        times.append(time.perf_counter() - start)
    steady = times[10:]
    # Device-bound: enqueue replays without reading logits back, then wait for the device.
    replays = 200
    ttnn.synchronize_device(mesh_device)
    start = time.perf_counter()
    for step in range(replays):
        generator.decode_forward(torch.tensor([[token]]), torch.tensor([len(tokens) + STEPS + step]), page_table=page_table,
                                 kv_cache=[tt_kv_cache], enable_trace=True, read_from_device=False)
    ttnn.synchronize_device(mesh_device)
    device_ms = (time.perf_counter() - start) / replays * 1000
    print(f"ABLATION layers={NUM_LAYERS} variant={','.join(sorted(VARIANT)) or 'baseline'} "
          f"step_ms_p50={statistics.median(steady) * 1000:.2f} min={min(steady) * 1000:.2f} device_bound_ms={device_ms:.3f}")
