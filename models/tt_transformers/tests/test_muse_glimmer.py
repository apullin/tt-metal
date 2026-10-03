# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Muse Glimmer on tt_transformers against the plain-torch reference, prefill then decode.

Needs HF_MODEL=meta-models/Muse-Glimmer-30B and MUSE_GLIMMER_GGUF (see tt/muse_glimmer.py). Four layers
cover both attention kinds: layers 0-2 are sliding-window with RoPE, layer 3 is full attention with NoPE.
The long prompt runs past the 2048-token window, so prefill and decode both cut keys at the window edge.
"""

import bz2
import os

import pytest
import torch
from loguru import logger
from transformers import PretrainedConfig

import ttnn
from models.common.utility_functions import comp_pcc
from models.tt_transformers.tests.muse_glimmer_reference import MuseGlimmerReference
from models.tt_transformers.tt.common import PagedAttentionConfig, create_tt_model
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import DecodersPrecision
from models.tt_transformers.tt.muse_glimmer import load_hf_state_dict_from_gguf

NUM_LAYERS = 4
DECODE_STEPS = 8


@torch.no_grad()
@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
@pytest.mark.parametrize("prompt_len, max_seq_len", [(300, 1024), (2300, 4096)], ids=["short", "past_window"])
def test_muse_glimmer_logits(prompt_len, max_seq_len, mesh_device, reset_seeds, ensure_gc):
    if "Muse-Glimmer" not in os.environ.get("HF_MODEL", ""):
        pytest.skip("HF_MODEL is not a Muse Glimmer checkpoint")
    page_block_size = 32
    paged_attention_config = PagedAttentionConfig(
        block_size=page_block_size, max_num_blocks=max_seq_len // page_block_size
    )
    model_args, tt_model, tt_kv_cache, _ = create_tt_model(
        mesh_device,
        instruct=True,
        max_batch_size=1,
        optimizations=lambda args: DecodersPrecision.performance(args.n_layers, args.model_name),
        max_seq_len=max_seq_len,
        paged_attention_config=paged_attention_config,
        num_layers=NUM_LAYERS,
    )
    generator = Generator([tt_model], [model_args], mesh_device, tokenizer=model_args.tokenizer)

    with bz2.open(os.path.join(os.path.dirname(__file__), "tale-of-two-cities.txt.bz2"), "rt", encoding="utf-8") as f:
        tokens = model_args.encode_prompt(f.read(), instruct=False)[:prompt_len]

    text_config = PretrainedConfig.get_config_dict(model_args.CKPT_DIR)[0]["text_config"]
    hf_state_dict = load_hf_state_dict_from_gguf(
        os.environ["MUSE_GLIMMER_GGUF"],
        model_args.n_heads,
        model_args.n_kv_heads,
        model_args.qk_scale_factor,
        n_layers=NUM_LAYERS,
    )
    reference = MuseGlimmerReference(hf_state_dict, text_config, NUM_LAYERS)

    page_table = torch.argsort(torch.randperm(paged_attention_config.max_num_blocks)).reshape(1, -1)
    tt_logits = generator.prefill_forward_text(
        torch.tensor([tokens]), page_table=page_table, kv_cache=[tt_kv_cache], prompt_lens=[len(tokens)]
    )
    ref_logits = reference.forward(torch.tensor(tokens))[-1]
    pccs = [comp_pcc(ref_logits, tt_logits.reshape(-1)[: ref_logits.numel()].float(), 0.0)[1]]

    # Teacher-force the reference's greedy tokens so both sides see the same inputs.
    for step in range(DECODE_STEPS):
        token = int(ref_logits.argmax())
        output = generator.decode_forward(
            torch.tensor([[token]]),
            torch.tensor([len(tokens) + step]),
            page_table=page_table,
            kv_cache=[tt_kv_cache],
            enable_trace=False,
            reset_batch=step == 0,
        )
        tt_logits = output[0] if isinstance(output, tuple) else output
        ref_logits = reference.forward(torch.tensor([token]))[-1]
        pccs.append(comp_pcc(ref_logits, tt_logits.reshape(-1)[: ref_logits.numel()].float(), 0.0)[1])

    logger.info(f"prefill logits PCC {pccs[0]:.4f}; decode logits PCC {[round(p, 4) for p in pccs[1:]]}")
    assert min(pccs) > 0.97, f"logits PCC below 0.97: {pccs}"
