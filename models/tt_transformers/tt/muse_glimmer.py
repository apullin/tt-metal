# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Muse Glimmer (MuseGlimmerForConditionalGeneration) text weights for tt_transformers.

Two steps around the generic HF -> Meta conversion in ModelArgs.load_state_dict:

1. ``load_state_dict`` builds the HF-keyed text weights. The transformers release tt-metal pins has
   no Muse Glimmer model, and the HF safetensors checkpoint is ~60 GB, so the weights come from a
   BF16 GGUF converted by llama.cpp (``MUSE_GLIMMER_GGUF`` = path of its first shard).
   llama.cpp's converter (conversion/muse_glimmer.py) changes three things, which
   ``load_hf_state_dict_from_gguf`` undoes so callers see exactly the HF tensors:
   - the four decoder norms are stored as ``1 + w`` (HF keeps the zero-centered ``w``);
   - Q/K projections are permuted from HF's rotate_half layout to ggml's interleaved (NORM) layout;
   - weightless QK norms are materialized as ``attn_q_norm`` (``qk_scale_factor``) and
     ``attn_k_norm`` (ones) tensors, which have no HF counterpart and are dropped.
   Tensors are views of the memory-mapped GGUF where possible, so building the dict reads little.
   Requires llama.cpp's ``gguf-py`` on ``PYTHONPATH``.

2. ``fold_into_meta_state_dict`` rewrites the Meta-keyed weights so the generic tt_transformers
   blocks compute Glimmer exactly (see its docstring).
"""

import glob
import os
import re

import numpy as np
import torch

HF_PREFIX = "model.language_model."

_TOP_LEVEL = {
    "token_embd.weight": HF_PREFIX + "embed_tokens.weight",
    "output_norm.weight": HF_PREFIX + "norm.weight",
    "output.weight": "lm_head.weight",
}
# GGUF per-layer suffix -> (HF per-layer suffix, value llama.cpp added to the tensor).
_LAYER = {
    "attn_norm.weight": ("input_layernorm.weight", 1.0),
    "post_attention_norm.weight": ("post_attention_layernorm.weight", 1.0),
    "ffn_norm.weight": ("pre_feedforward_layernorm.weight", 1.0),
    "post_ffw_norm.weight": ("post_feedforward_layernorm.weight", 1.0),
    "attn_q.weight": ("self_attn.q_proj.weight", 0.0),
    "attn_k.weight": ("self_attn.k_proj.weight", 0.0),
    "attn_v.weight": ("self_attn.v_proj.weight", 0.0),
    "attn_output.weight": ("self_attn.o_proj.weight", 0.0),
    "attn_gate.weight": ("self_attn.gate_proj.weight", 0.0),
    "ffn_gate.weight": ("mlp.gate_proj.weight", 0.0),
    "ffn_up.weight": ("mlp.up_proj.weight", 0.0),
    "ffn_down.weight": ("mlp.down_proj.weight", 0.0),
}
_SYNTHESIZED = ("attn_q_norm.weight", "attn_k_norm.weight")
_LAYER_NAME = re.compile(r"^blk\.(\d+)\.(.+)$")


def gguf_shards(path):
    """All shards of a split GGUF given any one of them (``*-00001-of-00002.gguf``)."""
    match = re.match(r"^(.*)-\d{5}-of-(\d{5})\.gguf$", path)
    if not match:
        return [path]
    shards = sorted(glob.glob(f"{match.group(1)}-*-of-{match.group(2)}.gguf"))
    if len(shards) != int(match.group(2)):
        raise FileNotFoundError(f"expected {int(match.group(2))} GGUF shards next to {path}, found {len(shards)}")
    return shards


def _to_torch(tensor):
    from gguf import GGMLQuantizationType

    if tensor.tensor_type == GGMLQuantizationType.BF16:
        # The reader exposes BF16 as raw bytes shaped (rows, 2 * cols).
        return torch.from_numpy(tensor.data.view(np.int16)).view(torch.bfloat16)
    if tensor.tensor_type == GGMLQuantizationType.F32:
        return torch.from_numpy(tensor.data)
    raise ValueError(f"{tensor.name}: unsupported GGUF type {tensor.tensor_type.name}; use a BF16 GGUF")


def _rotate_half_layout(weight, n_heads):
    """Inverse of llama.cpp's ``_unpermute_for_rope``: interleaved rows back to HF rotate_half rows."""
    rows, cols = weight.shape
    return weight.view(n_heads, rows // n_heads // 2, 2, cols).transpose(1, 2).reshape(rows, cols)


def load_hf_state_dict_from_gguf(path, n_heads, n_kv_heads, qk_scale_factor, n_layers=None):
    """HF-named text weights (``model.language_model.*`` and ``lm_head.weight``) from a BF16 GGUF.

    n_layers keeps only layers [0, n_layers) for truncated test models. The synthesized QK-norm
    tensors are checked against ``qk_scale_factor`` and ones, so a GGUF from a different converter
    fails loudly instead of loading silently wrong attention scales.
    """
    from gguf import GGUFReader

    state_dict = {}
    for shard in gguf_shards(path):
        for tensor in GGUFReader(shard, "r").tensors:
            if tensor.name in _TOP_LEVEL:
                # output_norm is F32 in the GGUF; the HF checkpoint stores it (like every tensor) in BF16.
                state_dict[_TOP_LEVEL[tensor.name]] = _to_torch(tensor).to(torch.bfloat16)
                continue
            match = _LAYER_NAME.match(tensor.name)
            if not match:
                raise KeyError(f"unexpected GGUF tensor {tensor.name}")
            layer, suffix = int(match.group(1)), match.group(2)
            if n_layers is not None and layer >= n_layers:
                continue
            value = _to_torch(tensor)
            if suffix in _SYNTHESIZED:
                expected = qk_scale_factor if suffix == "attn_q_norm.weight" else 1.0
                if not torch.all(value == expected):
                    raise ValueError(f"{tensor.name} is not the synthesized constant {expected}")
                continue
            hf_suffix, shift = _LAYER[suffix]
            if shift:
                value = (value - shift).to(torch.bfloat16)
            elif suffix == "attn_q.weight":
                value = _rotate_half_layout(value, n_heads)
            elif suffix == "attn_k.weight":
                value = _rotate_half_layout(value, n_kv_heads)
            state_dict[f"{HF_PREFIX}layers.{layer}.{hf_suffix}"] = value
    return state_dict


def load_state_dict(args):
    """HF-keyed text weights for ModelArgs.load_state_dict, ahead of the generic HF -> Meta conversion.

    The attention's ``self_attn.gate_proj`` becomes ``self_attn.wg``: the generic key map turns any
    ``gate_proj`` into the MLP's ``w1``.
    """
    path = os.environ.get("MUSE_GLIMMER_GGUF")
    if not path:
        raise ValueError(
            "Muse Glimmer weights load from llama.cpp's BF16 GGUF: set MUSE_GLIMMER_GGUF to its first shard "
            "(the installed transformers has no Muse Glimmer model to load the HF checkpoint with)"
        )
    state_dict = load_hf_state_dict_from_gguf(
        path, args.n_heads, args.n_kv_heads, args.qk_scale_factor, n_layers=args.n_layers
    )
    return {key.replace(".self_attn.gate_proj.", ".self_attn.wg."): value for key, value in state_dict.items()}


def fold_into_meta_state_dict(state_dict, args):
    """Fold Glimmer's weight-side ops into Meta-keyed weights so the generic blocks compute them exactly.

    - The weightless RMSNorm after the embedding acts row-wise, so the table rows are normalized here
      (fp32, same formula as HF, then BF16 like HF's output cast).
    - The four zero-centered decoder norms compute ``x_hat * (1 + w)``: the ``1 +`` is added here, so
      the norms run as plain RMSNorms. The final norm is not zero-centered.
    - Logits are ``lm_head(norm(x)) * output_multiplier`` before the tanh softcap: the multiplier is
      folded into the final norm weight.
    - The QK norms are weightless: they get all-ones weights so the existing q_norm/k_norm path runs.
      Q's ``qk_scale_factor`` is folded into the SDPA scale by the attention.
    """
    table = state_dict["tok_embeddings.weight"].float()
    state_dict["tok_embeddings.weight"] = (
        table * torch.pow(table.pow(2).mean(-1, keepdim=True) + args.norm_eps, -0.5)
    ).to(torch.bfloat16)
    state_dict["norm.weight"] = (state_dict["norm.weight"].float() * args.output_multiplier).to(torch.bfloat16)
    ones = torch.ones(args.head_dim, dtype=torch.bfloat16)
    for layer in range(args.n_layers):
        prefix = f"layers.{layer}."
        for name in ("attention_norm", "ffn_norm", "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            key = f"{prefix}{name}.weight"
            state_dict[key] = (1.0 + state_dict[key].float()).to(torch.bfloat16)
        state_dict[f"{prefix}attention.q_norm.weight"] = ones
        state_dict[f"{prefix}attention.k_norm.weight"] = ones
    return state_dict
