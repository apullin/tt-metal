# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Muse Glimmer text model in plain torch, for tt_transformers PCC tests.

The transformers release tt-metal pins has no Muse Glimmer model, so this mirrors
transformers 5.16's modeling_muse_glimmer (MuseGlimmerTextModel + the CausalLM logit transform)
on HF-keyed weights. It computes in fp32 from the BF16 weights.
"""

import torch

from models.tt_transformers.tt.muse_glimmer import HF_PREFIX


def _rms(x, eps):
    return x * torch.pow(x.pow(2).mean(-1, keepdim=True) + eps, -0.5)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class MuseGlimmerReference:
    """Prefill then token-by-token decode with a KV cache; returns fp32 logits for every input position."""

    def __init__(self, state_dict, text_config, n_layers):
        self.w = {key[len(HF_PREFIX) :] if key.startswith(HF_PREFIX) else key: value for key, value in state_dict.items()}
        self.c = text_config
        self.n_layers = n_layers
        self.n_heads = text_config["num_attention_heads"]
        self.n_kv_heads = text_config["num_key_value_heads"]
        self.head_dim = text_config["head_dim"]
        self.eps = text_config["rms_norm_eps"]
        self.post_eps = text_config["post_norm_eps"]
        theta = text_config["rope_parameters"]["rope_theta"]
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        self.cache = [None] * n_layers  # per layer: (k, v), each [n_kv_heads, positions, head_dim]
        self.pos = 0

    def _weight(self, name):
        return self.w[name].float()

    def _attention(self, layer, x, positions):
        p = f"layers.{layer}.self_attn."
        seq = x.shape[0]
        q = (x @ self._weight(p + "q_proj.weight").T).view(seq, self.n_heads, self.head_dim).transpose(0, 1)
        k = (x @ self._weight(p + "k_proj.weight").T).view(seq, self.n_kv_heads, self.head_dim).transpose(0, 1)
        v = (x @ self._weight(p + "v_proj.weight").T).view(seq, self.n_kv_heads, self.head_dim).transpose(0, 1)
        q = _rms(q, self.eps) * self.c["qk_scale_factor"]
        k = _rms(k, self.eps)
        if self.c["layer_rope_theta"][layer]:  # NoPE layers have theta 0
            freqs = positions[:, None].float() * self.inv_freq[None, :]
            emb = torch.cat((freqs, freqs), dim=-1)
            cos, sin = emb.cos(), emb.sin()
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
        if self.cache[layer] is not None:
            k = torch.cat([self.cache[layer][0], k], dim=1)
            v = torch.cat([self.cache[layer][1], v], dim=1)
        self.cache[layer] = (k, v)
        group = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(group, dim=0)
        v = v.repeat_interleave(group, dim=0)
        scores = (q @ k.transpose(-1, -2)) * self.head_dim**-0.5
        key_pos = torch.arange(k.shape[1])
        allowed = key_pos[None, :] <= positions[:, None]
        if self.c["layer_types"][layer] == "sliding_attention":
            allowed &= key_pos[None, :] > positions[:, None] - self.c["sliding_window"]
        scores = scores.masked_fill(~allowed, float("-inf"))
        attn = (torch.softmax(scores, dim=-1) @ v).transpose(0, 1).reshape(seq, -1)
        attn = attn * torch.sigmoid(x @ self._weight(p + "gate_proj.weight").T)
        return attn @ self._weight(p + "o_proj.weight").T

    def _layer(self, layer, h, positions):
        p = f"layers.{layer}."
        x = _rms(h, self.eps) * (1.0 + self._weight(p + "input_layernorm.weight"))
        attn = self._attention(layer, x, positions)
        h = h + _rms(attn, self.post_eps) * (1.0 + self._weight(p + "post_attention_layernorm.weight"))
        x = _rms(h, self.eps) * (1.0 + self._weight(p + "pre_feedforward_layernorm.weight"))
        gate = torch.nn.functional.silu(x @ self._weight(p + "mlp.gate_proj.weight").T)
        mlp = (gate * (x @ self._weight(p + "mlp.up_proj.weight").T)) @ self._weight(p + "mlp.down_proj.weight").T
        return h + _rms(mlp, self.post_eps) * (1.0 + self._weight(p + "post_feedforward_layernorm.weight"))

    @torch.no_grad()
    def forward(self, tokens):
        """Logits [len(tokens), vocab] for 1-D int tokens appended after everything seen so far."""
        positions = torch.arange(self.pos, self.pos + len(tokens))
        h = _rms(self.w["embed_tokens.weight"][tokens].float(), self.eps)
        for layer in range(self.n_layers):
            h = self._layer(layer, h, positions)
        self.pos += len(tokens)
        h = _rms(h, self.eps) * self._weight("norm.weight")
        logits = (h @ self._weight("lm_head.weight").T) * self.c["output_multiplier"]
        cap = self.c["final_logit_softcapping"]
        return torch.tanh(logits / cap) * cap
