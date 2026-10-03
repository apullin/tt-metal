# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Target k-row verify step: one decode forward over k consecutive tokens of ONE sequence.

- Full-attention layers: k batch rows whose page-table rows alias the sequence's row, each with its
  own cur_pos (t+i), so paged SDPA-decode row i sees exactly positions <= t+i. The KV writes run as
  one paged_update_cache per row (`TPAttention.kv_update_idxs`; other rows carry -1), because rows
  whose positions share a cache tile race inside a single call.
- GDN layers: everything but the recurrence runs batched over the rows; the recurrence runs row by
  row on the single state, with the state after every row but the last kept for rollback
  (`gdn_rows`).
- Norms, MLP and the LM head run over the rows as one tile, as in plain decode.

All inputs are device tensors and nothing is allocated persistently, so the step is traceable.
"""
import ttnn
from models.demos.blackhole.qwen36.tt.spec.gdn_rows import GDNRowState, gdn_decode_rows
from models.tt_transformers.tt.common import Mode


def decode_layer(layer, x, attention_fn):
    """Qwen36DecoderLayer.forward (TP, decode) with the attention call supplied by the caller."""
    norm_config = layer.args.get_norm_config("attn", Mode.DECODE)
    attn_input = layer.attention_norm(x, mode=Mode.DECODE, norm_config=norm_config)
    attn_output = attention_fn(attn_input)
    ttnn.deallocate(attn_input)
    h = ttnn.add(x, attn_output)
    ttnn.deallocate(attn_output)
    ff_input = layer.ffn_norm(h, mode=Mode.DECODE, norm_config=norm_config)
    ff_output = layer.feed_forward.forward(ff_input, mode="decode")
    ttnn.deallocate(ff_input)
    output = ttnn.add(h, ff_output)
    ttnn.deallocate(h)
    ttnn.deallocate(ff_output)
    return output


class TargetVerifier:
    """k-row verify forward + GDN rollback for a TP Qwen36Model holding one sequence."""

    def __init__(self, model, rows):
        assert model.num_devices > 1, "the verify step is implemented for the TP path"
        self.model = model
        self.rows = rows
        self.gdn_states = None

    def allocate(self):
        """Snapshot buffers for every GDN layer. Call after allocate_kv_caches, before any trace."""
        assert self.gdn_states is None
        self.gdn_states = [
            None if layer.is_full_attention else GDNRowState(layer.attention, self.rows) for layer in self.model.layers
        ]

    def free(self):
        for state in self.gdn_states or ():
            if state is not None:
                state.free()
        self.gdn_states = None

    def forward(self, tokens, cos, sin, cur_pos, page_table, kv_update_idxs):
        """tokens [rows, 1] uint32; cos/sin [1, rows, 1, rope_dim]; cur_pos [rows] int32 (t..t+rows-1);
        page_table [rows, blocks] (every row = the sequence's row); kv_update_idxs: `rows` int32 [rows]
        tensors, tensor i holding t+i at row i and -1 elsewhere.

        Returns (vocab-sharded logits [1, 1, rows, vocab/devices], post-final-norm hidden
        [1, 1, rows, dim] replicated)."""
        model = self.model
        x = model.embd(tokens)
        x = ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[1], x.shape[-1]))
        assert x.shape[-2] == self.rows
        for layer, state in zip(model.layers, self.gdn_states):
            if layer.is_full_attention:
                attention = layer.attention

                def attention_fn(a, attention=attention):
                    attention.kv_update_idxs = kv_update_idxs
                    try:
                        return attention.forward_decode(a, cur_pos, cos, sin, page_table=page_table)
                    finally:
                        attention.kv_update_idxs = None

            else:

                def attention_fn(a, dn=layer.attention, state=state):
                    return gdn_decode_rows(dn, a, state)

            x_new = decode_layer(layer, x, attention_fn)
            ttnn.deallocate(x)
            x = x_new
        hidden = model._final_norm_decode(x)
        ttnn.deallocate(x)
        logits = ttnn.linear(hidden, model.lm_head_weight)
        return logits, hidden

    def rollback(self, row):
        """Restore every GDN layer to its state after verify row `row` (< rows-1)."""
        for state in self.gdn_states:
            if state is not None:
                state.restore(row)
