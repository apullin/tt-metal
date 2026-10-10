"""Deterministic long coding-agent prompts for the long-context runs (throwaway, gitignored).

Body: the Python sources under models/demos/blackhole/qwen36 and models/tt_transformers/tt of this
tt-metal tree, in sorted path order, each under a "### File: <path>" header, cut to the token
budget; then a final instruction. Chat template applied (user turn + generation prompt), token ids
built as template prefix + body ids + template suffix, so the length is exact.

usage: make_longprompt.py N [N ...]  ->  longprompt-<N>.pt ({"ids": [1, N] int64, "meta": {...}}) and
longprompt-<N>.json (tt_bench's TT_BENCH_PROMPT_JSON format, [{"prompt": user text}]; the wrapper's
chat template + re-tokenization of that text gives `json_tokens` ids, within a few of N)
"""
import hashlib
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

TT = Path(__file__).resolve().parents[2]
ROOTS = ["models/demos/blackhole/qwen36", "models/tt_transformers/tt"]
INSTRUCTION = (
    "Above is the source of the Qwen3.6 TT-Metal model port and the tt_transformers library it builds on. "
    "Explain how a decode step flows through the code: which functions run, in what order, and where the "
    "paged KV cache and the DeltaNet state are read and written."
)
MARK = "<<<BODY>>>"


def body_text():
    files = sorted(p for root in ROOTS for p in (TT / root).rglob("*.py") if "__pycache__" not in p.parts)
    return "".join(f"### File: {p.relative_to(TT)}\n{p.read_text(encoding='utf-8')}\n" for p in files), len(files)


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")
    text, n_files = body_text()
    body_ids = tok(text, add_special_tokens=False)["input_ids"]
    chat = tok.apply_chat_template([{"role": "user", "content": MARK + "\n\n" + INSTRUCTION}], add_generation_prompt=True, tokenize=False)
    pre, post = chat.split(MARK)
    pre_ids = tok(pre, add_special_tokens=False)["input_ids"]
    post_ids = tok(post, add_special_tokens=False)["input_ids"]
    for n in map(int, sys.argv[1:]):
        nb = n - len(pre_ids) - len(post_ids)
        assert 0 < nb <= len(body_ids), f"{n}: body has {len(body_ids)} tokens"
        ids = torch.tensor([pre_ids + body_ids[:nb] + post_ids], dtype=torch.int64)
        assert ids.shape[1] == n
        meta = {
            "tokens": n,
            "body_tokens": nb,
            "files": n_files,
            "roots": ROOTS,
            "instruction": INSTRUCTION,
            "sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
        }
        out = Path(__file__).parent / f"longprompt-{n}.pt"
        user_text = tok.decode(body_ids[:nb]) + "\n\n" + INSTRUCTION
        chat_n = tok.apply_chat_template([{"role": "user", "content": user_text}], add_generation_prompt=True, tokenize=False)
        meta["json_tokens"] = len(tok(chat_n, add_special_tokens=False)["input_ids"])
        if out.exists():  # keep the file the runs read; it must hold the same ids
            assert torch.load(out, weights_only=False)["meta"]["sha256"] == meta["sha256"], out
        else:
            torch.save({"ids": ids, "meta": meta}, out)
        out.with_suffix(".json").write_text(json.dumps([{"prompt": user_text}]))
        print(out, meta["sha256"][:16], f"body {nb} of {len(body_ids)} tokens; json prompt {meta['json_tokens']} tokens")


if __name__ == "__main__":
    main()
