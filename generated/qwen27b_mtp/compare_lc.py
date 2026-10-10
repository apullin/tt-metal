"""Summarize long-context results across processes (throwaway): python compare_lc.py <plain-tag> <tag> ...
Each later tag's methods are compared token for token with the plain run of the first tag."""
import json
import sys
from pathlib import Path

D = Path(__file__).parent


def load(tag):
    return json.loads((D / f"results-{tag}.json").read_text())


base = load(sys.argv[1])
ref = base["plain"]
for tag in sys.argv[1:]:
    r = load(tag)
    T = r["prompt"]["tokens"]
    for m in r["methods"]:
        e = r[m]
        n = min(len(e["tokens"]), len(ref["tokens"]))
        mism = next((i for i in range(n) if e["tokens"][i] != ref["tokens"][i]), None)
        dram = e.get("dram") or e.get("dram_after_run") or {}
        free = dram.get("free_bytes_per_bank", 0) * dram.get("num_banks", 0) / 2**20
        row = {
            "tag": tag,
            "method": m,
            "prompt": T,
            "window": e.get("context_tokens"),
            "ttft_s": round(e["ttft_s"], 2),
            "prefill_tok_s": round(T / e["ttft_s"], 1),
            "prompt_s": round(e.get("prompt_s", e["ttft_s"]), 2),
            "decode_tok_s": round(e["decode_tok_s"], 2),
            "accepted": f"{e.get('accepted')}/{e.get('drafts')}" if "drafts" in e else None,
            "tok_per_step": round(e["mean_accepted_length"], 2) if "mean_accepted_length" in e else None,
            "step_ms": round(e["step_ms_mean"], 1) if "step_ms_mean" in e else None,
            "verify_ms": round(e["verify_ms_mean"], 1) if "verify_ms_mean" in e else None,
            "free_MiB": round(free),
            "equal_plain": mism is None and n == len(ref["tokens"]),
            "first_mismatch": mism,
            "text_equal": e["text"] == ref["text"],
        }
        print(json.dumps(row))
