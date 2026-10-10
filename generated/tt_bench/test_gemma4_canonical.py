"""Written by localmaxxing-n300/tt_bench.py: canonical-prompt Gemma 4 batches through run_generation."""
import json
import math
import os
from pathlib import Path

from loguru import logger
from transformers import AutoTokenizer

from models.demos.gemma4.demo.text_demo import _encode_demo_prompt, model_path, run_generation  # noqa: F401 (fixture import)
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric


@parametrize_mesh_with_fabric([(1, 2)])
def test_canonical_prompt(mesh_device, model_path):
    prompt = json.loads(Path(os.environ["TT_BENCH_PROMPT_JSON"]).read_text())[0]["prompt"]
    prompt_ids = _encode_demo_prompt(AutoTokenizer.from_pretrained(model_path), prompt, instruct=True)
    logger.info(f"Encoded prompt lengths:{len(prompt_ids)}")
    max_new_tokens = int(os.environ["TT_BENCH_MAX_TOKENS"])
    max_seq_len = int(os.environ["TT_BENCH_MAX_SEQ_LEN"])
    page_params = {"page_block_size": 64, "page_max_num_blocks": math.ceil(max_seq_len / 64)}
    for batch in range(int(os.environ["TT_BENCH_BATCHES"])):
        logger.info(f"Processing batch {batch}")
        text = run_generation(
            mesh_device=mesh_device, model_path=model_path, prompts=[prompt], max_new_tokens=max_new_tokens,
            max_seq_len=max_seq_len, page_params=page_params, enable_decode_trace=True,
        )[0]
        text = text[len(prompt):] if text.startswith(prompt) else text  # run_generation returns prompt + output
        logger.info(f"\n==REPEAT BATCH {batch}\n==USER 0 - PROMPT\n(canonical reasoning-v1)\n==USER 0 - OUTPUT\n{text.strip()}\n==END OUTPUT\n")
