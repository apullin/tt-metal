"""test_longctx.py with the MMIO-chip telemetry sampler (telem.py) running while the test body runs
(started after the mesh is open; the sampler exits with this process). Output: hang/telem-<LC_TAG>.jsonl."""
import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("qwen27b_test_longctx", HERE.parent / "test_longctx.py")
lc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lc)


def start_telem(tag):
    return subprocess.Popen(
        ["/home/pullin/.tenstorrent-venv/bin/python", str(HERE / "telem.py"), str(HERE / f"telem-{tag}.jsonl"), "1"],
        stdout=subprocess.DEVNULL,
        stderr=open(HERE / f"telem-{tag}.err", "w"),
    )


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("device_params", [lc.ts._ns["DEVICE_PARAMS"]], indirect=True)
def test_lc_telem(mesh_device):
    p = start_telem(os.environ.get("LC_TAG", "lc"))
    try:
        lc.test_longctx(mesh_device)
    finally:
        p.terminate()
