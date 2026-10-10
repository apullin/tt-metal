"""Telemetry sampler for the MMIO chip only (UMD device 0 = N300 L; no remote/ethernet access).

usage: telem.py <out.jsonl> [period_s]   (run with ~/.tenstorrent-venv/bin/python; stops on SIGTERM or
when its parent process exits)
One JSON line per sample: t, aiclk MHz, temp C, tdp W, tdc A, vcore mV, throttler bits.
"""
import json
import os
import signal
import sys
import time

import tt_umd as u
from tt_umd import wormhole

out, period = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
parent = os.getppid()
stop = False
signal.signal(signal.SIGTERM, lambda *a: globals().__setitem__("stop", True))
dev = u.TTDevice.create(0)
dev.init_tt_device()
r = u.SmBusArcTelemetryReader(dev)
T = wormhole.TelemetryTag
with open(out, "a") as f:
    while not stop and os.getppid() == parent:
        v = {k: r.read_entry(getattr(T, k).value) for k in ("AICLK", "ASIC_TEMPERATURE", "TDP", "TDC", "VCORE", "THROTTLER")}
        f.write(json.dumps({
            "t": round(time.time(), 2),
            "aiclk": v["AICLK"] & 0xFFFF,
            "temp": round((v["ASIC_TEMPERATURE"] & 0xFFFF) / 16, 1),
            "tdp": v["TDP"] & 0xFFFF,
            "tdc": v["TDC"] & 0xFFFF,
            "vcore": v["VCORE"],
            "throttler": v["THROTTLER"],
        }) + "\n")
        f.flush()
        time.sleep(period)
