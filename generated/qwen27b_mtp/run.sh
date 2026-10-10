#!/bin/bash
# usage: run.sh <log-name> [VAR=value ...] [-- pytest-target]
# Device protocol: shared flock, refuse if the device has an owner, scoped reset, env -i with
# tt_bench's build_env() variables + the four 27B knobs.
set -u
TT=/home/pullin/personal/tenstorrent/generated/tt-metal-v0.79.0
LOG="$TT/generated/qwen27b_mtp/$1.log"
shift
TARGET=generated/qwen27b_mtp/test_mtp1.py
EXTRA=()
for a in "$@"; do case "$a" in TARGET=*) TARGET="${a#TARGET=}" ;; *) EXTRA+=("$a") ;; esac; done
cd "$TT" || exit 1
flock -o /tmp/n300-device.lock bash -c '
if [ -n "$(fuser /dev/tenstorrent/0 2>/dev/null)" ]; then echo "device busy" >&2; exit 99; fi
echo "LOCKSTATE start $(date -Is) $(uptime) gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1)"
timeout 240 /home/pullin/.tenstorrent-venv/bin/tt-smi -r >/dev/null 2>&1
env -i "$@"; rc=$?
echo "LOCKSTATE end $(date -Is) $(uptime) gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1)"
exit $rc' _ \
  PATH=/usr/bin:/bin HOME="$HOME" TT_METAL_HOME="$TT" TT_METAL_RUNTIME_ROOT="$TT" \
  PYTHONPATH="$TT/ttnn:$TT:$TT/tools:/home/pullin/personal/serving/llama-upstream/gguf-py" \
  ARCH_NAME=wormhole_b0 HF_MODEL=Qwen/Qwen3.8-27B HF_HUB_OFFLINE=1 MESH_DEVICE=N300 \
  TT_CACHE_PATH="$HOME/.cache/tt_transformers/Qwen/Qwen3.8-27B/N300" \
  QWEN36_GATEUP_AGMM=0 QWEN36_DTYPE_OVERRIDES=mlp.down_proj=bfloat4_b QWEN36_PREFILL_CHUNK=512 QWEN36_MLP_PREFILL_ROWS=256 TT_MM_THROTTLE_PERF=5 \
  TT_METAL_OPERATION_TIMEOUT_SECONDS=120 "${EXTRA[@]}" \
  timeout 1800 "$TT/python_env/bin/python" -m pytest -s -p no:cacheprovider --timeout=0 "$TARGET" >"$LOG" 2>&1
echo "exit $? log $LOG"
