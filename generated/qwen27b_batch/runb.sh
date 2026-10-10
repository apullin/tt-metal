#!/bin/bash
# usage: runb.sh <log-name> [VAR=value ...] [TARGET=pytest-target]
# Device protocol: shared flock, refuse if the device has an owner, scoped reset, env -i with
# tt_bench's build_env() variables + the four 27B knobs. Logs uptime + GPU state at start and end
# (inside the lock) to <log-name>.load.
set -u
TT=/home/pullin/personal/tenstorrent/generated/tt-metal-v0.79.0
LOG="$TT/generated/qwen27b_batch/$1.log"
LOAD="$TT/generated/qwen27b_batch/$1.load"
shift
TARGET=generated/qwen27b_batch/test_batch.py
EXTRA=()
for a in "$@"; do case "$a" in TARGET=*) TARGET="${a#TARGET=}" ;; *) EXTRA+=("$a") ;; esac; done
cd "$TT" || exit 1
flock /tmp/n300-device.lock bash -c '
LOAD="$1"; shift
if [ -n "$(fuser /dev/tenstorrent/0 2>/dev/null)" ]; then echo "device busy" >&2; exit 99; fi
for i in $(seq 12); do awk "{exit !(\$1 > 4)}" /proc/loadavg || break; sleep 5; done  # let the previous holder load decay (max 60 s)
{ echo "start $(date -u +%FT%TZ) $(uptime)"; echo "start gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"; top -b -n1 -o %CPU | sed -n "7,10p"; } >"$LOAD"
timeout 240 /home/pullin/.tenstorrent-venv/bin/tt-smi -r >/dev/null 2>&1
env -i "$@"
rc=$?
{ echo "end $(date -u +%FT%TZ) $(uptime)"; echo "end gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"; top -b -n1 -o %CPU | sed -n "7,10p"; echo "rc $rc"; } >>"$LOAD"
exit $rc' _ "$LOAD" \
  PATH=/usr/bin:/bin HOME="$HOME" TT_METAL_HOME="$TT" TT_METAL_RUNTIME_ROOT="$TT" \
  PYTHONPATH="$TT/ttnn:$TT:$TT/tools:/home/pullin/personal/serving/llama-upstream/gguf-py" \
  ARCH_NAME=wormhole_b0 HF_MODEL=Qwen/Qwen3.8-27B HF_HUB_OFFLINE=1 MESH_DEVICE=N300 \
  TT_CACHE_PATH="$HOME/.cache/tt_transformers/Qwen/Qwen3.8-27B/N300" \
  QWEN36_GATEUP_AGMM=0 QWEN36_DTYPE_OVERRIDES=mlp.down_proj=bfloat4_b QWEN36_PREFILL_CHUNK=512 QWEN36_MLP_PREFILL_ROWS=256 TT_MM_THROTTLE_PERF=5 \
  TT_METAL_OPERATION_TIMEOUT_SECONDS=120 "${EXTRA[@]}" \
  timeout 1800 "$TT/python_env/bin/python" -m pytest -s -p no:cacheprovider --timeout=0 "$TARGET" >"$LOG" 2>&1
echo "exit $? log $LOG"
