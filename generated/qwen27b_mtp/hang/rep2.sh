#!/bin/bash
# usage: hang/rep2.sh <tag> [VAR=value ...]
# One long-prefill attempt via run.sh (lock, owner check, tt-smi -r, env -i) with a 30 s dispatch
# timeout that runs tt-triage on the hung device (inspector data is serialized first).
# Appends "<tag> <exit> <hung?> <args>" to hang/runs.txt.
set -u
TT=/home/pullin/personal/tenstorrent/generated/tt-metal-v0.79.0
H=$TT/generated/qwen27b_mtp/hang
tag=$1
shift
TRI="cd $TT && TT_METAL_HOME=$TT $TT/generated/triage-venv/bin/python $TT/tools/tt-triage.py --disable-progress --disable-colors --skip-version-check -v --additional-scripts-directory=$H/triage_extra --llm-output-path=$H/$tag-triage.csv >$H/$tag-triage.txt 2>&1; cp -r $TT/generated/inspector $H/$tag-inspector"
$TT/generated/qwen27b_mtp/run.sh "hang/$tag" TARGET=generated/qwen27b_mtp/test_longctx.py LC_HOST_EMB=1 \
  QWEN36_TRACE_REGION_MB=128 QWEN_SDPA_BF8=1 LC_TAG="hang-$tag" \
  TT_METAL_OPERATION_TIMEOUT_SECONDS=30 TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="$TRI" "$@"
rc=$?
hung=no
grep -q "Timeout detected" "$H/$tag.log" && hung=HANG
res=$(grep -o -E "LONGCTX [a-z0-9_]*: T=.*|HANGSTRESS rep .*|\[TP chunked collect\] .*|\[TP chunk-replay\] .*" "$H/$tag.log" | cut -c1-110 | tail -4 | tr '\n' ';')
echo "$(date -Is) $tag rc=$rc $hung | $* | $res" >>"$H/runs.txt"
echo "$tag rc=$rc $hung $res"
