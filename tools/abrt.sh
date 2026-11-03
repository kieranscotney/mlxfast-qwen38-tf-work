#!/bin/bash
# Counterbalanced A/B: $1 = reps, rest = binaries in .build/experiments. Rep i runs the list rotated by i (each binary
# takes each position). One discarded warm-up first. PROMPT=<name> runs work/accept/indep_<name>.json instead of public.
R=${REPO:-.}
A=${WORK:-../work}/accept
cd $R
n=$1; shift; bins=("$@"); k=${#bins[@]}
run() {
  cp .build/experiments/$1 Vendor/TensorFold/zig-out/bin/bench-worker || { echo "cp failed $1"; return; }
  ./tools/stage-bench-worker.sh >/dev/null 2>&1
  if [ -n "$PROMPT" ]; then ACC_GOLDEN=$A/indep_$PROMPT.json MLXFAST_SCORE_PATH=score.json $A/run-golden.sh >/dev/null 2>&1
  else yukon run >/dev/null 2>&1; fi
  f=$(ls -t .build/resident/*.log | head -1)
  m=$(python3 -c "import json;x=json.load(open('score.json'));m=x['metrics'];print(x['passed'],round(1/m['prefill_seconds_per_token'],1),round(1/m['decode_seconds_per_token'],2))" 2>/dev/null)
  echo "$2 $1 $m $(grep 'RT p10' $f | tail -1 | sed 's/.*RT //')"
}
run ${bins[0]} warm
for i in $(seq 0 $((n-1))); do for j in $(seq 0 $((k-1))); do run ${bins[$(( (i+j) % k ))]} r$i; done; done
cp .build/experiments/t1 Vendor/TensorFold/zig-out/bin/bench-worker; ./tools/stage-bench-worker.sh >/dev/null 2>&1
