#!/bin/bash
# run_examples.sh [example ...] - build and run every example (default: all of examples/*), print a summary.
#
# Each example is built from scratch (NKI compile -> analyze -> gpsimd-cc -> inject) and run with
# gpsimd-run: outputs compared bit-exactly with the numpy reference, every GPSIMD core must leave a
# receipt for every call site. Logs: examples/<name>/build/{build,run}.log
# Environment: NEURON_VENV, GPSIMD_TOOLS, SEED (default 1), NEURON_RT_VISIBLE_CORES (default 0)
set -uo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT/examples"
if [ $# -gt 0 ]; then LIST=("$@"); else mapfile -t LIST < <(ls -d */ | tr -d / | sort); fi
pass=0; fail=0; rows=()
for e in "${LIST[@]}"; do
    [ -f "$e/kernel.py" ] || continue
    t0=$(date +%s)
    rm -rf "$e/build"; mkdir -p "$e/build"
    if ! "$ROOT/bin/gpsimd-build" "$e" > "$e/build/build.log" 2>&1 || [ ! -s "$e/build/final.neff" ]; then
        rows+=("$(printf '%-18s BUILD-FAIL  %4ss  see %s' "$e" $(( $(date +%s) - t0 )) "examples/$e/build/build.log")"); fail=$((fail + 1)); continue
    fi
    "$ROOT/bin/gpsimd-run-example" "$e/build" > "$e/build/run.log" 2>&1
    rc=$?
    sites=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["sites"])' "$e/build/manifest.json")
    ncheck=$(grep -o 'bit-exact' "$e/build/run.log" | wc -l)
    cyc=$(grep -o '[0-9]* core cycles' "$e/build/run.log" | head -1)
    if [ $rc -eq 0 ] && grep -q 'RESULT: PASS' "$e/build/run.log"; then
        status=PASS; pass=$((pass + 1))
    else
        status=FAIL; fail=$((fail + 1))
    fi
    rows+=("$(printf '%-18s %-10s %4ss  %d site(s), %3d output checks, %s' "$e" "$status" $(( $(date +%s) - t0 )) "$sites" "$ncheck" "$cyc")")
    echo "${rows[-1]}"
done
echo
echo "== summary"
printf '%s\n' "${rows[@]}"
echo "passed $pass, failed $fail"
[ $fail -eq 0 ]
