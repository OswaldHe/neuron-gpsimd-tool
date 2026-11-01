#!/bin/bash
# probe_run.sh <probe.c> [extra cflags...]: build, inject into the NKI test NEFF, run with a 30 s
# timeout, then print kprint logs (in-process on success, post-mortem over BAR0 after a hang).
set -e
R=$(cd "$(dirname "$0")/.." && pwd); W=${TMPDIR:-/tmp}/trn3probe; mkdir -p $W
src=$1; shift; name=$(basename "$src" .c)
EXTRA_CFLAGS="-DKLOG_SIZE=${KLOG_SIZE:-2048} $*" $R/kl/build.sh "$src" $W/$name.so | grep -v '^note: GNU' || true
PY=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/python
$PY -I $R/tools/inject_kl.py $R/nki/${NEFF:-kernel_trn3.neff} $W/$name.so ${KERNEL:-dbg_kernel} $W/$name.neff ${INJECT_ARGS} >/dev/null
set +e
NEURON_RT_VISIBLE_CORES=0 NEURON_RT_LOG_LEVEL=ERROR NEURON_RT_EXEC_TIMEOUT=${EXEC_TIMEOUT:-30} timeout 600 \
  $PY -I $R/tools/run_neff.py $W/$name.neff --klog ${RUN_ARGS} > $W/$name.run.log 2>&1
rc=$?
grep -E '^(tensor|output_0|---|nrt_)|^[^0-9]' $W/$name.run.log | grep -vE 'NRT:|NMGR|TDRV|HAL:|^\s*$|cut here' | head -60
if ! grep -q '^---' $W/$name.run.log; then echo "[run rc=$rc; reading logs post-mortem]"; python3 -I $R/tools/klog.py --tpb 4; fi
