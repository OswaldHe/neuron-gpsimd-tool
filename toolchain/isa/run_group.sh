#!/bin/bash
# run_group.sh GROUPMACRO [extra device cflags]: run a test group on GPSIMD and diff against the host
set -u
H=$(cd "$(dirname "$0")" && pwd); R=$H/../..; G=$1; shift
W=${TMPDIR:-/tmp}/isa_$G; mkdir -p $W
gcc -O2 -ffp-contract=off -DTESTS="$G" -I$H -o $W/host $H/host_ref.c -lm && $W/host > $W/host.txt
KLOG_SIZE=4096 KERNEL=dbg_kernel $R/tools/probe_run.sh $H/isa_probe.c "-I$H -DTESTS=$G -ffp-contract=off $*" > $W/dev.log 2>&1
grep -E '^\s*(\[TPB4 core 0\] )?[a-z0-9_]+:' $W/dev.log | sed -E 's/^.*\] //; s/^\s+//' | grep -v '^---' > $W/dev.txt
python3 - $W/host.txt $W/dev.txt <<'PY'
import sys
host = dict(l.split(":", 1) for l in open(sys.argv[1]).read().splitlines() if ":" in l)
dev_lines = [l for l in open(sys.argv[2]).read().splitlines() if ":" in l]
dev = {}
for l in dev_lines:
    k, v = l.split(":", 1); dev[k.strip()] = v.strip()
for k in host:
    if k not in dev: print(f"  {k:12s} NOT REACHED (hang or missing)"); continue
    d, h = dev[k].split(), host[k].split()
    if len(d) < len(h): print(f"  {k:12s} HANG during test (partial: {dev[k]})"); continue
    bad = [i for i in range(len(h)) if d[i] != h[i]]
    print(f"  {k:12s} {'OK (bit-exact)' if not bad else 'MISMATCH at ' + str(bad) + ' dev ' + ' '.join(d[i] for i in bad) + ' host ' + ' '.join(h[i] for i in bad)}")
PY
