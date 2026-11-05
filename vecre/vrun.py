"""vrun: run test bundles in the trn3 GPSIMD sandbox (toolchain/probes/vec) and decode the results.
usage (library): results = run(tests) with tests = [(hex, regs, pattern)] (see probes/vec/kernel.py)"""
import os, subprocess, sys, tempfile, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.path.join(ROOT, "toolchain/probes/vec/build")
sys.path.insert(0, os.path.join(ROOT, "toolchain/probes/vec")); sys.path.insert(0, os.path.join(ROOT, "toolchain/python"))
from kernel import build_input  # noqa: E402

CAUSE = {1: "instr", 2: "addr", 3: "external", 4: "debug", 5: "syscall", 6: "hw", 7: "mem", 8: "cp-disabled"}


OUT_SBUF_OFF = 17664          # output tile offset in SBUF (build/sites_resolved.json)
TPB = 4


def read_sbuf_rows(n):
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import devmem
    dev = devmem.Dev(0)
    rows = np.zeros((n, 64), np.uint32)
    for t in range(n):
        rows[t] = np.array(dev.tpb_read(TPB, t * 0x40000 + OUT_SBUF_OFF, 64), np.uint32)
    return rows


def _run_chunk(chunk, timeout):
    d = tempfile.mkdtemp(prefix="vrun_", dir=os.environ.get("VRUN_TMP", "/tmp"))
    np.save(f"{d}/x.npy", build_input(chunk))
    p = subprocess.run([f"{ROOT}/toolchain/bin/gpsimd-run", f"{B}/final.neff", "--in", f"x={d}/x.npy",
                        "--sites", "1", "--save", f"{d}/out"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, timeout=timeout,
                       env={**os.environ, "NEURON_RT_EXEC_TIMEOUT": os.environ.get("NEURON_RT_EXEC_TIMEOUT", "4")})
    if os.path.exists(f"{d}/out/output_0.npy") and p.returncode == 0:
        o = np.load(f"{d}/out/output_0.npy").view(np.uint32).reshape(128, 64)
        bad = [t for t in range(len(chunk)) if o[t, 0] in (3, 5)]
        return o, (bad[0] if bad else None)
    rows = read_sbuf_rows(128)
    started = [t for t in range(len(chunk)) if rows[t, 0] in (3, 5)]
    if len(started) == 1:
        return rows, started[0]
    done = [t for t in range(len(chunk)) if rows[t, 0] in (1, 2)]
    if not started and done:
        # the hang came after a completed test: blame the last completed test (it left bad state behind)
        t = max(done)
        rows[t, 0] = 6
        return rows, t
    raise RuntimeError(f"run failed and no unique started row ({started}):\n" + p.stdout[-2000:])


def run(tests, timeout=180):
    out = []
    i = 0
    while i < len(tests):
        chunk = tests[i:i + 127]
        o, hang = _run_chunk(chunk, timeout)
        n = len(chunk) if hang is None else hang
        if not o[127, 0] and hang is None:
            raise RuntimeError(f"exception table not found: {[hex(x) for x in o[127, :6]]}")
        diag = o[127] if hang is None else np.zeros(64, np.uint32)
        for t, tc in enumerate(chunk[:n]):
            r = o[t]
            sb = int(r[4])
            regs = {k: int(r[16 + k]) for k in range(3, 16)}
            out.append({"hex": tc[0], "status": int(r[0]), "cause": int(r[1]), "epc": int(r[2]), "vaddr": int(r[3]),
                        "sbase": sb, "nchanged": int(r[5]), "first": int(r[6]), "last": int(r[7]), "regs": regs,
                        "mem": [int(x) for x in r[32:64]], "diag": [int(x) for x in diag[:6]],
                        "vmask": int(r[9]), "vregs": [int(r[10]), int(r[11])]})
        if hang is not None:
            out.append({"hex": chunk[hang][0], "status": {5: 5, 6: 6}.get(int(o[hang, 0]), 4), "cause": 0, "epc": 0, "vaddr": 0, "sbase": 0, "nchanged": 0,
                        "first": 0, "last": 0, "regs": {k: 0 for k in range(3, 16)}, "mem": [0] * 32, "diag": []})
            print(f"vrun: {({5: 'CORRUPTION', 6: 'BREAKS-NEXT'}).get(int(o[hang, 0]), 'HANG')} at test {i + hang}: {chunk[hang][0]}", file=sys.stderr)
            n += 1
        i += n
    return out


def vpat(k, i):
    x = ((k * 16 + i + 1) * 0x9E3779B9) & 0xFFFFFFFF
    return x ^ (x >> 15)


def regs_init(r, sb, tc):
    init = {}
    for k in range(3, 16):
        v = tc[1].get(k, 0x80000000 | (0x1000 + 0x100 * k))
        init[k] = sb + (v & 0xffff) if (v & 0xffff0000) == 0x80000000 else v
    return init


def show(results, tests):
    for r, tc in zip(results, tests):
        init = regs_init(r, r["sbase"], tc)
        ch = {f"a{k}": f"{init[k] - r['sbase']:+#x}->{r['regs'][k] - r['sbase']:+#x}" if abs(init[k] - r['sbase']) < 0x10000
              else f"{init[k]:#x}->{r['regs'][k]:#x}" for k in range(3, 16) if r["regs"][k] != init[k]}
        st = {6: "BREAKS-NEXT", 5: "CORRUPTS", 4: "HANG", 1: "ok", 2: f"EXC {CAUSE.get(r['cause'], r['cause'])} pc+{r['epc']} vaddr={r['vaddr']:#x}"}.get(r["status"], f"status{r['status']}")
        mem = f" mem[{r['first']:#x}..{r['last']:#x}] {r['nchanged']}B" if r["nchanged"] else ""
        print(f"{r['hex']:34} {st}{mem} {ch}")


if __name__ == "__main__":
    tests = [(h, {}, None) for h in sys.argv[1:]]
    res = run(tests)
    show(res, tests)
    print("diag", [hex(x) for x in res[0]["diag"]])
