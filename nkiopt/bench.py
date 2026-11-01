#!/usr/bin/env python3
"""bench.py - build, verify and profile a pure-NKI baseline against its GPSIMD-assisted version.

usage: bench.py <case_dir> [--repeats R] [--iters N] [--only baseline|gpsimd] [--no-build] [--seed S]

A case directory holds case.py with:
    baseline_kernel        @nki.jit kernel written with NKI only (the best NKI formulation we know)
    gpsimd_kernel          @nki.jit kernel that calls gp.call(...) for the GPSIMD custom operator(s)
    make_inputs(rng)       dict name -> numpy array (kernel parameter order)
    reference(**inputs)    list of expected outputs (numpy), in return order
    check(outs, inputs)    optional: (ok, message) for outputs that need not be bit-exact
                           (default: every output bit-exact against reference)
    OPS, CFLAGS, HBM       as for toolchain examples (operator sources, gpsimd-cc flags, mapped tensors)

Both NEFFs run through the same runner (toolchain/bin/gpsimd-run, libnrt) with the same inputs.
Each repeat executes the NEFF --iters times and captures a device profile of the last (warm)
execution with nrt_profile_start/stop; `neuron-explorer view --output-format summary-json` then
reads total_exec_time out of the NTFF. The GPSIMD run also checks the per-core receipts, so a
result cannot come from anywhere but the GPSIMD cores.
Writes <case>/build/results.json.
"""
import argparse
import importlib.util
import json
import os
import shutil
import statistics
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TC = os.path.join(os.path.dirname(HERE), "toolchain")
BIN = os.path.join(TC, "bin")
sys.path.insert(0, os.path.join(TC, "python"))
EXPLORER = "/opt/aws/neuron/bin/neuron-explorer"


def load_case(case_dir):
    sys.path.insert(0, case_dir)
    spec = importlib.util.spec_from_file_location("case", os.path.join(case_dir, "case.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def storable(a):
    a = np.ascontiguousarray(a)
    return a.view(np.uint16) if a.dtype.name == "bfloat16" else a


def sh(cmd, **kw):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kw)
    if p.returncode:
        sys.stdout.write(p.stdout[-6000:])
        raise SystemExit(f"bench: command failed ({p.returncode}): {' '.join(cmd)}")
    return p.stdout


def build(case, case_dir, out, which, inputs):
    import gpsimd_nki as gp
    d = os.path.join(out, which)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    kern = case.baseline_kernel if which == "baseline" else case.gpsimd_kernel
    gp.compile(kern, list(kernel_inputs(case, which, inputs).values()), d)
    sites = json.load(open(os.path.join(d, "sites.json")))["sites"]
    if which == "baseline":
        if sites:
            raise SystemExit("bench: baseline_kernel must not call gp.call")
        shutil.copy(os.path.join(d, "kernel.neff"), os.path.join(d, "final.neff"))
        return d, 0
    py = [sys.executable, "-I"]
    sh(py + [os.path.join(BIN, "gpsimd-neff"), "analyze", f"{d}/kernel.neff", f"{d}/sites.json", "-o", f"{d}/sites_resolved.json"])
    ops = [os.path.join(case_dir, o) for o in getattr(case, "OPS", ["op.cpp"])]
    print(sh(py + [os.path.join(BIN, "gpsimd-cc"), "--sites", f"{d}/sites_resolved.json", "-o", f"{d}/ops.so"] + ops
             + list(getattr(case, "CFLAGS", []))).strip())
    sh(py + [os.path.join(BIN, "gpsimd-neff"), "inject", f"{d}/kernel.neff", f"{d}/sites_resolved.json", f"{d}/ops.so",
             "-o", f"{d}/final.neff"])
    return d, len(sites)


def kernel_inputs(case, which, inputs):
    """the inputs a version's kernel takes (case.BASELINE_SKIP_INPUTS: inputs only the GPSIMD version uses)"""
    if hasattr(case, "INPUTS"):                       # explicit per-version input lists
        return {k: inputs[k] for k in case.INPUTS[which]}
    skip = set(getattr(case, "BASELINE_SKIP_INPUTS", [])) if which == "baseline" else set()
    return {k: v for k, v in inputs.items() if k not in skip}


def total_exec_time(neff, ntff):
    p = subprocess.run([EXPLORER, "view", "-n", neff, "-s", ntff, "--output-format", "summary-json"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=900)
    t = p.stdout
    doc = json.loads(t[t.find("{"):t.rfind("}") + 1])
    s = next(iter(doc.values()))
    keep = ["total_exec_time", "tensor_engine_active_time", "scalar_engine_active_time", "vector_engine_active_time",
            "gpsimd_engine_active_time", "sync_engine_active_time", "dma_active_time",
            "gpsimd_engine_instruction_count", "vector_engine_instruction_count", "scalar_engine_instruction_count",
            "tensor_engine_instruction_count", "sync_engine_instruction_count", "dma_transfer_count"]
    return {k: s.get(k) for k in keep}


def run(case, d, nsites, inputs, ref, iters, repeats, out_names):
    datad = os.path.join(d, "data")
    os.makedirs(datad, exist_ok=True)
    args = []
    for k, a in kernel_inputs(case, "gpsimd" if nsites else "baseline", inputs).items():
        f = os.path.join(datad, f"in_{k}.npy")
        np.save(f, storable(a))
        args += ["--in", f"{k}={f}"]
    for h in getattr(case, "HBM", []) if nsites else []:
        args += ["--hbm", h]
    args += ["--sites", str(nsites), "--iters", str(iters)]
    if not nsites:
        args += ["--no-map"]
    samples, summ = [], None
    for r in range(repeats):
        ntff = os.path.join(d, f"r{r}.ntff")
        save = os.path.join(d, "out")
        shutil.rmtree(save, ignore_errors=True)
        log = sh([os.path.join(BIN, "gpsimd-run"), os.path.join(d, "final.neff")] + args + ["--save", save, "--profile", ntff])
        if "RESULT: PASS" not in log:
            sys.stdout.write(log[-3000:])
            raise SystemExit("bench: run failed")
        if r == 0:   # correctness on the first repeat's outputs
            outs = []
            for j, name in enumerate(out_names):
                raw = np.load(os.path.join(save, f"{name}.npy"))
                want = ref[j]
                outs.append(raw.view(storable(want).dtype).reshape(want.shape).view(want.dtype))
            ok, msg = check(case, outs, inputs, ref, "gpsimd" if nsites else "baseline")
            print(f"    correctness: {'PASS' if ok else 'FAIL'}  {msg}")
            if not ok:
                raise SystemExit("bench: wrong outputs")
            gp_line = [l for l in log.splitlines() if l.startswith("  GPSIMD:")]
            if gp_line:
                print("   " + gp_line[0])
        s = total_exec_time(os.path.join(d, "final.neff"), ntff)
        if os.environ.get("KEEP_NTFF") and r == repeats - 1:
            os.replace(ntff, os.path.join(d, "profile.ntff"))   # the last repeat's profile, for neuron-explorer
        else:
            os.remove(ntff)
        samples.append(s["total_exec_time"] * 1e6)
        summ = s
        print(f"    repeat {r}: total_exec_time {samples[-1]:9.3f} us")
    return samples, summ


def check(case, outs, inputs, ref, which):
    if hasattr(case, "check"):
        import inspect
        if len(inspect.signature(case.check).parameters) >= 3:
            return case.check(outs, inputs, which)
        return case.check(outs, inputs)
    msgs = []
    for j, (o, r) in enumerate(zip(outs, ref)):
        if o.tobytes() != np.ascontiguousarray(r).tobytes():
            bad = int(np.sum(storable(o) != storable(r)))
            return False, f"output_{j}: {bad} element(s) differ from the reference"
        msgs.append(f"output_{j} {r.size}/{r.size} bit-exact")
    return True, ", ".join(msgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--only", choices=["baseline", "gpsimd"])
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="environment variable for case.py (problem size etc.); results go to build_<tag>")
    ap.add_argument("--tag", help="build directory suffix (default: from --set values)")
    a = ap.parse_args()
    for kv in a.set:
        k, v = kv.split("=", 1)
        os.environ[k] = v
    tag = a.tag or "_".join(kv.replace("=", "") for kv in a.set)
    case_dir = os.path.abspath(a.case)
    case = load_case(case_dir)
    out = os.path.join(case_dir, "build" + (f"_{tag}" if tag else ""))
    os.makedirs(out, exist_ok=True)
    inputs = case.make_inputs(np.random.default_rng(a.seed))
    ref = case.reference(**inputs)
    ref = list(ref) if isinstance(ref, (list, tuple)) else [ref]
    out_names = [f"output_{j}" for j in range(len(ref))]
    res_path = os.path.join(out, "results.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    results["case"] = os.path.basename(case_dir) + (f" [{tag}]" if tag else "")
    results["params"] = getattr(case, "PARAMS", {})
    for which in ["baseline", "gpsimd"]:
        if a.only and which != a.only:
            continue
        print(f"[{which}]")
        if a.no_build:
            d = os.path.join(out, which)
            nsites = len(json.load(open(os.path.join(d, "sites.json")))["sites"])
        else:
            d, nsites = build(case, case_dir, out, which, inputs)
        samples, summ = run(case, d, nsites, inputs, ref, a.iters, a.repeats, out_names)
        med = statistics.median(samples)
        results[which] = {"total_exec_time_us": {"median": round(med, 3), "min": round(min(samples), 3),
                                                 "max": round(max(samples), 3), "samples": [round(x, 3) for x in samples]},
                          "profile": summ, "gpsimd_sites": nsites}
        print(f"  {which}: median total_exec_time {med:.3f} us  (min {min(samples):.3f}, max {max(samples):.3f})")
    if "baseline" in results and "gpsimd" in results:
        b, g = results["baseline"]["total_exec_time_us"]["median"], results["gpsimd"]["total_exec_time_us"]["median"]
        results["speedup"] = round(b / g, 3)
        print(f"== {results['case']}: baseline {b:.3f} us -> GPSIMD {g:.3f} us  ({b / g:.2f}x)")
    json.dump(results, open(res_path, "w"), indent=1)


if __name__ == "__main__":
    main()
