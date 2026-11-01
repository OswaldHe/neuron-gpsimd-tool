"""Compile an example's NKI kernel and write its test data (run with the Neuron venv python).

usage: python gpsimd_nki_build.py <example_dir> <build_dir> <seed>

The example's kernel.py defines:
    kernel                         @nki.jit kernel that calls gp.call(...)
    make_inputs(rng) -> dict       name -> numpy array, in the kernel's parameter order
    reference(**inputs) -> list    expected outputs (numpy arrays, bit-exact), in return order
    OPS = ["op.cpp", ...]          operator sources (relative to the example dir)
    CFLAGS = [...]                 optional extra gpsimd-cc flags
    HBM = ["tensor", ...]          optional: tensors whose device memory gpsimd-run maps for the GPSIMD
                                   (core addresses passed in the int32 input tensor gp_hbm, see gp::hbm)
    ITERS = 2                      optional: executions per run
bfloat16 arrays (ml_dtypes.bfloat16) are stored as uint16 bit patterns.
"""
import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpsimd_nki as gp  # noqa: E402


def as_storable(a):
    a = np.ascontiguousarray(a)
    if a.dtype.name == "bfloat16":
        return a.view(np.uint16)
    return a


def main():
    ex, build, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    ex = os.path.abspath(ex)
    os.makedirs(build, exist_ok=True)
    sys.path.insert(0, ex)
    spec = importlib.util.spec_from_file_location("gpsimd_example", os.path.join(ex, "kernel.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rng = np.random.default_rng(seed)
    inputs = mod.make_inputs(rng)
    gp.compile(mod.kernel, list(inputs.values()), build)
    outs = mod.reference(**inputs)
    if not isinstance(outs, (list, tuple)):
        outs = [outs]
    man = {"example": os.path.basename(ex), "seed": seed, "inputs": {}, "outputs": {},
           "ops": [os.path.join(ex, o) for o in getattr(mod, "OPS", ["op.cpp"])],
           "cflags": list(getattr(mod, "CFLAGS", [])), "iters": int(getattr(mod, "ITERS", 2)),
           "hbm": list(getattr(mod, "HBM", []))}
    os.makedirs(os.path.join(build, "data"), exist_ok=True)
    for name, a in inputs.items():
        f = os.path.join(build, "data", f"in_{name}.npy")
        np.save(f, as_storable(a))
        man["inputs"][name] = f
    for j, a in enumerate(outs):
        f = os.path.join(build, "data", f"expected_output_{j}.npy")
        np.save(f, as_storable(a))
        man["outputs"][f"output_{j}"] = f
    meta = json.load(open(os.path.join(build, "sites.json")))
    man["sites"] = len(meta["sites"])
    json.dump(man, open(os.path.join(build, "manifest.json"), "w"), indent=1)
    print(f"[nki] {os.path.join(build, 'kernel.neff')}: {man['sites']} GPSIMD call site(s); "
          f"inputs {list(man['inputs'])}, outputs {list(man['outputs'])}")


if __name__ == "__main__":
    main()
