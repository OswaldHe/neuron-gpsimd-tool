"""transcendental: float32 elementary functions on the GPSIMD (include/gpsimd/math.hpp).

exp, log, tanh, sigmoid, sin, cos, gelu (tanh form), rsqrt on random arguments plus special values.
The reference is python/gpsimd_math_ref.py, the same algorithms in numpy float32 operations, so the
comparison is bit-exact; the build also prints each function's max error in ulp vs float64 numpy.
"""
import os
import sys

import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

import gpsimd_nki as gp
import gpsimd_math_ref as mref

OPS = ["op.cpp"]
P, F = 128, 32
FUNCS = ["exp", "log", "tanh", "sigmoid", "sin", "cos", "gelu", "rsqrt"]


@nki.jit
def kernel(x, y):
    outs = [nl.ndarray((P, F), dtype=nl.float32, buffer=nl.shared_hbm) for _ in FUNCS]
    tx = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    ty = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tx, src=x)
    nisa.dma_copy(dst=ty, src=y)
    touts = [nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf) for _ in FUNCS]
    gp.call("transcendental", ins=[tx, ty], outs=touts)
    for o, t in zip(outs, touts):
        nisa.dma_copy(dst=o, src=t)
    return tuple(outs)


def make_inputs(rng):
    x = rng.uniform(-12, 12, (P, F)).astype(np.float32)
    x = np.where(rng.random((P, F)) < 0.2, rng.uniform(-100, 100, (P, F)), x).astype(np.float32)
    sp = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 88.7, 88.8, -103.9, -104.5, -87.5, 1e-30, -1e-30, 0.625,
                   -0.625, 44.4, 3.14159265, 1.5707964, 1e-8], np.float32)
    x = np.where(rng.random((P, F)) < 0.08, sp[rng.integers(0, len(sp), (P, F))], x).astype(np.float32)
    y = np.exp(rng.uniform(-80, 80, (P, F))).astype(np.float32)
    ys = np.array([0.0, -0.0, np.inf, np.nan, -1.0, 1.0, 1e-45, 1e-40, 3.4e38, 0.70710677, 0.7071068], np.float32)
    y = np.where(rng.random((P, F)) < 0.08, ys[rng.integers(0, len(ys), (P, F))], y).astype(np.float32)
    return {"x": x, "y": y}


def _ulp_report(name, got, arg, truth_fn):
    with np.errstate(all="ignore"):
        t = truth_fn(arg.astype(np.float64))
        ok = np.isfinite(t) & np.isfinite(got) & (np.abs(t) > 1e-30) & (np.abs(t) < 3e38)
        if not ok.any():
            return
        tf = t[ok]
        mag = np.maximum(np.abs(tf), 1.0) if name == "gelu" else tf     # gelu: absolute error (cancellation for x << 0)
        ulp = np.abs(np.spacing(np.abs(mag).astype(np.float32)).astype(np.float64))
        err = np.abs(got[ok].astype(np.float64) - tf) / ulp
        unit = "ulp(max(|f|,1))" if name == "gelu" else "ulp"
        print(f"  {name:8s} max error {err.max():5.2f} {unit}, mean {err.mean():.3f} ({ok.sum()} finite samples)",
              file=sys.stderr)


def reference(x, y):
    outs = [mref.exp(x), mref.log(y), mref.tanh(x), mref.sigmoid(x), mref.sin(x), mref.cos(x), mref.gelu(x),
            mref.rsqrt(y)]
    truth = {"exp": np.exp, "log": np.log, "tanh": np.tanh, "sigmoid": lambda v: 1 / (1 + np.exp(-v)),
             "sin": np.sin, "cos": np.cos,
             "gelu": lambda v: 0.5 * v * (1 + np.tanh(np.sqrt(2 / np.pi) * (v + 0.044715 * v**3))),
             "rsqrt": lambda v: 1 / np.sqrt(v)}
    print("accuracy vs float64 numpy:", file=sys.stderr)
    for n, o in zip(FUNCS, outs):
        arg = y if n in ("log", "rsqrt") else x
        if n in ("sin", "cos"):
            sel = np.abs(arg) < 8192
            _ulp_report(n, o[sel], arg[sel], truth[n])
        else:
            _ulp_report(n, o, arg, truth[n])
    return outs
