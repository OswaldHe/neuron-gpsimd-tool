"""softmax_rows: numerically stable softmax over the free dimension of every partition (float32).

y[p, :] = exp(x[p, :] - max(x[p, :])) / sum(exp(...)), partitions split over the 8 cores. Summation in
index order; the reference performs the same float32 operations in the same order (bit-exact).
"""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

import gpsimd_nki as gp
import gpsimd_math_ref as mref

OPS = ["op.cpp"]
P, F = 128, 128


@nki.jit
def kernel(x):
    out = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.shared_hbm)
    tx = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    ty = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tx, src=x)
    gp.call("softmax_rows", ins=[tx], outs=[ty])
    nisa.dma_copy(dst=out, src=ty)
    return out


def make_inputs(rng):
    x = (rng.normal(0, 3, (P, F))).astype(np.float32)
    x[5, :] = 0.0
    x[6, 3] = 80.0
    return {"x": x}


def reference(x):
    m = x.max(axis=1, keepdims=True)
    e = mref.exp(x - m)
    s = np.zeros((P, 1), np.float32)
    for i in range(F):
        s = (s + e[:, i:i + 1]).astype(np.float32)
    return [(e / s).astype(np.float32)]
