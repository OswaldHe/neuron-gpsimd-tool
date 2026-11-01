"""vadd_i32: c = a + b (int32, wrap-around). Two inputs, partitions split across the 8 GPSIMD cores."""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

import gpsimd_nki as gp

OPS = ["op.cpp"]
P, F = 128, 512


@nki.jit
def kernel(a, b):
    out = nl.ndarray((P, F), dtype=nl.int32, buffer=nl.shared_hbm)
    ta = nl.ndarray((P, F), dtype=nl.int32, buffer=nl.sbuf)
    tb = nl.ndarray((P, F), dtype=nl.int32, buffer=nl.sbuf)
    tc = nl.ndarray((P, F), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ta, src=a)
    nisa.dma_copy(dst=tb, src=b)
    gp.call("vadd_i32", ins=[ta, tb], outs=[tc])
    nisa.dma_copy(dst=out, src=tc)
    return out


def make_inputs(rng):
    return {"a": rng.integers(-2**31, 2**31, (P, F), dtype=np.int64).astype(np.int32),
            "b": rng.integers(-2**31, 2**31, (P, F), dtype=np.int64).astype(np.int32)}


def reference(a, b):
    return [(a.astype(np.int64) + b).astype(np.int32)]
