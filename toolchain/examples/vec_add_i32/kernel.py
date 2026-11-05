"""vec_add_i32: c = a + b (int32) with the trn3 GPSIMD vector unit (gpsimd/vec.h): 16 lanes per instruction,
operating directly on the SBUF tiles. Same NKI kernel as vadd_i32 (scalar), for comparison."""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

import gpsimd_nki as gp

OPS = ["op.cpp"]
CFLAGS = ["--allow-vector"] + [f"-D{x}" for x in __import__("os").environ.get("VEC_DEFS", "").split() if x]
P, F = 128, 512


@nki.jit
def kernel(a, b):
    out = nl.ndarray((P, F), dtype=nl.int32, buffer=nl.shared_hbm)
    ta = nl.ndarray((P, F), dtype=nl.int32, buffer=nl.sbuf)
    tb = nl.ndarray((P, F), dtype=nl.int32, buffer=nl.sbuf)
    tc = nl.ndarray((P, F), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ta, src=a)
    nisa.dma_copy(dst=tb, src=b)
    gp.call("vec_add_i32", ins=[ta, tb], outs=[tc])
    nisa.dma_copy(dst=out, src=tc)
    return out


def make_inputs(rng):
    return {"a": rng.integers(-2**31, 2**31, (P, F), dtype=np.int64).astype(np.int32),
            "b": rng.integers(-2**31, 2**31, (P, F), dtype=np.int64).astype(np.int32)}


def reference(a, b):
    return [(a.astype(np.int64) + b).astype(np.int32)]
