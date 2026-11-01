"""transpose: cross-partition data movement inside SBUF.

out32[j, i] = x32[i, j] for a 128 x 128 int32 tile, and outb[j, i] = xb[i, j] for a 64 x 96 bf16
tile (96 x 64 result). Core k writes its block of output partitions, reading a column across all
input partitions (a strided access pattern no NeuronCore engine instruction expresses directly).
"""
import ml_dtypes
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

import gpsimd_nki as gp

OPS = ["op.cpp"]


@nki.jit
def kernel(x32, xb):
    o32 = nl.ndarray((128, 128), dtype=nl.int32, buffer=nl.shared_hbm)
    ob = nl.ndarray((96, 64), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    t32 = nl.ndarray((128, 128), dtype=nl.int32, buffer=nl.sbuf)
    tb = nl.ndarray((64, 96), dtype=nl.bfloat16, buffer=nl.sbuf)
    u32 = nl.ndarray((128, 128), dtype=nl.int32, buffer=nl.sbuf)
    ub = nl.ndarray((96, 64), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=t32, src=x32)
    nisa.dma_copy(dst=tb, src=xb)
    gp.call("transpose", ins=[t32, tb], outs=[u32, ub])
    nisa.dma_copy(dst=o32, src=u32)
    nisa.dma_copy(dst=ob, src=ub)
    return o32, ob


def make_inputs(rng):
    return {"x32": rng.integers(-2**31, 2**31, (128, 128), dtype=np.int64).astype(np.int32),
            "xb": rng.normal(0, 10, (64, 96)).astype(np.float32).astype(ml_dtypes.bfloat16)}


def reference(x32, xb):
    return [np.ascontiguousarray(x32.T), np.ascontiguousarray(xb.T)]
