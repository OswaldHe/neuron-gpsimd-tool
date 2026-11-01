"""prefix_scan: inclusive prefix sum of one long row (1 partition x 8192 int32), element-split.

Core k scans its contiguous block of 1024 elements locally, the cores exchange their block totals
(gp_exscan_add_u32: exclusive scan across cores through SBUF), and each core adds its offset.
Output: inclusive scan (int32 wrap-around) and, as a second output, the scan of the int16 input
x16 in a 3-D view tile [1, 64, 128] to exercise multi-dimensional free-axis indexing.
"""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

import gpsimd_nki as gp

OPS = ["op.cpp"]
N = 8192


@nki.jit
def kernel(x, x16):
    out = nl.ndarray((1, N), dtype=nl.int32, buffer=nl.shared_hbm)
    out16 = nl.ndarray((1, 64, 128), dtype=nl.int32, buffer=nl.shared_hbm)
    tx = nl.ndarray((1, N), dtype=nl.int32, buffer=nl.sbuf)
    tx16 = nl.ndarray((1, 64, 128), dtype=nl.int16, buffer=nl.sbuf)
    to = nl.ndarray((1, N), dtype=nl.int32, buffer=nl.sbuf)
    to16 = nl.ndarray((1, 64, 128), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tx, src=x)
    nisa.dma_copy(dst=tx16, src=x16)
    gp.call("prefix_scan", ins=[tx, tx16], outs=[to, to16])
    nisa.dma_copy(dst=out, src=to)
    nisa.dma_copy(dst=out16, src=to16)
    return out, out16


def make_inputs(rng):
    return {"x": rng.integers(-2**31, 2**31, (1, N), dtype=np.int64).astype(np.int32),
            "x16": rng.integers(-2**15, 2**15, (1, 64, 128), dtype=np.int64).astype(np.int16)}


def reference(x, x16):
    s = (np.cumsum(x.astype(np.int64), axis=1) & 0xFFFFFFFF).astype(np.uint32).view(np.int32)
    s16 = np.cumsum(x16.astype(np.int64).reshape(1, -1), axis=1).reshape(1, 64, 128).astype(np.int32)
    return [s, s16]
