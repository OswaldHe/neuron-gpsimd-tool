"""single_core_sort: a sequential algorithm on one GPSIMD core.

Core 0 copies a 1 x 1024 int32 row from SBUF into its local data RAM (stack), heap-sorts it there
(local RAM is much faster than the SBUF window) and writes it back; cores 1..7 return immediately.
A second output is written by core 7 only: the number of distinct values (shows any core can be the
designated worker).
"""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

import gpsimd_nki as gp

OPS = ["op.cpp"]
N = 1024


@nki.jit
def kernel(x):
    out = nl.ndarray((1, N), dtype=nl.int32, buffer=nl.shared_hbm)
    cnt = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    tx = nl.ndarray((1, N), dtype=nl.int32, buffer=nl.sbuf)
    to = nl.ndarray((1, N), dtype=nl.int32, buffer=nl.sbuf)
    tc = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tx, src=x)
    gp.call("single_core_sort", ins=[tx], outs=[to, tc])
    nisa.dma_copy(dst=out, src=to)
    nisa.dma_copy(dst=cnt, src=tc)
    return out, cnt


def make_inputs(rng):
    return {"x": rng.integers(-500, 500, (1, N)).astype(np.int32)}


def reference(x):
    return [np.sort(x, axis=1), np.array([[len(np.unique(x))]], np.int32)]
