"""gather_hist: data-dependent addressing and a cross-core histogram.

  out[p, i]  = table[idx[p, i]]       table: 4096 float32 entries stored as a [32, 128] tile (entry k at
                                      partition k // 128, element k % 128) - gathers across partitions
  hist[0, b] = #{idx >> 6 == b}       64 bins; each core counts its partitions in local RAM, then adds its
                                      counts into the shared SBUF output under gp_lock (software mutex)
"""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

import gpsimd_nki as gp

OPS = ["op.cpp"]
P, F = 128, 64


@nki.jit
def kernel(idx, table):
    out = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.shared_hbm)
    hist = nl.ndarray((1, 64), dtype=nl.int32, buffer=nl.shared_hbm)
    ti = nl.ndarray((P, F), dtype=nl.int32, buffer=nl.sbuf)
    tt = nl.ndarray((32, 128), dtype=nl.float32, buffer=nl.sbuf)
    to = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    th = nl.ndarray((1, 64), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ti, src=idx)
    nisa.dma_copy(dst=tt, src=table)
    gp.call("gather_hist", ins=[ti, tt], outs=[to, th])
    nisa.dma_copy(dst=out, src=to)
    nisa.dma_copy(dst=hist, src=th)
    return out, hist


def make_inputs(rng):
    idx = (rng.integers(0, 4096, (P, F)) ** 2 // 4096).astype(np.int32)     # skewed distribution
    return {"idx": idx, "table": rng.normal(0, 1, (32, 128)).astype(np.float32)}


def reference(idx, table):
    return [table.ravel()[idx], np.bincount((idx >> 6).ravel(), minlength=64).reshape(1, 64).astype(np.int32)]
