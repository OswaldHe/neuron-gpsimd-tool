"""atomics: cross-core cooperation on the 8 GPSIMD cores.

  * atomic increments of one shared SBUF word under a software lock (Lamport bakery in SBUF; the
    core's L32EX/S32EX exclusive access hangs on trn3): core k adds 1 (k + 1) * 100 times -> 3600
  * gp_barrier between phases, all-reduce (sum, max, min) and exclusive scan over the cores
  * an atomic fetch-and-add work queue: cores claim rows of the input dynamically
Output row 0 (int32): [counter, 0, allreduce_sum, allreduce_max, allreduce_min,
                       exscan(core 3), total, rows_claimed_total, then per-core claimed row counts are
                       not deterministic and only checked to sum to P], row 1..: per-row checksums of the
                       dynamically claimed rows (deterministic: each row processed exactly once).
"""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

import gpsimd_nki as gp

OPS = ["op.cpp"]
ITERS = 3
P, F = 128, 32


@nki.jit
def kernel(x):
    out = nl.ndarray((1, 8), dtype=nl.int32, buffer=nl.shared_hbm)
    sums = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    tx = nl.ndarray((P, F), dtype=nl.int32, buffer=nl.sbuf)
    to = nl.ndarray((1, 8), dtype=nl.int32, buffer=nl.sbuf)
    ts = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tx, src=x)
    gp.call("atomics", ins=[tx], outs=[to, ts])
    nisa.dma_copy(dst=out, src=to)
    nisa.dma_copy(dst=sums, src=ts)
    return out, sums


def make_inputs(rng):
    return {"x": rng.integers(-1000, 1000, (P, F)).astype(np.int32)}


def reference(x):
    vals = [(k * 37 + 11) % 100 for k in range(8)]          # per-core values used for the reductions
    out = np.array([[sum((k + 1) * 100 for k in range(8)), 0, sum(vals), max(vals), min(vals),
                     sum(vals[:3]), sum(vals), P]], dtype=np.int32)
    sums = x.astype(np.int64).sum(axis=1, keepdims=True).astype(np.int32)
    return [out, sums]
