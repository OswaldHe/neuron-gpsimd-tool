"""global_reduce: reductions over a whole tile, computed cooperatively by the 8 cores.

Each core reduces its block of partitions; the partial results are combined with the runtime's
all-reduce (gp_allreduce_*), which exchanges them through the call's SBUF scratch tile.
Outputs: int32 [sum (wrap), max, min, argmax (first flat index of the max), negatives count, 0, 0, 0]
         float32 [sum, max]  - the float sum is deterministic: each core adds its partitions in index
         order, then the 8 partial sums are added in core order (the reference does exactly this).
"""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

import gpsimd_nki as gp

OPS = ["op.cpp"]
P, F = 128, 64


@nki.jit
def kernel(x, xf):
    oi = nl.ndarray((1, 8), dtype=nl.int32, buffer=nl.shared_hbm)
    of = nl.ndarray((1, 2), dtype=nl.float32, buffer=nl.shared_hbm)
    tx = nl.ndarray((P, F), dtype=nl.int32, buffer=nl.sbuf)
    txf = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    toi = nl.ndarray((1, 8), dtype=nl.int32, buffer=nl.sbuf)
    tof = nl.ndarray((1, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tx, src=x)
    nisa.dma_copy(dst=txf, src=xf)
    gp.call("global_reduce", ins=[tx, txf], outs=[toi, tof])
    nisa.dma_copy(dst=oi, src=toi)
    nisa.dma_copy(dst=of, src=tof)
    return oi, of


def make_inputs(rng):
    x = rng.integers(-2**31, 2**31, (P, F), dtype=np.int64).astype(np.int32)
    x[77, 5] = 2**31 - 1
    x[90, 1] = 2**31 - 1          # duplicate max: argmax must report the first one
    xf = (rng.normal(0, 1, (P, F)) * np.exp2(rng.integers(-20, 20, (P, F)))).astype(np.float32)
    return {"x": x, "xf": xf}


def reference(x, xf):
    flat = x.ravel()
    oi = np.array([[flat.astype(np.int64).sum() & 0xFFFFFFFF, flat.max(), flat.min(), int(np.argmax(flat)),
                    int((flat < 0).sum()), 0, 0, 0]], dtype=np.int64).astype(np.uint32).view(np.int32)
    per = P // 8
    partial = []
    for k in range(8):
        s = np.float32(0)
        for p in range(k * per, (k + 1) * per):
            for i in range(F):
                s = np.float32(s + xf[p, i])
        partial.append(s)
    tot = partial[0]
    for k in range(1, 8):
        tot = np.float32(tot + partial[k])
    of = np.array([[tot, xf.max()]], dtype=np.float32)
    return [oi, of]
