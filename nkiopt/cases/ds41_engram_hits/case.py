"""ds41_engram_hits: the GPSIMD half of ../ds41_engram on its own (no code of the DeepSeek port): the
engram_hits call compacts this rank's n-gram lookups, NKI zero-fills emb and gathers / scatters only those
rows in a runtime-bounded loop. Checks emb exactly against numpy; times the gather stage alone.
Run with --only gpsimd (the hand-optimized baseline lives in ../ds41_engram)."""
import os

import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import oob_mode
import ml_dtypes

import gpsimd_nki as gp

T = int(os.environ.get("T", "1024"))
RANK = int(os.environ.get("RANK", "21"))
WORLD, NCOL, HD, PT = 64, 24, 256, 128
E_IN = NCOL * HD
TP = (T + PT - 1) // PT * PT
NT = TP // PT
CAP = min(TP * NCOL, 4 * TP)


def _primes(n, start=16_000_000 - 1):
    out, c = [], start
    while len(out) < n:
        c += 1
        if all(c % d for d in range(2, int(c ** 0.5) + 1)):
            out.append(c)
    return out


PRIMES = _primes(NCOL)
OFFSETS = np.cumsum([0] + PRIMES[:-1]).astype(np.int64)
ROWW = int(os.environ.get("ROWW", str(-(-int(sum(PRIMES)) // WORLD))))
OPS = ["../ds41_engram/op.cpp"]
CFLAGS = [f"-DEG_NT={NT}", f"-DEG_NCOL={NCOL}", f"-DEG_ROWW={ROWW}", f"-DEG_CAP={CAP}"]
PARAMS = {"T": T, "rank": RANK, "rows_per_rank": ROWW}


@nki.jit
def baseline_kernel(hash_ids, etable):
    raise NotImplementedError("run with --only gpsimd")


@nki.jit
def gpsimd_kernel(hash_ids, etable):
    i32 = nl.int32
    ids_sb = nl.ndarray((PT, NT * NCOL), dtype=i32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ids_sb, src=hash_ids.ap(pattern=[[NCOL, PT], [PT * NCOL, NT], [1, NCOL]], offset=0))
    hits_sb = nl.ndarray((2, CAP), dtype=i32, buffer=nl.sbuf)
    cnt_sb = nl.ndarray((1, 1), dtype=i32, buffer=nl.sbuf)
    gp.call("engram_hits", ins=[ids_sb], outs=[hits_sb, cnt_sb])
    hits = nl.ndarray((2, CAP), dtype=i32, buffer=nl.private_hbm)
    nisa.dma_copy(dst=hits, src=hits_sb)
    emb = nl.ndarray((TP, E_IN), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="emb")
    zero = nl.ndarray((PT, E_IN), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=zero, value=0.0)
    for i in range(NT):
        nisa.dma_copy(dst=emb[i * PT:(i + 1) * PT, :], src=zero)
    rows = emb.reshape((TP * NCOL, HD))
    h3 = hits.reshape((2, CAP // PT, PT))
    reg = nisa.register_alloc()
    nisa.register_load(reg, cnt_sb)

    def chunk(b):
        si = nl.ndarray((PT, 1), dtype=i32, buffer=nl.sbuf)
        nisa.dma_copy(dst=si, src=h3.ap(pattern=[[1, PT], [1, 1]], offset=0, scalar_offset=b, indirect_dim=1))
        di = nl.ndarray((PT, 1), dtype=i32, buffer=nl.sbuf)
        nisa.dma_copy(dst=di, src=h3.ap(pattern=[[1, PT], [1, 1]], offset=CAP, scalar_offset=b, indirect_dim=1))
        g = nl.ndarray((PT, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=g, src=etable.ap(pattern=[[HD, PT], [1, HD]], offset=0, vector_offset=si, indirect_dim=0))
        nisa.dma_copy(dst=rows.ap(pattern=[[HD, PT], [1, HD]], offset=0, vector_offset=di, indirect_dim=0), src=g,
                      oob_mode=oob_mode.skip)

    nl.fori_loop(0, reg, chunk, step=1)
    return emb


def make_inputs(rng):
    cols = [OFFSETS[j] + rng.integers(0, PRIMES[j], T) for j in range(NCOL)]
    ids = (np.stack(cols, 1) - RANK * ROWW).astype(np.int64)
    bits = rng.integers(0, 1 << 16, size=(ROWW, HD), dtype=np.uint16)
    bits &= np.uint16(0x807F)
    bits |= np.uint16(0x3E80)
    return {"hash_ids": ids.astype(np.int32), "etable": bits.view(ml_dtypes.bfloat16)}


def reference(hash_ids, etable):
    ids = hash_ids.astype(np.int64)
    ok = (ids >= 0) & (ids < ROWW)
    emb = np.zeros((TP, NCOL, HD), ml_dtypes.bfloat16)
    t, j = np.nonzero(ok)
    emb[t, j] = etable[ids[t, j]]
    return emb.reshape(TP, E_IN)
