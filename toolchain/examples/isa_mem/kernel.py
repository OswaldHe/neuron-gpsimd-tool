"""isa_mem: every load/store instruction of the base ISA, explicitly (inline asm), on SBUF (through the
core's SBUF window) and on the core's local data RAM.

For each input word w at x[p, i] (and its neighbour w2 = x[p, i ^ 1]) the operator records:
  l8ui at byte offsets 0..3, l16ui and l16si at offsets 0 and 2, l32i, l32ai (acquire), l32r (literal),
  l32i with immediate offset (+4 => the neighbour word in the same partition),
  stores: s8i x4 (byte-reversed word), s16i x2 (swapped halves), s32ri (release) of w ^ 0xA5A5A5A5,
  all of the above again on a copy in local data RAM (stack).
"""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

import gpsimd_nki as gp

OPS = ["op.cpp"]
P, N = 128, 16
NT = 15                       # results per word and memory kind
NK = 2                        # SBUF, local RAM


@nki.jit
def kernel(x):
    out = nl.ndarray((P, NK * NT * N), dtype=nl.uint32, buffer=nl.shared_hbm)
    st = nl.ndarray((P, NK * 3 * N), dtype=nl.uint32, buffer=nl.shared_hbm)
    tx = nl.ndarray((P, N), dtype=nl.uint32, buffer=nl.sbuf)
    to = nl.ndarray((P, NK * NT * N), dtype=nl.uint32, buffer=nl.sbuf)
    ts = nl.ndarray((P, NK * 3 * N), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tx, src=x)
    gp.call("isa_mem", ins=[tx], outs=[to, ts])
    nisa.dma_copy(dst=out, src=to)
    nisa.dma_copy(dst=st, src=ts)
    return out, st


def make_inputs(rng):
    return {"x": rng.integers(0, 2**32, (P, N), dtype=np.uint64).astype(np.uint32)}


def reference(x):
    w = x.astype(np.int64)
    w2 = x[:, np.arange(N) ^ 1].astype(np.int64)
    b = lambda k: (w >> (8 * k)) & 0xFF
    h = lambda k: (w >> (16 * k)) & 0xFFFF
    hs = lambda k: np.where(h(k) >= 0x8000, h(k) - 0x10000, h(k))
    loads = [b(0), b(1), b(2), b(3), h(0), h(1), hs(0), hs(1), w, w, np.full_like(w, 0x12345678), w2,
             b(0) | b(1) << 8 | b(2) << 16 | b(3) << 24, w ^ 0xFFFFFFFF, (w + w2)]
    assert len(loads) == NT
    one = np.concatenate([(np.asarray(v) & 0xFFFFFFFF).astype(np.uint32) for v in loads], axis=1)
    bswap = (b(3) | b(2) << 8 | b(1) << 16 | b(0) << 24)
    hswap = (h(1) | h(0) << 16)
    stores = np.concatenate([bswap.astype(np.uint32), hswap.astype(np.uint32), (w ^ 0xA5A5A5A5).astype(np.uint32)], axis=1)
    return [np.concatenate([one, one], axis=1), np.concatenate([stores, stores], axis=1)]
