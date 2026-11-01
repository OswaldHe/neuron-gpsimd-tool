"""dma_tiles: bulk data movement with the GPSIMD cores' iDMA (gp_dma_*, Tile::fetch/flush).

  out0 int32 [128, 512]  = a * 3 + 1          rows staged with one 2D descriptor per core (16 partitions)
  out1 int32 [128, 32]   = 7 * v              v = view b[:, :, 2:14:3] of an int16 [128, 8, 16] tile:
                                              fetch() stages the strided span, offset() indexes it
  out2 int8  [128, 37]   = c8 ^ 0x5A          odd row width (37 bytes) - byte-granular transfers
  out3 int32 [128, 512]  = a                  SBUF -> SBUF copy without local staging
  out4 int32 [128, 128]  = t^T                transpose: one 2D descriptor gathers a column (128 x 4 B,
                                              source pitch = partition stride) per output partition
  out5 int32 [128, 512]  = a - 5              async double buffering, 64 KB stack buffer
  out6 int32 [128, 64]   = table[idx]         HBM table: the referenced 4 KB blocks fetched by iDMA
"""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

import gpsimd_nki as gp

OPS = ["op.cpp"]
HBM = ["table"]
P = 128


@nki.jit
def kernel(a, b, c8, t, idx, table, gp_hbm):
    shapes = [((P, 512), nl.int32), ((P, 32), nl.int32), ((P, 37), nl.int8), ((P, 512), nl.int32),
              ((P, 128), nl.int32), ((P, 512), nl.int32), ((P, 64), nl.int32)]
    outs = [nl.ndarray(s, dtype=d, buffer=nl.shared_hbm) for s, d in shapes]
    tin = []
    for x in (a, b, c8, t, idx, gp_hbm):
        tt = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=tt, src=x)
        tin.append(tt)
    ta, tb, tc8, tt_, tidx, thbm = tin
    tout = [nl.ndarray(s, dtype=d, buffer=nl.sbuf) for s, d in shapes]
    gp.call("dma_tiles", ins=[ta, tb[:, :, 2:14:3], tc8, tt_, tidx, thbm], outs=tout)
    for o, x in zip(outs, tout):
        nisa.dma_copy(dst=o, src=x)
    return tuple(outs)


def make_inputs(rng):
    return {"a": rng.integers(-2**31, 2**31, (P, 512), dtype=np.int64).astype(np.int32),
            "b": rng.integers(-2**15, 2**15, (P, 8, 16), dtype=np.int64).astype(np.int16),
            "c8": rng.integers(-128, 128, (P, 37)).astype(np.int8),
            "t": rng.integers(-2**31, 2**31, (P, 128), dtype=np.int64).astype(np.int32),
            "idx": rng.integers(0, 1 << 20, (P, 64)).astype(np.int32),
            "table": rng.integers(-2**31, 2**31, (1024, 1024), dtype=np.int64).astype(np.int32),
            "gp_hbm": np.zeros((1, 16), np.int32)}


def reference(a, b, c8, t, idx, table, gp_hbm):
    w = lambda x: (x.astype(np.int64) & 0xFFFFFFFF).astype(np.uint32).view(np.int32)
    return [w(a.astype(np.int64) * 3 + 1), (7 * b[:, :, 2:14:3].reshape(P, 32).astype(np.int32)),
            (c8 ^ np.int8(0x5A)).astype(np.int8), a.copy(), np.ascontiguousarray(t.T), w(a.astype(np.int64) - 5),
            table.ravel()[idx]]
