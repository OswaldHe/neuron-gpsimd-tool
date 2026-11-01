"""views: calls on tile views - strided, multi-dimensional, partition offsets (aligned and unaligned),
and an in-place call (the same tile is input and output).

  call 0 "views_scale": in  T[:, :, 2:14:3]   (int16 [128, 8, 16] tile, every 3rd element of 2..13)
                        out U[32:96, :]        (int32 [128, 64] tile, partitions 32..95)
                        U[32+p, j*4+k] = 10 * T[p, j, 2 + 3k] for p < 64
  call 1 "views_inc":   in/out V[5:45, 8:40]   (int32 [128, 64] tile, unaligned partition range 5..44;
                        incremented in place: +7)
"""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

import gpsimd_nki as gp

OPS = ["op.cpp"]


@nki.jit
def kernel(t, u, v):
    ou = nl.ndarray((128, 64), dtype=nl.int32, buffer=nl.shared_hbm)
    ov = nl.ndarray((128, 64), dtype=nl.int32, buffer=nl.shared_hbm)
    tt = nl.ndarray((128, 8, 16), dtype=nl.int16, buffer=nl.sbuf)
    tu = nl.ndarray((128, 64), dtype=nl.int32, buffer=nl.sbuf)
    tv = nl.ndarray((128, 64), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tt, src=t)
    nisa.dma_copy(dst=tu, src=u)
    nisa.dma_copy(dst=tv, src=v)
    gp.call("views_scale", ins=[tt[0:64, :, 2:14:3]], outs=[tu[32:96, :]])
    gp.call("views_inc", ins=[tv[5:45, 8:40]], outs=[tv[5:45, 8:40]])
    nisa.dma_copy(dst=ou, src=tu)
    nisa.dma_copy(dst=ov, src=tv)
    return ou, ov


def make_inputs(rng):
    return {"t": rng.integers(-3000, 3000, (128, 8, 16)).astype(np.int16),
            "u": rng.integers(-100, 100, (128, 64)).astype(np.int32),
            "v": rng.integers(-100, 100, (128, 64)).astype(np.int32)}


def reference(t, u, v):
    ou = u.copy()
    ou[32:96, :] = 10 * t[0:64, :, 2:14:3].reshape(64, 32).repeat(1, axis=1).astype(np.int32).reshape(64, 32).repeat(2, axis=1)[:, ::2].repeat(2, axis=1)[:, :64] if False else ou[32:96, :]
    ou[32:96, :] = np.concatenate([10 * t[0:64, :, 2:14:3].reshape(64, 32).astype(np.int32)] * 2, axis=1)
    ov = v.copy()
    ov[5:45, 8:40] += 7
    return [ou, ov]
