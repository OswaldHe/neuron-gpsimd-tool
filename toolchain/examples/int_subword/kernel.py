"""int_subword: 8- and 16-bit integer formats, byte/halfword loads and stores through the SBUF window.

Inputs int8, uint8, int16, uint16 tiles. Outputs:
  0 int8   saturating a8 + b8                      (element-interleaved over cores: core k does i % 8 == k,
  1 uint8  rounding average (u8 + v8 + 1) >> 1      so all 8 cores store bytes into the same 32-bit words)
  2 int16  Q15 multiply: (a16 * b16 + 2^14) >> 15, saturated  (interleaved over cores)
  3 uint16 byte swap of u16 ^ (u16 >> 3)            (partition split)
  4 int32  widening sum a8 + u8 + a16 + u16          (partition split, mixed-width loads)
  5 int16  a8 sign-extended * u8 zero-extended       (element-interleaved 16-bit stores)
"""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

import gpsimd_nki as gp

OPS = ["op.cpp"]
P, F = 128, 96


@nki.jit
def kernel(a8, b8, u8, v8, a16, b16, u16):
    outs_hbm = [nl.ndarray((P, F), dtype=dt, buffer=nl.shared_hbm)
                for dt in (nl.int8, nl.uint8, nl.int16, nl.uint16, nl.int32, nl.int16)]
    tiles_in = []
    for x in (a8, b8, u8, v8, a16, b16, u16):
        t = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=x)
        tiles_in.append(t)
    tiles_out = [nl.ndarray((P, F), dtype=o.dtype, buffer=nl.sbuf) for o in outs_hbm]
    gp.call("int_subword", ins=tiles_in, outs=tiles_out)
    for o, t in zip(outs_hbm, tiles_out):
        nisa.dma_copy(dst=o, src=t)
    return tuple(outs_hbm)


def make_inputs(rng):
    def r(dt):
        info = np.iinfo(dt)
        x = rng.integers(info.min, info.max + 1, (P, F), dtype=np.int64)
        edge = np.array([info.min, info.max, 0, -1 if info.min < 0 else 1, info.min + 1, info.max - 1])
        x = np.where(rng.random((P, F)) < 0.2, edge[rng.integers(0, len(edge), (P, F))], x)
        return x.astype(dt)
    return {"a8": r(np.int8), "b8": r(np.int8), "u8": r(np.uint8), "v8": r(np.uint8),
            "a16": r(np.int16), "b16": r(np.int16), "u16": r(np.uint16)}


def reference(a8, b8, u8, v8, a16, b16, u16):
    i = lambda x: x.astype(np.int64)
    o0 = np.clip(i(a8) + i(b8), -128, 127).astype(np.int8)
    o1 = ((i(u8) + i(v8) + 1) >> 1).astype(np.uint8)
    o2 = np.clip((i(a16) * i(b16) + (1 << 14)) >> 15, -32768, 32767).astype(np.int16)
    x = i(u16) ^ (i(u16) >> 3)
    o3 = (((x & 0xFF) << 8) | (x >> 8)).astype(np.uint16)
    o4 = (i(a8) + i(u8) + i(a16) + i(u16)).astype(np.int32)
    o5 = (i(a8) * i(u8)).astype(np.int16)
    return [o0, o1, o2, o3, o4, o5]
