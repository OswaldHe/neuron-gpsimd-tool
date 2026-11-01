"""half_formats: bfloat16 and float16 tiles, conversions and mixed-precision arithmetic.

Inputs x (bf16), y (fp16), z (fp32). Outputs:
  0 bf16 = bf16(f32(x) * f32(y) + z)        two float32 roundings, then round to nearest even to bf16
  1 fp16 = f16(f32(x) - f32(y))             overflow to inf, fp16 subnormals
  2 fp32 = f32(x) + f32(y)                  exact widening, one rounding
  3 bf16 = bf16(z)                          RNE narrowing (NaN -> canonical quiet NaN, sign kept)
  4 fp16 = f16(z)                           RNE narrowing with subnormals / overflow / NaN payloads
  5 int8 = clamp(rint(f32(x) * 16), -128, 127)   symmetric quantization (NaN -> 0)
Reference: numpy float32 + ml_dtypes.bfloat16 / numpy float16 conversions (bit-exact).
"""
import ml_dtypes
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

import gpsimd_nki as gp

OPS = ["op.cpp"]
P, F = 128, 64
BF16 = ml_dtypes.bfloat16


@nki.jit
def kernel(x, y, z):
    shapes = [(nl.bfloat16,), (nl.float16,), (nl.float32,), (nl.bfloat16,), (nl.float16,), (nl.int8,)]
    outs = [nl.ndarray((P, F), dtype=s[0], buffer=nl.shared_hbm) for s in shapes]
    tin = []
    for v in (x, y, z):
        t = nl.ndarray(v.shape, dtype=v.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=v)
        tin.append(t)
    tout = [nl.ndarray((P, F), dtype=o.dtype, buffer=nl.sbuf) for o in outs]
    gp.call("half_formats", ins=tin, outs=tout)
    for o, t in zip(outs, tout):
        nisa.dma_copy(dst=o, src=t)
    return tuple(outs)


def make_inputs(rng):
    def f32_bits(n_special=0.15):
        v = (rng.normal(0, 1, (P, F)) * np.exp2(rng.integers(-30, 30, (P, F)))).astype(np.float32)
        sp = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 65504.0, 65520.0, 6.1e-5, 6e-8, 3e-8, 1e-40,
                       3.4e38, -3.4e38, 1.0, 1.0 + 2**-8, 1.0 + 2**-9, 1.0 + 3 * 2**-9], np.float32)
        v = np.where(rng.random((P, F)) < n_special, sp[rng.integers(0, len(sp), (P, F))], v)
        bits = v.view(np.uint32)
        nan_payload = rng.integers(0x7F800001, 0x80000000, (P, F), dtype=np.int64).astype(np.uint32)
        bits = np.where(rng.random((P, F)) < 0.02, nan_payload | (rng.integers(0, 2, (P, F)) << 31).astype(np.uint32), bits)
        return bits.view(np.float32)
    with np.errstate(all="ignore"):
        x = (rng.normal(0, 4, (P, F))).astype(np.float32)
        x = np.where(rng.random((P, F)) < 0.1, f32_bits(0.5), x).astype(BF16)
        y = (rng.normal(0, 4, (P, F))).astype(np.float32)
        y = np.where(rng.random((P, F)) < 0.1, f32_bits(0.5), y).astype(np.float16)
    return {"x": x, "y": y, "z": f32_bits()}


def reference(x, y, z):
    with np.errstate(all="ignore"):
        fx, fy = x.astype(np.float32), y.astype(np.float32)
        o0 = (fx * fy + z).astype(BF16)
        o1 = (fx - fy).astype(np.float16)
        o2 = (fx + fy).astype(np.float32)
        o3 = z.astype(BF16)
        o4 = z.astype(np.float16)
        q = np.rint(fx * np.float32(16))
        o5 = np.where(np.isnan(q), 0, np.clip(q, -128, 127)).astype(np.int8)
    return [o0, o1, o2, o3, o4, o5]
