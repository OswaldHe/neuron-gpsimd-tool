"""fp32_math: IEEE binary32 arithmetic on the GPSIMD cores (soft-float, round to nearest even).

Inputs: float32 tiles a, b (random bit patterns, all exponents, +-0, +-inf, quiet/signaling NaNs,
subnormals, halves for rounding), c (finite, for fma), i (int32). Output: uint32 bit patterns, one
column block per operation (order of OPSPEC); reference = numpy float32 (x86 SSE semantics incl. NaN
propagation), fma = exact rational arithmetic rounded once.
"""
from fractions import Fraction

import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

import gpsimd_nki as gp

OPS = ["op.cpp"]
P, F = 128, 16
OPSPEC = ["add", "sub", "mul", "div", "sqrt", "fma", "min", "max", "floor", "ceil", "trunc", "rint",
          "round", "to_i32", "from_i32", "cmp", "neg_abs", "from_u32"]
NB = len(OPSPEC)


@nki.jit
def kernel(a, b, c, i):
    out = nl.ndarray((P, NB * F), dtype=nl.uint32, buffer=nl.shared_hbm)
    ts = []
    for x in (a, b, c, i):
        t = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=x)
        ts.append(t)
    to = nl.ndarray((P, NB * F), dtype=nl.uint32, buffer=nl.sbuf)
    gp.call("fp32_math", ins=ts, outs=[to])
    nisa.dma_copy(dst=out, src=to)
    return out


SPECIAL = np.array([0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7FA00001, 0xFF812345,
                    0x00000001, 0x80000001, 0x007FFFFF, 0x00800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x3F800000, 0xBF800000,
                    0x3F000000, 0x3FC00000, 0x40200000, 0xBFC00000, 0x4B000001, 0x4EFFFFFF, 0x4F000000, 0xCF000000,
                    0xCF000001, 0x3EFFFFFF, 0x34000000, 0x0C000000], np.uint32)


def make_inputs(rng):
    def bits(finite=False):
        x = rng.integers(0, 2**32, (P, F), dtype=np.uint64).astype(np.uint32)
        moderate = (rng.normal(0, 30, (P, F)).astype(np.float32)).view(np.uint32)
        x = np.where(rng.random((P, F)) < 0.4, moderate, x)
        halves = ((rng.integers(-40, 40, (P, F)) + 0.5).astype(np.float32)).view(np.uint32)
        x = np.where(rng.random((P, F)) < 0.1, halves, x)
        x = np.where(rng.random((P, F)) < 0.25, SPECIAL[rng.integers(0, len(SPECIAL), (P, F))], x)
        if finite:
            x = np.where((x & 0x7F800000) == 0x7F800000, x & 0xBFFFFFFF, x)
        return x
    return {"a": bits().view(np.float32), "b": bits().view(np.float32), "c": bits(True).view(np.float32),
            "i": rng.integers(-2**31, 2**31, (P, F), dtype=np.int64).astype(np.int32)}


def f32_round(q):
    """exact rational -> float32 bits, round to nearest even (with subnormals and overflow)."""
    if q == 0:
        return 0
    sign = 0x80000000 if q < 0 else 0
    q = abs(q)
    e = q.numerator.bit_length() - q.denominator.bit_length()
    if Fraction(2) ** e > q:
        e -= 1
    e = max(e, -126)
    m = q / Fraction(2) ** (e - 23)                 # significand scaled to 24 bits
    n = m.numerator // m.denominator
    rem = m - n
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and n & 1):
        n += 1
    if n == 1 << 24:
        n >>= 1
        e += 1
    if e > 127:
        return sign | 0x7F800000
    if n < 1 << 23:
        return sign | n                              # subnormal
    return sign | ((e + 127) << 23) | (n & 0x7FFFFF)


def fma_ref(a, b, c):
    out = np.zeros(a.shape, np.uint32)
    for idx in np.ndindex(a.shape):
        x, y, z = (float(v[idx]) for v in (a, b, c))
        if any(np.isinf(v) or np.isnan(v) for v in (x, y, z)):
            raise ValueError("fma reference expects finite c")
        if np.isinf(x * 1.0) or np.isnan(x):
            pass
        exact = Fraction(x) * Fraction(y) + Fraction(z)
        r = f32_round(exact)
        if exact == 0:
            prod_neg = (np.signbit(a[idx]) != np.signbit(b[idx]))
            r = 0x80000000 if (prod_neg and np.signbit(c[idx]) and Fraction(x) * Fraction(y) == 0) else 0
        out[idx] = r
    return out


def reference(a, b, c, i):
    u = lambda x: np.asarray(x, np.float32).view(np.uint32)
    blocks = {}
    with np.errstate(all="ignore"):
        blocks["add"], blocks["sub"], blocks["mul"], blocks["div"] = u(a + b), u(a - b), u(a * b), u(a / b)
        blocks["sqrt"] = u(np.sqrt(a))
        fin = lambda x: np.where(np.isfinite(x), x, np.float32(1.5))
        blocks["fma"] = fma_ref(fin(a), fin(b), c)
        blocks["min"], blocks["max"] = u(np.where(a < b, a, b)), u(np.where(b < a, a, b))
        blocks["floor"], blocks["ceil"], blocks["trunc"], blocks["rint"] = u(np.floor(a)), u(np.ceil(a)), u(np.trunc(a)), u(np.rint(a))
        t = np.trunc(a)
        blocks["round"] = u(np.where(np.abs(a - t) >= np.float32(0.5), t + np.copysign(np.float32(1), a), t).astype(np.float32))
        blocks["round"] = np.where(np.isnan(a), u(np.trunc(a)), blocks["round"])
        valid = np.isfinite(a) & (np.abs(t) < 2.0**31) | (t == -2.0**31)
        blocks["to_i32"] = np.where(valid, np.where(valid, t, 0).astype(np.int64), -2**31).astype(np.int64).astype(np.uint32)
        blocks["from_i32"] = u(i.astype(np.float32))
        blocks["cmp"] = ((a < b).astype(np.uint32) | (a <= b).astype(np.uint32) << 1 | (a == b).astype(np.uint32) << 2 |
                         (np.isnan(a) | np.isnan(b)).astype(np.uint32) << 3)
        blocks["neg_abs"] = u(-a) ^ u(np.abs(b))
        blocks["from_u32"] = u(i.view(np.uint32).astype(np.float32))
    return [np.concatenate([blocks[k].astype(np.uint32) for k in OPSPEC], axis=1)]
