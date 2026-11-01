"""numpy float32 mirror of include/gpsimd/math.hpp (bit-exact reference for GPSIMD math functions).

Each function performs the same sequence of correctly rounded float32 operations as the C++ code.
"""
import numpy as np

f = np.float32


def _ldexp(y, n):
    y = np.asarray(y, np.float32).copy()
    n = np.asarray(n, np.int64).copy()
    with np.errstate(all="ignore"):
        while True:
            m = n > 127
            if not m.any():
                break
            y = np.where(m, y * f(2.0**127), y)
            n = np.where(m, n - 127, n)
        while True:
            m = n < -126
            if not m.any():
                break
            y = np.where(m, y * f(2.0**-126), y)
            n = np.where(m, n + 126, n)
        return (y * np.ldexp(np.float32(1), n.astype(np.int32)).astype(np.float32)).astype(np.float32)


def _to_i32(x):
    with np.errstate(all="ignore"):
        t = np.trunc(x)
        ok = np.isfinite(t) & (np.abs(t) < 2.0**31)
        return np.where(ok, np.where(ok, t, 0).astype(np.int64), -2**31)


def exp(x):
    x0 = np.asarray(x, np.float32)
    # the C++ code returns early outside [-103.97, 88.72] and for NaN: compute on in-range values only
    x = np.where((x0 >= f(-103.972084)) & (x0 <= f(88.72283905206835)), x0, f(0)).astype(np.float32)
    with np.errstate(all="ignore"):
        n = np.rint(x * f(1.44269504088896341))
        r = x - n * f(0.693359375)
        r = r - n * f(-2.12194440e-4)
        z = r * r
        y = f(1.9875691500E-4)
        for k in (1.3981999507E-3, 8.3334519073E-3, 4.1665795894E-2, 1.6666665459E-1, 5.0000001201E-1):
            y = y * r + f(k)
        y = y * z + r + f(1.0)
        res = _ldexp(y, _to_i32(n))
        res = np.where(x0 < f(-103.972084), f(0), res)
        res = np.where(x0 > f(88.72283905206835), f(np.inf), res)
        return np.where(np.isnan(x0), x0, res).astype(np.float32)


def _frexp(x):
    m, e = np.frexp(x.astype(np.float32))
    return m.astype(np.float32), e.astype(np.int64)


def log(x):
    x = np.asarray(x, np.float32)
    with np.errstate(all="ignore"):
        safe = np.where(np.isfinite(x) & (x > 0), x, f(1))
        m, e = _frexp(safe)
        small = m < f(0.707106781186547524)
        e = np.where(small, e - 1, e)
        m = np.where(small, m + m - f(1.0), m - f(1.0))
        z = m * m
        y = f(7.0376836292E-2)
        for k in (-1.1514610310E-1, 1.1676998740E-1, -1.2420140846E-1, 1.4249322787E-1, -1.6668057665E-1,
                  2.0000714765E-1, -2.4999993993E-1, 3.3333331174E-1):
            y = y * m + f(k)
        y = y * m * z
        fe = e.astype(np.float32)
        y = y + fe * f(-2.12194440e-4)
        y = y + f(-0.5) * z
        r = m + y
        r = r + fe * f(0.693359375)
        r = np.where(x == np.inf, x, r)
        r = np.where(x == 0, f(-np.inf), r)
        r = np.where((x < 0) & (x != 0), f(np.nan), r)
        return np.where(np.isnan(x), x, r).astype(np.float32)


def tanh(x):
    x = np.asarray(x, np.float32)
    with np.errstate(all="ignore"):
        ax = np.abs(x)
        s = exp(ax + ax)
        big = f(1.0) - f(2.0) / (s + f(1.0))
        big = np.copysign(big, x)
        z = x * x
        y = f(-5.70498872745E-3)
        for k in (2.06390887954E-2, -5.37397155531E-2, 1.33314422036E-1, -3.33332819422E-1):
            y = y * z + f(k)
        small = y * z * x + x
        small = np.where(ax == 0, x, small)
        r = np.where(ax >= f(0.625), big, small)
        r = np.where(ax > f(44.3614195558365), np.copysign(f(1.0), x), r)
        return np.where(np.isnan(x), x, r).astype(np.float32)


def sigmoid(x):
    x = np.asarray(x, np.float32)
    with np.errstate(all="ignore"):
        return (f(1.0) / (f(1.0) + exp(-x))).astype(np.float32)


def _sincos(x, want_cos):
    x = np.asarray(x, np.float32)
    with np.errstate(all="ignore"):
        ax = np.abs(x)
        j = _to_i32(ax * f(1.27323954473516))
        y = j.astype(np.float32)
        odd = (j & 1) == 1
        j = np.where(odd, j + 1, j)
        y = np.where(odd, y + f(1.0), y)
        j = j & 7
        if want_cos:
            sign = np.zeros(x.shape, bool)
            hi = j > 3
            j = np.where(hi, j - 4, j)
            sign ^= hi
            sign ^= j > 1
        else:
            sign = np.signbit(x)
            hi = j > 3
            sign = sign ^ hi
            j = np.where(hi, j - 4, j)
        z = ((ax - y * f(0.78515625)) - y * f(2.4187564849853515625e-4)) - y * f(3.77489497744594108e-8)
        zz = z * z
        s = f(-1.9515295891E-4)
        s = s * zz + f(8.3321608736E-3)
        s = s * zz + f(-1.6666654611E-1)
        s = s * zz * z + z
        c = f(2.443315711809948E-005)
        c = c * zz + f(-1.388731625493765E-003)
        c = c * zz + f(4.166664568298827E-002)
        c = c * zz * zz - f(0.5) * zz + f(1.0)
        j12 = (j == 1) | (j == 2)
        poly_cos = ~j12 if want_cos else j12
        r = np.where(poly_cos, c, s)
        r = np.where(sign, -r, r)
        return np.where(np.isnan(x) | np.isinf(x), f(np.nan), r).astype(np.float32)


def sin(x):
    return _sincos(x, False)


def cos(x):
    return _sincos(x, True)


def gelu(x):
    x = np.asarray(x, np.float32)
    with np.errstate(all="ignore"):
        inner = f(0.7978845608028654) * (x + f(0.044715) * x * x * x)
        return (f(0.5) * x * (f(1.0) + tanh(inner))).astype(np.float32)


def rsqrt(x):
    x = np.asarray(x, np.float32)
    with np.errstate(all="ignore"):
        return (f(1.0) / np.sqrt(x)).astype(np.float32)
