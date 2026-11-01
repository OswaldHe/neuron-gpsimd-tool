"""int64_bits: 64-bit integer arithmetic and bit utilities (compiler-runtime helpers in runtime/builtins.c,
all base-ISA code) plus mem* functions, on 64-bit operand pairs stored as uint32 [lo, hi] words.

Result block k (2 words per operand pair) = OPS64[k]; reference: Python integers.
"""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

import gpsimd_nki as gp

OPS = ["op.cpp"]
P, N = 128, 8           # N 64-bit operands per partition
M64 = (1 << 64) - 1


def s64(x):
    return x - (1 << 64) if x >> 63 else x


def tdiv(a, b):
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def nz(b):
    return b if b else 1


OPS64 = [
    ("add", lambda a, b: a + b), ("sub", lambda a, b: a - b), ("mul", lambda a, b: a * b),
    ("udiv", lambda a, b: a // nz(b)), ("umod", lambda a, b: a % nz(b)),
    ("sdiv", lambda a, b: a if (s64(a) == -2**63 and s64(nz(b)) == -1) else tdiv(s64(a), s64(nz(b)))),
    ("smod", lambda a, b: 0 if (s64(a) == -2**63 and s64(nz(b)) == -1) else s64(a) - tdiv(s64(a), s64(nz(b))) * s64(nz(b))),
    ("shl", lambda a, b: a << (b & 63)), ("lshr", lambda a, b: a >> (b & 63)), ("ashr", lambda a, b: s64(a) >> (b & 63)),
    ("cmp", lambda a, b: (s64(a) < s64(b)) | (a < b) << 1 | (a == b) << 2),
    ("popcount", lambda a, b: bin(a).count("1") | bin(a & 0xFFFFFFFF).count("1") << 8),
    ("clz", lambda a, b: (64 - a.bit_length() if a else 64) | (32 - (a & 0xFFFFFFFF).bit_length()) << 8),
    ("ctz", lambda a, b: ((a & -a).bit_length() - 1 if a else 64) | (((a & 0xFFFFFFFF) & -(a & 0xFFFFFFFF)).bit_length() - 1 if a & 0xFFFFFFFF else 32) << 8),
    ("ffs", lambda a, b: (a & -a).bit_length()),
    ("bswap", lambda a, b: int.from_bytes(a.to_bytes(8, "little"), "big")),
    ("parity", lambda a, b: bin(a).count("1") & 1),
    ("mulhi", lambda a, b: (a * b) >> 64),
    ("mem", None),
]


@nki.jit
def kernel(a, b):
    out = nl.ndarray((P, len(OPS64) * 2 * N), dtype=nl.uint32, buffer=nl.shared_hbm)
    ta = nl.ndarray((P, 2 * N), dtype=nl.uint32, buffer=nl.sbuf)
    tb = nl.ndarray((P, 2 * N), dtype=nl.uint32, buffer=nl.sbuf)
    to = nl.ndarray((P, len(OPS64) * 2 * N), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ta, src=a)
    nisa.dma_copy(dst=tb, src=b)
    gp.call("int64_bits", ins=[ta, tb], outs=[to])
    nisa.dma_copy(dst=out, src=to)
    return out


EDGE = [0, 1, 2, 3, 63, 64, 2**31, 2**32 - 1, 2**32, 2**63 - 1, 2**63, 2**64 - 1, 2**64 - 2, 0x0123456789ABCDEF]


def make_inputs(rng):
    def gen():
        x = rng.integers(0, 2**63, (P, N), dtype=np.uint64) * np.uint64(2) + rng.integers(0, 2, (P, N), dtype=np.uint64)
        small = rng.integers(0, 70, (P, N)).astype(np.uint64)
        x = np.where(rng.random((P, N)) < 0.2, small, x)
        shift = rng.integers(0, 64, (P, N)).astype(np.uint64)
        x = np.where(rng.random((P, N)) < 0.2, x >> shift, x)
        x = np.where(rng.random((P, N)) < 0.15, np.array(EDGE, np.uint64)[rng.integers(0, len(EDGE), (P, N))], x)
        return x
    a, b = gen(), gen()
    b[0, 0], a[0, 1], b[0, 1] = 0, 2**63, 2**64 - 1        # division by 0 -> 1, INT64_MIN / -1
    return {"a": a.view(np.uint32).reshape(P, 2 * N), "b": b.view(np.uint32).reshape(P, 2 * N)}


def mem_ref(a, b):
    buf = bytearray(a.to_bytes(8, "little") + b.to_bytes(8, "little"))
    buf[3:11] = buf[0:8]                         # memmove (overlapping)
    buf[12:16] = bytes([buf[1]] * 4)             # memset
    cmp = (bytes(buf[0:8]) > bytes(buf[8:16])) - (bytes(buf[0:8]) < bytes(buf[8:16]))
    return (int.from_bytes(buf[0:8], "little") ^ int.from_bytes(buf[8:16], "little")) + (cmp & 0xF)


def reference(a, b):
    A, B = a.view(np.uint64), b.view(np.uint64)
    out = np.zeros((P, len(OPS64) * N), np.uint64)
    for k, (name, fn) in enumerate(OPS64):
        for p in range(P):
            for i in range(N):
                x, y = int(A[p, i]), int(B[p, i])
                out[p, k * N + i] = (mem_ref(x, y) if fn is None else int(fn(x, y))) & M64
    return [out.view(np.uint32).reshape(P, len(OPS64) * 2 * N)]
