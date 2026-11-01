"""isa_alu: exhaustive check of the trn3 GPSIMD core's base-ISA integer instructions.

Every instruction (ALU, multiply/divide, min/max, shifts through SAR incl. funnel shifts, sign
extension/clamping/bit-field ops, conditional moves, DEPBITS, every branch condition) runs through
inline asm (op.cpp) on 2048 operand triples (random values mixed with edge cases); OPS below is the
numpy model of each instruction and the reference for the bit-exact comparison.
"""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

import gpsimd_nki as gp

OPS = ["op.cpp"]
P, N = 128, 16
M32 = 0xFFFFFFFF


def u(x):
    return np.asarray(x, dtype=np.int64) & M32


def s(x):
    x = u(x)
    return np.where(x >= 2**31, x - 2**32, x)


def sext(x, bit):
    x = u(x) & ((1 << (bit + 1)) - 1)
    return np.where(x >> bit, x - (1 << (bit + 1)), x)


def nsau(x):
    x = u(x)
    r = np.full(x.shape, 32, np.int64)
    for k in range(32):
        r = np.where((x >> (31 - k)) & 1, np.minimum(r, k), r)
    return r


def nsa(x):
    x = s(x)
    y = np.where(x < 0, ~x, x)
    return np.where(y == 0, 31, nsau(y) - 1)


def trunc_div(a, b):
    q = np.abs(a) // np.abs(b)
    return np.where((a < 0) ^ (b < 0), -q, q)


def nz(b):
    return np.where(u(b) == 0, 1, u(b))


def funnel(a, c, sar):
    return ((u(a) << 32 | u(c)) >> sar) & M32


def depbits(r, a, sh, w):
    m = ((1 << w) - 1) << sh
    return (u(r) & ~m | (u(a) << sh) & m) & M32


def clamps(a, bit):
    return np.clip(s(a), -(1 << bit), (1 << bit) - 1)


def mul16s(a, b):
    return sext(a, 15) * sext(b, 15)


# (name, f(a, b, c)) in the order of OPS(X) in op.cpp
TABLE = [
    ("add", lambda a, b, c: u(a) + u(b)), ("sub", lambda a, b, c: u(a) - u(b)),
    ("addx2", lambda a, b, c: 2 * u(a) + u(b)), ("addx4", lambda a, b, c: 4 * u(a) + u(b)),
    ("addx8", lambda a, b, c: 8 * u(a) + u(b)), ("subx2", lambda a, b, c: 2 * u(a) - u(b)),
    ("subx4", lambda a, b, c: 4 * u(a) - u(b)), ("subx8", lambda a, b, c: 8 * u(a) - u(b)),
    ("and", lambda a, b, c: u(a) & u(b)), ("or", lambda a, b, c: u(a) | u(b)), ("xor", lambda a, b, c: u(a) ^ u(b)),
    ("neg", lambda a, b, c: -u(a)), ("abs", lambda a, b, c: np.abs(s(a))),
    ("mull", lambda a, b, c: u(a) * u(b)),
    ("muluh", lambda a, b, c: np.array([(int(x) * int(y)) >> 32 for x, y in zip(u(a).ravel(), u(b).ravel())]).reshape(a.shape)),
    ("mulsh", lambda a, b, c: np.array([(int(x) * int(y)) >> 32 for x, y in zip(s(a).ravel(), s(b).ravel())]).reshape(a.shape)),
    ("mul16u", lambda a, b, c: (u(a) & 0xFFFF) * (u(b) & 0xFFFF)), ("mul16s", lambda a, b, c: mul16s(a, b)),
    ("quou", lambda a, b, c: u(a) // nz(b)), ("quos", lambda a, b, c: trunc_div(s(a), s(nz(b)))),
    ("remu", lambda a, b, c: u(a) % nz(b)),
    ("rems", lambda a, b, c: s(a) - trunc_div(s(a), s(nz(b))) * s(nz(b))),
    ("min", lambda a, b, c: np.minimum(s(a), s(b))), ("max", lambda a, b, c: np.maximum(s(a), s(b))),
    ("minu", lambda a, b, c: np.minimum(u(a), u(b))), ("maxu", lambda a, b, c: np.maximum(u(a), u(b))),
    ("salt", lambda a, b, c: (s(a) < s(b)).astype(np.int64)), ("saltu", lambda a, b, c: (u(a) < u(b)).astype(np.int64)),
    ("nsa", lambda a, b, c: nsa(a)), ("nsau", lambda a, b, c: nsau(a)),
    ("sext7", lambda a, b, c: sext(a, 7)), ("sext15", lambda a, b, c: sext(a, 15)), ("sext22", lambda a, b, c: sext(a, 22)),
    ("clamps7", lambda a, b, c: clamps(a, 7)), ("clamps15", lambda a, b, c: clamps(a, 15)),
    ("extui5_12", lambda a, b, c: (u(a) >> 5) & 0xFFF), ("extui0_1", lambda a, b, c: u(a) & 1),
    ("extui28_4", lambda a, b, c: u(a) >> 28), ("extui16_16", lambda a, b, c: u(a) >> 16),
    ("slli1", lambda a, b, c: u(a) << 1), ("slli17", lambda a, b, c: u(a) << 17), ("slli31", lambda a, b, c: u(a) << 31),
    ("srli1", lambda a, b, c: u(a) >> 1), ("srli15", lambda a, b, c: u(a) >> 15),
    ("srai3", lambda a, b, c: s(a) >> 3), ("srai31", lambda a, b, c: s(a) >> 31),
    ("addi-128", lambda a, b, c: u(a) - 128), ("addi127", lambda a, b, c: u(a) + 127),
    ("addmi-32768", lambda a, b, c: u(a) - 32768), ("addmi32512", lambda a, b, c: u(a) + 32512),
    ("ssl+sll", lambda a, b, c: u(a) << (u(b) & 31)), ("ssr+srl", lambda a, b, c: u(a) >> (u(b) & 31)),
    ("ssr+sra", lambda a, b, c: s(a) >> (u(b) & 31)), ("ssr+src", lambda a, b, c: funnel(a, c, u(b) & 31)),
    ("ssa8l+src", lambda a, b, c: funnel(a, c, (u(b) & 3) * 8)),
    ("ssa8b+src", lambda a, b, c: funnel(a, c, 32 - (u(b) & 3) * 8)),
    ("ssai13+src", lambda a, b, c: funnel(a, c, 13)), ("ssai0+src", lambda a, b, c: funnel(a, c, 0)),
    ("moveqz", lambda a, b, c: np.where(u(c) == 0, u(b), u(a))), ("movnez", lambda a, b, c: np.where(u(c) != 0, u(b), u(a))),
    ("movltz", lambda a, b, c: np.where(s(c) < 0, u(b), u(a))), ("movgez", lambda a, b, c: np.where(s(c) >= 0, u(b), u(a))),
    ("depbits4_8", lambda a, b, c: depbits(c, a, 4, 8)), ("depbits0_1", lambda a, b, c: depbits(c, a, 0, 1)),
    ("depbits20_12", lambda a, b, c: depbits(c, a, 20, 12)), ("depbits0_16", lambda a, b, c: depbits(c, a, 0, 16)),
    ("beq", lambda a, b, c: u(a) == u(b)), ("bne", lambda a, b, c: u(a) != u(b)), ("blt", lambda a, b, c: s(a) < s(b)),
    ("bge", lambda a, b, c: s(a) >= s(b)), ("bltu", lambda a, b, c: u(a) < u(b)), ("bgeu", lambda a, b, c: u(a) >= u(b)),
    ("bany", lambda a, b, c: (u(a) & u(b)) != 0), ("bnone", lambda a, b, c: (u(a) & u(b)) == 0),
    ("ball", lambda a, b, c: (u(a) & u(b)) == u(b)), ("bnall", lambda a, b, c: (u(a) & u(b)) != u(b)),
    ("bbc", lambda a, b, c: ((u(a) >> (u(b) & 31)) & 1) == 0), ("bbs", lambda a, b, c: ((u(a) >> (u(b) & 31)) & 1) == 1),
    ("beqz", lambda a, b, c: u(a) == 0), ("bnez", lambda a, b, c: u(a) != 0), ("bltz", lambda a, b, c: s(a) < 0),
    ("bgez", lambda a, b, c: s(a) >= 0), ("beqi4", lambda a, b, c: s(a) == 4), ("bnei-1", lambda a, b, c: s(a) != -1),
    ("blti8", lambda a, b, c: s(a) < 8), ("bgei256", lambda a, b, c: s(a) >= 256), ("bltui16", lambda a, b, c: u(a) < 16),
    ("bgeui32768", lambda a, b, c: u(a) >= 32768), ("bbci31", lambda a, b, c: (u(a) >> 31) == 0),
    ("bbsi0", lambda a, b, c: (u(a) & 1) == 1),
]
NOPS = len(TABLE)


@nki.jit
def kernel(a, b, c):
    out = nl.ndarray((P, NOPS * N), dtype=nl.uint32, buffer=nl.shared_hbm)
    ta = nl.ndarray((P, N), dtype=nl.uint32, buffer=nl.sbuf)
    tb = nl.ndarray((P, N), dtype=nl.uint32, buffer=nl.sbuf)
    tc = nl.ndarray((P, N), dtype=nl.uint32, buffer=nl.sbuf)
    to = nl.ndarray((P, NOPS * N), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ta, src=a)
    nisa.dma_copy(dst=tb, src=b)
    nisa.dma_copy(dst=tc, src=c)
    gp.call("isa_alu", ins=[ta, tb, tc], outs=[to])
    nisa.dma_copy(dst=out, src=to)
    return out


EDGE = [0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 127, 128, 255, 256, 0x7FFF, 0x8000, 0xFFFF, 0x10000, 32767, 32768,
        0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFF80, 0xFFFF8000, 0x55555555, 0xAAAAAAAA]


def make_inputs(rng):
    def gen():
        x = rng.integers(0, 2**32, (P, N), dtype=np.uint64)
        small = rng.integers(-300, 300, (P, N))
        x = np.where(rng.random((P, N)) < 0.25, small & M32, x)
        mask = rng.random((P, N)) < 0.3
        x = np.where(mask, np.array(EDGE, dtype=np.uint64)[rng.integers(0, len(EDGE), (P, N))], x)
        return x.astype(np.uint32)
    a, b = gen(), gen()
    b = np.where(rng.random((P, N)) < 0.1, a, b)            # equal operands for beq/ball/...
    c = gen()
    c = np.where(rng.random((P, N)) < 0.3, 0, c).astype(np.uint32)   # zeros for moveqz/movnez
    # deterministic edge pairs in partition 0: overflow / division by zero / equal / all ones
    pairs = [(0x80000000, 0xFFFFFFFF), (0x80000000, 0), (0x7FFFFFFF, 0x7FFFFFFF), (0xFFFFFFFF, 0xFFFFFFFF),
             (0, 0), (1, 0xFFFFFFFF), (0x80000000, 1), (0x80000000, 0x80000000), (5, 0xFFFFFFFB),
             (0xFFFFFFFB, 5), (0x7FFFFFFF, 0x80000000), (0xFFFF, 0xFFFF), (0x8000, 0x8000), (4, 4), (0, 0xFFFFFFFF),
             (0x12345678, 0x87654321)]
    for i, (x, y) in enumerate(pairs[:N]):
        a[0, i], b[0, i] = x, y
    return {"a": a, "b": b, "c": c}


def reference(a, b, c):
    cols = []
    for name, f in TABLE:
        r = np.asarray(f(a.astype(np.int64), b.astype(np.int64), c.astype(np.int64)), dtype=np.int64)
        cols.append((r & M32).astype(np.uint32))
    return [np.concatenate(cols, axis=1)]
