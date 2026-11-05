"""vec: sandbox for reverse engineering trn3 GPSIMD vector (FLIX/IVP) instructions; see op.cpp.
Default inputs: a few calibration tests; vecre/vrun.py writes other test sets into the same input."""
import os
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl
import gpsimd_nki as gp

OPS = ["op.cpp"]
ITERS = 1
CFLAGS = ["--log-size", "1024"]


@nki.jit
def kernel(x):
    out = nl.ndarray((128, 64), dtype=nl.int32, buffer=nl.shared_hbm)
    tx = nl.ndarray((128, 64), dtype=nl.int32, buffer=nl.sbuf)
    to = nl.ndarray((128, 64), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tx, src=x)
    gp.call("vprobe", ins=[tx], outs=[to])
    nisa.dma_copy(dst=out, src=to)
    return out


def build_input(tests):
    """tests: list of (hex code, {k: a_k init}, (pattern words16, offset) or None)"""
    x = np.zeros((128, 64), np.uint32)
    for t, tc in enumerate(tests[:127]):
        hx, regs, pat = tc[:3]
        flags = tc[3] if len(tc) > 3 else 0
        b = bytes.fromhex(hx) + b"\0" * 56
        x[t, 0] = len(hx) // 2
        x[t, 1:15] = np.frombuffer(b[:56], "<u4")
        for k in range(3, 16):
            x[t, 16 + k] = regs.get(k, 0x80000000 | (0x1000 + 0x100 * k))
        x[t, 48] = 0xFFFFFFFF
        x[t, 49] = flags
        if pat is not None:
            words, off = pat
            x[t, 32:48] = words
            x[t, 48] = off
    return x.view(np.int32)


DEFAULT = [
    ("323303", {}, None),                       # addi a3, a3, 3? (calibration: base ISA)
    ("4963", {}, None),                         # s32i.n a4, a3, 24 (store)
    ("2f45c05600c95132", {}, None),             # trn1 N0 bundle (ivp_addn_2x32): expected illegal on trn3
    ("2f45c4e108b05032", {}, None),             # trn3 N0 bundle from the built-in kernels
]


def make_inputs(rng):
    return {"x": build_input(DEFAULT)}


def reference(x):
    return [np.zeros((128, 64), np.int32)]
