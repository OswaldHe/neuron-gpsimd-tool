"""mem2: SBUF access without memw (asm loads/stores) and iDMA register discovery (results in logs)."""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl
import gpsimd_nki as gp

OPS = ["op.cpp"]
ITERS = 1
CFLAGS = ["--log-size", "2048"]


@nki.jit
def kernel(x):
    out = nl.ndarray((128, 1024), dtype=nl.int32, buffer=nl.shared_hbm)
    tx = nl.ndarray((128, 1024), dtype=nl.int32, buffer=nl.sbuf)
    to = nl.ndarray((128, 1024), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tx, src=x)
    gp.call("mem2", ins=[tx], outs=[to])
    nisa.dma_copy(dst=out, src=to)
    return out


def make_inputs(rng):
    return {"x": np.arange(128 * 1024, dtype=np.int32).reshape(128, 1024)}


def reference(x):
    return [x]
