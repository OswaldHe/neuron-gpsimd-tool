"""idma: copy SBUF rows to local data RAM and back with the core's iDMA (output must equal input)."""
import os
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl
import gpsimd_nki as gp

OPS = ["op.cpp"]
ITERS = 2
CFLAGS = ["--log-size", "2048", "-DMODE=" + os.environ.get("MODE", "4")]


@nki.jit
def kernel(x):
    out = nl.ndarray((128, 1024), dtype=nl.int32, buffer=nl.shared_hbm)
    tx = nl.ndarray((128, 1024), dtype=nl.int32, buffer=nl.sbuf)
    to = nl.ndarray((128, 1024), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tx, src=x)
    gp.call("idma_probe", ins=[tx], outs=[to])
    nisa.dma_copy(dst=out, src=to)
    return out


def make_inputs(rng):
    return {"x": rng.integers(-2**31, 2**31, (128, 1024), dtype=np.int64).astype(np.int32)}


def reference(x):
    return [x]
