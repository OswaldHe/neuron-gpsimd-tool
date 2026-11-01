"""bench: cycle costs of memory accesses and soft-float on the GPSIMD (results in the core logs)."""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl
import gpsimd_nki as gp

OPS = ["op.cpp"]
HBM = ["table"]
ITERS = 1
CFLAGS = ["--log-size", "2048"]


@nki.jit
def kernel(x, table, gp_hbm):
    out = nl.ndarray((128, 1024), dtype=nl.int32, buffer=nl.shared_hbm)
    tx = nl.ndarray((128, 1024), dtype=nl.int32, buffer=nl.sbuf)
    tp = nl.ndarray((1, 16), dtype=nl.int32, buffer=nl.sbuf)
    to = nl.ndarray((128, 1024), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tx, src=x)
    nisa.dma_copy(dst=tp, src=gp_hbm)
    gp.call("bench", ins=[tx, tp], outs=[to])
    nisa.dma_copy(dst=out, src=to)
    return out


def make_inputs(rng):
    return {"x": np.arange(128 * 1024, dtype=np.int32).reshape(128, 1024),
            "table": np.arange(1 << 20, dtype=np.int32).reshape(1024, 1024), "gp_hbm": np.zeros((1, 16), np.int32)}


def reference(x, table, gp_hbm):
    return [x]
