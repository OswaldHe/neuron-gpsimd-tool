import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl
import gpsimd_nki as gp

OPS = ["op.cpp"]
ITERS = 1


@nki.jit
def kernel(x):
    out = nl.ndarray((1, 16), dtype=nl.int32, buffer=nl.shared_hbm)
    tx = nl.ndarray((1, 16), dtype=nl.int32, buffer=nl.sbuf)
    to = nl.ndarray((1, 16), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tx, src=x)
    gp.call("probe_excl", ins=[tx], outs=[to])
    nisa.dma_copy(dst=out, src=to)
    return out


def make_inputs(rng):
    return {"x": np.arange(16, dtype=np.int32).reshape(1, 16)}


def reference(x):
    return [np.zeros((1, 16), np.int32)]
