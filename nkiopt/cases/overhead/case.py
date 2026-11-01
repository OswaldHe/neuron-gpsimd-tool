"""overhead: fixed cost of one GPSIMD custom call (tiny copy) vs one NKI tensor_copy."""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl
import gpsimd_nki as gp

OPS = ["op.cpp"]
P, F = 128, 4


@nki.jit
def baseline_kernel(x):
    out = nl.ndarray((P, F), dtype=nl.int32, buffer=nl.shared_hbm)
    t = nl.ndarray((P, F), dtype=nl.int32, buffer=nl.sbuf)
    u = nl.ndarray((P, F), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    nisa.tensor_copy(dst=u, src=t, engine=nisa.gpsimd_engine)
    nisa.dma_copy(dst=out, src=u)
    return out


@nki.jit
def gpsimd_kernel(x):
    out = nl.ndarray((P, F), dtype=nl.int32, buffer=nl.shared_hbm)
    t = nl.ndarray((P, F), dtype=nl.int32, buffer=nl.sbuf)
    u = nl.ndarray((P, F), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    gp.call("copy_small", ins=[t], outs=[u])
    nisa.dma_copy(dst=out, src=u)
    return out


def make_inputs(rng):
    return {"x": rng.integers(-2**31, 2**31, (P, F), dtype=np.int64).astype(np.int32)}


def reference(x):
    return [x.copy()]
