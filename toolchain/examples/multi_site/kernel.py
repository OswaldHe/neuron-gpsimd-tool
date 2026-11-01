"""multi_site: several GPSIMD call sites in one kernel, interleaved with other engines.

  t1 = a                                   (DMA)
  t2 = triple(t1)            GPSIMD site 0  t2 = 3 * t1
  t3 = t2 + 1                Vector engine  (tensor_scalar on DVE)
  t4 = mix(t3, t1)           GPSIMD site 1  t4 = t3 ^ t1
  t5 = t4 * 2                GpSimd engine  (a native Pool instruction between custom calls)
  t6 = triple(t5)            GPSIMD site 2  same operator, different call site
  outputs t6 and t4          (DMA)
Values stay below 2^20 so the native float-based engine ops are exact.
"""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

import gpsimd_nki as gp

OPS = ["op.cpp"]
P, F = 128, 64


@nki.jit
def kernel(a):
    o6 = nl.ndarray((P, F), dtype=nl.int32, buffer=nl.shared_hbm)
    o4 = nl.ndarray((P, F), dtype=nl.int32, buffer=nl.shared_hbm)
    t = [nl.ndarray((P, F), dtype=nl.int32, buffer=nl.sbuf) for _ in range(6)]
    nisa.dma_copy(dst=t[0], src=a)
    gp.call("triple", ins=[t[0]], outs=[t[1]])
    nisa.tensor_scalar(dst=t[2], data=t[1], op0=nl.add, operand0=1.0, engine=nisa.engine.vector)
    gp.call("mix", ins=[t[2], t[0]], outs=[t[3]])
    nisa.tensor_scalar(dst=t[4], data=t[3], op0=nl.multiply, operand0=2, engine=nisa.engine.gpsimd)
    gp.call("triple", ins=[t[4]], outs=[t[5]])
    nisa.dma_copy(dst=o6, src=t[5])
    nisa.dma_copy(dst=o4, src=t[3])
    return o6, o4


def make_inputs(rng):
    return {"a": rng.integers(-1000, 1000, (P, F)).astype(np.int32)}


def reference(a):
    t4 = (3 * a + 1) ^ a
    return [(3 * (2 * t4)).astype(np.int32), t4.astype(np.int32)]
