import os, sys, numpy as np
import nki, nki.language as nl, nki.isa as nisa
@nki.jit
def k(a, b):
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    ta = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    tb = nl.ndarray(b.shape, dtype=b.dtype, buffer=nl.sbuf)
    tc = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=ta, src=a)
    nisa.dma_copy(dst=tb, src=b)
    nisa.tensor_tensor(dst=tc, data1=ta, data2=tb, op=nl.bitwise_xor, engine=nisa.engine.gpsimd)
    nisa.dma_copy(dst=out, src=tc)
    return out
dt = getattr(np, sys.argv[1]) if len(sys.argv) > 1 else np.int32
shape = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (128, 64)
a = np.zeros(shape, dt); b = np.zeros(shape, dt)
try: k(a, b)
except ImportError: pass
