# call site with 2 inputs of different dtypes/shapes and 1 output of another dtype/shape
import sys, numpy as np
import nki, nki.language as nl, nki.isa as nisa
dt0, dt1, dto = sys.argv[1:4]
@nki.jit
def k(a, b):
    out = nl.ndarray((64, 128), dtype=getattr(nl, dto), buffer=nl.shared_hbm)
    ta = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    tb = nl.ndarray(b.shape, dtype=b.dtype, buffer=nl.sbuf)
    tc = nl.ndarray((64, 128), dtype=getattr(nl, dto), buffer=nl.sbuf)
    tdummy = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ta, src=a)
    nisa.dma_copy(dst=tb, src=b)
    # placeholders: one per input (reads it), one per output (writes it)
    nisa.tensor_scalar(dst=tdummy, data=ta[:, 0:1], op0=nl.add, operand0=1001.0, engine=nisa.engine.gpsimd)
    nisa.tensor_scalar(dst=tdummy, data=tb[:, 0:1], op0=nl.add, operand0=1002.0, engine=nisa.engine.gpsimd)
    nisa.iota(dst=tc, pattern=[[0, 128]], offset=1003, channel_multiplier=0)
    nisa.dma_copy(dst=out, src=tc)
    return out
try: k(np.zeros((128, 64), getattr(np, dt0) if dt0 != "bfloat16" else np.float32), np.zeros((128, 32), getattr(np, dt1)))
except ImportError: pass
