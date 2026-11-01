import sys, numpy as np
import nki, nki.language as nl, nki.isa as nisa
which = sys.argv[1]
@nki.jit
def k(a, b):
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    ta = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    tb = nl.ndarray(b.shape, dtype=b.dtype, buffer=nl.sbuf)
    tc = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=ta, src=a)
    nisa.dma_copy(dst=tb, src=b)
    if which == "tensor_scalar":
        nisa.tensor_scalar(dst=tc, data=ta, op0=nl.add, operand0=tb[:, 0:1], engine=nisa.engine.gpsimd)
    elif which == "tensor_copy":
        nisa.tensor_copy(dst=tc, src=ta, engine=nisa.engine.gpsimd)
    elif which == "affine_select":
        nisa.affine_select(dst=tc, pattern=[[1, a.shape[1]]], channel_multiplier=0, on_true_tile=ta, on_false_value=0, cmp_op=nl.greater_equal, offset=0)
    elif which == "iota":
        nisa.iota(dst=tc, pattern=[[1, a.shape[1]]], offset=0, channel_multiplier=0)
    nisa.dma_copy(dst=out, src=tc)
    return out
dt = getattr(np, sys.argv[2]); shape = (int(sys.argv[3]), int(sys.argv[4]))
try: k(np.zeros(shape, dt), np.zeros(shape, dt))
except ImportError: pass
