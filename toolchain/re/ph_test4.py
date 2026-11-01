import sys, numpy as np
import nki, nki.language as nl, nki.isa as nisa
MARK = 1048576.0
def gpsimd_call(site, ins, outs):
    tok = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    for k, t in enumerate(ins):
        nisa.tensor_scalar(dst=tok, data=t[0:1, 0:1], op0=nl.add, operand0=(MARK + site * 64 + k) if k == 0 else tok,
                           engine=nisa.engine.gpsimd)
    for j, t in enumerate(outs):
        nisa.iota(dst=t, pattern=[[0, t.shape[1]]], offset=int(MARK) + site * 64 + 32 + j, channel_multiplier=0)
        nisa.tensor_scalar(dst=t[0:1, 0:1], data=t[0:1, 0:1], op0=nl.add, operand0=tok, engine=nisa.engine.gpsimd)
@nki.jit
def k(a, b):
    out = nl.ndarray((32, 128), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    ta = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    tb = nl.ndarray(b.shape, dtype=b.dtype, buffer=nl.sbuf)
    tb2 = nl.ndarray(b.shape, dtype=b.dtype, buffer=nl.sbuf)
    tc = nl.ndarray((32, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ta, src=a)
    nisa.dma_copy(dst=tb, src=b)
    nisa.tensor_tensor(dst=tb2, data1=tb, data2=tb, op=nl.add)      # producer on another engine
    gpsimd_call(0, [ta, tb2], [tc])
    nisa.dma_copy(dst=out, src=tc)
    return out
try: k(np.zeros((128, 64), np.int8), np.zeros((64, 32), np.float32))
except ImportError: pass
