import sys, json, numpy as np
import nki, nki.language as nl, nki.isa as nisa
MARK = 1048576
SITES = []
def gpsimd_call(name, ins, outs):
    s = len(SITES)
    SITES.append({"site": s, "kernel": name, "ins": [[list(t.shape), str(t.dtype)] for t in ins], "outs": [[list(t.shape), str(t.dtype)] for t in outs]})
    tok = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=tok, value=float(MARK + s * 64), engine=nisa.engine.gpsimd)
    for t in ins:
        nisa.tensor_scalar(dst=tok, data=t[0:1, 0:1], op0=nl.add, operand0=tok, engine=nisa.engine.gpsimd)
    for j, t in enumerate(outs):
        nisa.iota(dst=t, pattern=[[0, t.shape[1]]], offset=MARK + s * 64 + 32 + j, channel_multiplier=0)
        nisa.tensor_scalar(dst=t[0:1, 0:1], data=t[0:1, 0:1], op0=nl.add, operand0=tok, engine=nisa.engine.gpsimd)
@nki.jit
def k(a, b):
    out = nl.ndarray((128, 64), dtype=nl.int16, buffer=nl.shared_hbm)
    ta = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    tb = nl.ndarray(b.shape, dtype=b.dtype, buffer=nl.sbuf)
    tc = nl.ndarray((128, 64), dtype=nl.int16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ta, src=a)
    nisa.dma_copy(dst=tb, src=b)
    gpsimd_call("k_first", [ta, tb], [tc])
    gpsimd_call("k_second", [tc], [tc])
    nisa.dma_copy(dst=out, src=tc)
    return out
try: k(np.zeros((128, 64), np.uint8), np.zeros((32, 16), np.float16))
except ImportError: pass
print(json.dumps(SITES))
