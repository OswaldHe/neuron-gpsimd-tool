# Minimal NKI kernel: HBM -> SBUF -> (GpSimd copy) -> SBUF -> HBM.
# The GpSimd tensor_copy is the instruction we later replace with a KERNEL_CALL.
import os, sys
import numpy as np
import nki, nki.language as nl, nki.isa as nisa

@nki.jit
def gpsimd_passthrough(a):
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    t_in = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    t_out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t_in, src=a)
    nisa.tensor_copy(dst=t_out, src=t_in, engine=nisa.engine.gpsimd)
    nisa.dma_copy(dst=out, src=t_out)
    return out

if __name__ == "__main__":
    a = (np.arange(128 * 64, dtype=np.float32).reshape(128, 64) * 0.25)
    out = gpsimd_passthrough(a)
    print("max abs err:", float(np.abs(np.asarray(out) - a).max()))
