# NKI test kernel (int32): HBM -> SBUF -> HBM. Same structure as kernel.py; the compiler elides the
# GpSimd copy and puts both DMAs on the Pool (GpSimd) queue: Pool stream = b1, d4 (in), d4 (out), a4.
# The KERNEL_CALL injected between the two DMAs operates on the SBUF tile in place.
import numpy as np
import nki, nki.language as nl, nki.isa as nisa

@nki.jit
def gpsimd_passthrough_i32(a):
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    t_in = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    t_out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t_in, src=a)
    nisa.tensor_copy(dst=t_out, src=t_in, engine=nisa.engine.gpsimd)
    nisa.dma_copy(dst=out, src=t_out)
    return out

if __name__ == "__main__":
    a = np.arange(128 * 64, dtype=np.int32).reshape(128, 64)
    try:
        gpsimd_passthrough_i32(a)
    except ImportError as e:   # execution needs nrtpy; the compiled NEFF is what we want
        print("compiled (execution skipped):", str(e)[:60])
