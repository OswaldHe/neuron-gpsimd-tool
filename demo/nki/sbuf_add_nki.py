"""NKI part of the demo: move a [128, 64] int32 tensor HBM -> SBUF -> HBM on the GpSimd (Pool) queue.

The kernel itself only copies. After compilation, tools/neff_tool.py inserts a KERNEL_CALL to the
GPSIMD custom operator (gpsimd/sbuf_add.cpp) between the two DMAs, so the custom operator sees the
tile in SBUF, updates it in place, and the second DMA writes the result to HBM.

The NeuronCore compiler elides the GpSimd tensor_copy and puts both DMAs on the Pool engine queue;
the resulting Pool instruction stream is: SET_ORDERING_MODE, DMA (in), DMA (out), NOP.

usage: python sbuf_add_nki.py <out.neff>
  Requires the torch-neuronx venv (nki 0.7, neuronx-cc 2.28) on PATH. Compiles for trn3.
"""
import os
import shutil
import sys
import tempfile

import numpy as np

SHAPE = (128, 64)          # 128 SBUF partitions x 64 int32 = 256 bytes per partition
DTYPE = np.int32


def build_kernel():
    import nki
    import nki.isa as nisa
    import nki.language as nl

    @nki.jit
    def sbuf_add_host_kernel(a):
        out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
        t_in = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
        t_out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t_in, src=a)                                    # HBM -> SBUF
        nisa.tensor_copy(dst=t_out, src=t_in, engine=nisa.engine.gpsimd)  # elided by the compiler
        nisa.dma_copy(dst=out, src=t_out)                                 # SBUF -> HBM
        return out

    return sbuf_add_host_kernel


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    out_neff = os.path.abspath(sys.argv[1])
    os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn3")
    os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "1")
    os.environ["NKI_DISABLE_COMPILE_CACHE"] = "1"
    art = tempfile.mkdtemp(prefix="nki_sbuf_add_")
    os.environ["NKI_ARTIFACTS_DIR"] = art
    kernel = build_kernel()
    a = np.zeros(SHAPE, dtype=DTYPE)
    try:
        # The standalone (numpy) path compiles to <artifacts>/kernel.neff, then tries to execute it.
        # Execution needs the nrtpy package; we only need the NEFF, so a missing runtime is fine.
        kernel(a)
    except ImportError as e:
        if "nki.runtime" not in str(e):
            raise
    neff = os.path.join(art, "kernel.neff")
    if not os.path.exists(neff):
        raise SystemExit(f"NKI compilation did not produce {neff}")
    shutil.copy(neff, out_neff)
    shutil.rmtree(art, ignore_errors=True)
    print(f"NKI kernel compiled for trn3: {out_neff} ({os.path.getsize(out_neff)} bytes)")


if __name__ == "__main__":
    main()
