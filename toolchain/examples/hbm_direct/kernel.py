"""hbm_direct: the GPSIMD cores read and write device memory (HBM) tensors directly.

  table  [1024, 1024] float32 input that never enters SBUF (4 MB)
  out    [128, 64]  = table.flat[idx]          gathered by the GPSIMD from HBM into an SBUF tile, then DMA
  direct [128, 64]  = 2 * idx + 1  (int32)     written by the GPSIMD straight into the output tensor's HBM
gpsimd-run maps the tensors listed in HBM through a GPSIMD address-translation window and passes their
core addresses in the int32 input gp_hbm (gp::hbm_ptr in the operator).
"""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

import gpsimd_nki as gp

OPS = ["op.cpp"]
HBM = ["table", "output_1"]
P, F = 128, 64


@nki.jit
def kernel(idx, table, gp_hbm):
    out = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.shared_hbm)
    direct = nl.ndarray((P, F), dtype=nl.int32, buffer=nl.shared_hbm)
    ti = nl.ndarray((P, F), dtype=nl.int32, buffer=nl.sbuf)
    tp = nl.ndarray((1, 16), dtype=nl.int32, buffer=nl.sbuf)
    to = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ti, src=idx)
    nisa.dma_copy(dst=tp, src=gp_hbm)
    gp.call("hbm_gather", ins=[ti, tp], outs=[to])
    nisa.dma_copy(dst=out, src=to)
    return out, direct


def make_inputs(rng):
    return {"idx": rng.integers(0, 1024 * 1024, (P, F)).astype(np.int32),
            "table": rng.normal(0, 1, (1024, 1024)).astype(np.float32),
            "gp_hbm": np.zeros((1, 16), np.int32)}


def reference(idx, table, gp_hbm):
    return [table.ravel()[idx], (2 * idx + 1).astype(np.int32)]
