"""trn3 feature probe: can the hardware DGE (descriptor generation in the DMA engines, not on GPSIMD) issue the
indirect row gathers / scatters that Engram retrieval uses?  On trn2 (NeuronCore-v3) the arch guide says hardware
DGE does not support indirect DMA.  Pure NKI; run with bench.py --only baseline.

  MODE=swdge|hwdge|unknown   dge_mode of the indirect DMAs
  VALID=f                    fraction of in-range indices (the rest are out of range and skipped)
  NB=n                       number of 128-row indirect DMAs
  OP=gather|scatter          gather table rows into SBUF, or scatter SBUF rows into an HBM tensor
"""
import os

import ml_dtypes
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import oob_mode

MODE = os.environ.get("MODE", "swdge")
VALID = float(os.environ.get("VALID", "1.0"))
NB = int(os.environ.get("NB", "16"))
OP = os.environ.get("OP", "gather")
R, HD, PT = 262144, 256, 128
DGE = {"swdge": nisa.dge_mode.swdge, "hwdge": nisa.dge_mode.hwdge, "unknown": nisa.dge_mode.unknown}[MODE]
PARAMS = {"mode": MODE, "valid": VALID, "nb": NB, "op": OP, "rows": R, "row_values": HD}


@nki.jit
def baseline_kernel(table, idx):
    ix = nl.ndarray((PT, NB), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ix, src=idx)
    if OP == "gather":
        out = nl.ndarray((NB * PT, HD), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        for b in range(NB):
            g = nl.ndarray((PT, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(g, 0)
            nisa.dma_copy(dst=g, src=table.ap(pattern=[[HD, PT], [1, HD]], offset=0, vector_offset=ix[:, b:b + 1],
                                              indirect_dim=0), dge_mode=DGE, oob_mode=oob_mode.skip)
            nisa.dma_copy(dst=out[b * PT:(b + 1) * PT, :], src=g)
        return out
    # scatter: row b*PT+p of `table` (first NB*PT rows) goes to out row idx[p, b]
    out = nl.ndarray((R, HD), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    z = nl.ndarray((PT, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(z, 0)
    flat = out.reshape((R * HD // 2048 // PT, PT, 2048))
    for k in range(R * HD // 2048 // PT):
        nisa.dma_copy(dst=flat[k], src=z)
    for b in range(NB):
        g = nl.ndarray((PT, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=g, src=table[b * PT:(b + 1) * PT, :])
        nisa.dma_copy(dst=out.ap(pattern=[[HD, PT], [1, HD]], offset=0, vector_offset=ix[:, b:b + 1], indirect_dim=0),
                      src=g, dge_mode=DGE, oob_mode=oob_mode.skip)
    return out


gpsimd_kernel = baseline_kernel


def make_inputs(rng):
    table = rng.standard_normal((R, HD), dtype=np.float32).astype(ml_dtypes.bfloat16)
    if OP == "gather":
        idx = rng.integers(0, R, size=(PT, NB), dtype=np.int64)
    else:   # distinct destinations
        idx = rng.permutation(R)[:PT * NB].reshape(NB, PT).T.astype(np.int64)
    bad = rng.random((PT, NB)) >= VALID
    idx[bad] = R + 7
    return {"table": table, "idx": np.ascontiguousarray(idx.astype(np.int32))}


def reference(table, idx):
    if OP == "gather":
        out = np.zeros((NB * PT, HD), dtype=ml_dtypes.bfloat16)
        for b in range(NB):
            for p in range(PT):
                if idx[p, b] < R:
                    out[b * PT + p] = table[idx[p, b]]
        return out
    out = np.zeros((R, HD), dtype=ml_dtypes.bfloat16)
    for b in range(NB):
        for p in range(PT):
            if idx[p, b] < R:
                out[idx[p, b]] = table[b * PT + p]
    return out
