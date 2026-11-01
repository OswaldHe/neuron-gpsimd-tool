"""dma_skip: what an indirect-DMA row gather costs per descriptor when most indices are out of
bounds (oob_mode.skip) vs all valid. NB batched gathers of [128, KT] rows of D bf16 (like the DeepSeek
port's attn_fetch: one DMA with a 2-D vector_offset per token); VALID = fraction of in-bounds indices."""
import os
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import oob_mode

NB = int(os.environ.get("NB", "64"))
KT = int(os.environ.get("KT", "5"))
D = 512
ROWS = 10752
VALID = float(os.environ.get("VALID", "1.0"))
SKIP = os.environ.get("SKIP", "1") == "1"
OPS = []


@nki.jit
def baseline_kernel(table, idx):
    acc = nl.ndarray((128, KT, D), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)
    for b in range(NB):
        ix = nl.ndarray((128, KT), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ix, src=idx[b])
        g = nl.ndarray((128, KT, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=g, value=0.0)
        kw = dict(oob_mode=oob_mode.skip) if SKIP else {}
        nisa.dma_copy(dst=g, src=table.ap(pattern=[[D, 128 * KT], [1, D]], offset=0, vector_offset=ix, indirect_dim=0), **kw)
        nisa.tensor_tensor(dst=acc, data1=acc, data2=g, op=nl.add)
    out = nl.ndarray((128, KT, D), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=acc)
    return out


gpsimd_kernel = baseline_kernel


def make_inputs(rng):
    table = (rng.integers(-4, 5, (ROWS, D))).astype(np.float32)
    import ml_dtypes
    idx = rng.integers(0, ROWS, (NB, 128, KT)).astype(np.int32)
    oob = rng.random((NB, 128, KT)) >= VALID
    idx[oob] = ROWS + 7
    return {"table": table.astype(ml_dtypes.bfloat16), "idx": idx}


def reference(table, idx):
    import ml_dtypes
    t = table.astype(np.float32)
    acc = np.zeros((128, KT, D), np.float32)
    for b in range(NB):
        ok = idx[b] < ROWS
        g = np.where(ok[..., None], t[np.minimum(idx[b], ROWS - 1)], 0.0)
        acc = (acc + g).astype(ml_dtypes.bfloat16).astype(np.float32)
    return acc.astype(ml_dtypes.bfloat16)


def check(outs, inputs):
    return True, "timing experiment (row placement not checked)"
