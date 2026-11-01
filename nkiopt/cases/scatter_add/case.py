"""scatter_add: nkilib experimental.misc.scatter_add with duplicate indices (unique_indices=False).

input[index[i], :] += src[i, :] with repeated indices -- embedding-gradient accumulation, or (D = 1, as here)
per-sequence token-count histograms for frequency/presence penalties in LLM sampling:
rows = b * V + token over B sequences of LG generated tokens (Zipf tokens: many repeats).
nkilib walks the K indices in 128-row tiles strictly in sequence (a later tile may hit the same rows):
per tile an index-equality matrix (broadcast DMA + compare), a PE matmul that pre-combines duplicates,
an indirect gather of the current rows (SW DGE), an add and an indirect scatter (SW DGE).
GPSIMD: one call. Core c owns destination rows [c*N/8, (c+1)*N/8) = 16 partitions of the [128, N/128]
SBUF copy of input: it scans all K indices once (iDMA-streamed), keeps (row, value) pairs of its range,
then applies them in index order (soft-float fp32 adds, same order as a sequential scatter_add) to its
rows staged in local RAM, and writes them back with the iDMA.
"""
import os

import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

import gpsimd_nki as gp
from nkilib.experimental.misc.scatter_add import scatter_add

B = int(os.environ.get("B", "8"))
V = int(os.environ.get("V", "32768"))
LG = int(os.environ.get("LG", "2048"))
N, K = B * V, B * LG
F = N // 128
OPS = ["op.cpp"]
PARAMS = {"rows": N, "D": 1, "K": K, "sequences": B, "vocab": V, "tokens_per_seq": LG}


@nki.jit
def baseline_kernel(inp, index, src):
    out = nl.ndarray((N, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    t = nl.ndarray((128, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=inp.reshape((128, F)))
    nisa.dma_copy(dst=out.reshape((128, F)), src=t)
    scatter_add(out, 0, index, src, unique_indices=False)
    return out


@nki.jit
def gpsimd_kernel(inp, index, src):
    out = nl.ndarray((N, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    t = nl.ndarray((128, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=inp.reshape((128, F)))
    ix = nl.ndarray((128, K // 128), dtype=nl.int32, buffer=nl.sbuf)
    sv = nl.ndarray((128, K // 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ix, src=index.reshape((128, K // 128)))
    nisa.dma_copy(dst=sv, src=src.reshape((128, K // 128)))
    o = nl.ndarray((128, F), dtype=nl.float32, buffer=nl.sbuf)
    gp.call("scatter_add_f32", ins=[t, ix, sv], outs=[o])
    nisa.dma_copy(dst=out.reshape((128, F)), src=o)
    return out


def make_inputs(rng):
    tok = np.minimum(rng.zipf(1.2, (B, LG)), V) - 1
    index = (np.arange(B)[:, None] * V + tok).reshape(K).astype(np.int32)
    inp = rng.integers(0, 4, (N, 1)).astype(np.float32)          # existing counts
    src = rng.integers(1, 4, (K, 1)).astype(np.float32)          # per-occurrence weights
    return {"inp": inp, "index": index, "src": src}


def reference(inp, index, src):
    out = inp.astype(np.float32).copy()
    np.add.at(out[:, 0], index, src[:, 0])                       # integers: exact in any order
    return [out]
