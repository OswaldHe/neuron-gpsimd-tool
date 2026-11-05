"""trn3 feature probe: tensor indirection (TI, `.indirect(index)`), the NeuronCore-v4 gather along the free
dimension of an SBUF tile, executed by a compute engine.  Pure NKI; run with bench.py --only baseline.

  ENG=gpsimd|vector|scalar   engine of the gathering tensor_scalar (data.indirect(idx) * 1.0)
  P=16|128                   partitions of the data tile; idx is (P, K) uint16, so P*K offsets
  K=n, REP=n                 offsets per partition, number of gathering instructions
"""
import os

import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

ENG = os.environ.get("ENG", "gpsimd")
P = int(os.environ.get("P", "16"))
K = int(os.environ.get("K", "12"))
REP = int(os.environ.get("REP", "16"))
F = 16384
N = int(os.environ.get("NE", str(P * K)))    # gathered columns (P * N <= 4096 per instruction)
ENGINE = {"gpsimd": nisa.engine.gpsimd, "vector": nisa.engine.vector, "scalar": nisa.engine.scalar}[ENG]
PARAMS = {"engine": ENG, "P": P, "K": K, "rep": REP, "F": F}


@nki.jit
def baseline_kernel(data, idx):
    out = nl.ndarray((REP * P, N), dtype=nl.float32, buffer=nl.shared_hbm)
    d = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=d, src=data)
    ix = nl.ndarray((P, K), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ix, src=idx)
    for r in range(REP):
        o = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=o, data=d.indirect(ix, num_elem=N), op0=nl.multiply, operand0=1.0, engine=ENGINE)
        nisa.dma_copy(dst=out[r * P:(r + 1) * P, :], src=o)
    return out


gpsimd_kernel = baseline_kernel


def make_inputs(rng):
    data = rng.integers(0, 1 << 20, size=(P, F)).astype(np.float32)
    idx = rng.integers(0, F, size=(P, K)).astype(np.uint16)
    return {"data": data, "idx": idx}


def reference(data, idx):
    if P > 16:
        return np.zeros((REP * P, N), dtype=np.float32)
    offs = [int(idx[i % 16, i // 16]) for i in range(N)]
    one = data[:, offs]
    return np.tile(one, (REP, 1))


def check(outs, inputs):
    got, d, ix = outs[0], inputs["data"], inputs["idx"]
    if P > 16:   # probe mode: save the output for offline analysis
        np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"ti_P{P}_K{K}.npy"), got)
        return True, "saved for analysis"
    want = reference(d, ix)
    if np.array_equal(got, want):
        return True, f"gather semantics: out[p, i] = data[p, idx[i % 16, i // 16]] ({got.size} values)"
    alt = np.tile(d[np.arange(P)[:, None], ix.astype(np.int64)].repeat(1, 1), (REP, 1))
    return False, (f"mismatch: {int(np.sum(got != want))}/{got.size} differ; first row got {got[0, :8]} want {want[0, :8]}; "
                   f"per-partition interpretation matches: {alt.shape == got.shape and np.array_equal(alt, got)}")
