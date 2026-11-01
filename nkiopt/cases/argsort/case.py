"""argsort: nkilib experimental.subkernels.argsort_unstable (MoE token grouping) vs a GPSIMD radix sort.

nkilib sorts a [1, N] row with N/8 sequential passes of max8 + nc_match_replace8 over the whole row
(O(N^2) work on one Vector-engine lane, 2N/8 dependent instructions). The GPSIMD operator is a stable
LSD radix sort (8-bit digits, constant digits skipped) spread over the 8 GPSIMD cores.

Inputs (env): N (default 4096), KEYS = experts (expert ids of T*K routed slots as float32, E=64; the
permute_routed_tokens use) or random (arbitrary float32).
"""
import os

import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

import gpsimd_nki as gp
from nkilib.experimental.subkernels.argsort_unstable import argsort_unstable

OPS = ["op.cpp"]
CFLAGS = ["-DARGSORT_TIMING"] if os.environ.get("TIMING") else []
N = int(os.environ.get("N", "4096"))
KEYS = os.environ.get("KEYS", "experts")
PARAMS = {"N": N, "keys": KEYS}


@nki.jit
def baseline_kernel(data):
    return argsort_unstable(data, descending=False)


@nki.jit
def gpsimd_kernel(data):
    # argsort indices row-major over 128 partitions: position j at [j // W, j % W] (W = N / 128), so the
    # 8 cores' scattered stores land in different partitions; N = 128 gives the [T*K, 1] partition-major
    # layout permute_routed_tokens consumes.
    W = N // 128
    out = nl.ndarray((1, N), dtype=nl.uint32, buffer=nl.shared_hbm)
    d = nl.ndarray((1, N), dtype=nl.float32, buffer=nl.sbuf)
    idx = nl.ndarray((128, W), dtype=nl.uint32, buffer=nl.sbuf)
    tmp = nl.ndarray((128, W), dtype=nl.uint32, buffer=nl.sbuf)         # ping-pong permutation
    hist = nl.ndarray((8, 128), dtype=nl.uint32, buffer=nl.sbuf)        # per-core digit histograms
    nisa.dma_copy(dst=d, src=data)
    gp.call("argsort_f32", ins=[d, tmp, hist], outs=[idx])
    nisa.dma_copy(dst=out.reshape((128, W)), src=idx)
    return out


def make_inputs(rng):
    if KEYS == "experts":
        x = rng.integers(0, 64, (1, N)).astype(np.float32)
    else:
        x = rng.standard_normal((1, N)).astype(np.float32) * 100
    return {"data": x}


def reference(data):
    return [np.argsort(data[0], kind="stable").astype(np.uint32)[None, :]]


def check(outs, inputs):
    """Any valid ascending argsort passes (nkilib's is unstable); report whether it is the stable one."""
    idx = outs[0][0].astype(np.int64)
    x = inputs["data"][0]
    if not np.array_equal(np.sort(idx), np.arange(N)):
        return False, "not a permutation"
    if np.any(np.diff(x[idx]) < 0):
        return False, "values not ascending"
    stable = np.array_equal(idx, np.argsort(x, kind="stable"))
    return True, f"valid ascending argsort of {N} keys ({'= numpy stable argsort, bit-exact' if stable else 'unstable tie order'})"
