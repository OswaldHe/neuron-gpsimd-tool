"""ds41_moe: DeepSeek-V4.1-Flash routed experts at decode, one NeuronCore of one TP rank.

Each of the 64 ranks holds 6 of the 384 routed experts (MXFP4 w1 / w3 / w2, inter 2304, dim 5120); every
rank sees all T rows of the step (bs sequences x 6 rows: 1 committed + 5 drafted tokens) and the router's
`wexp[t, e]` (the routing weight of local expert e for row t, 0 when not picked). A decode step at bs10
picks ~3.7 of a rank's 6 experts, at bs1 ~0.5.

baseline_kernel: the port's hand-optimized `mk_moe.routed_experts` (r1_decode copy, unmodified, ds41d/):
`build_expert_lists` (a `nonzero_with_count` over the transposed mask) and then, for every local expert,
its own runtime-bounded `nl.fori_loop` over its row tiles (0 or 1 at decode): gather the rows, MX fp4
w1 / w3, SwiGLU x weight, w2, scatter-add into the fp32 accumulator. An expert nobody picked streams no
weights, but each of the 6 loops costs its entry / exit and back-edge (engine drain, barrier, semaphore
reset) whether it runs or not.

gpsimd_kernel: one GPSIMD call (op.cpp `moe_sched`) reads `wexp`, builds every expert's row list and the
list of ACTIVE experts (in expert order) with their count; NKI then runs ONE `nl.fori_loop` over the
active experts, the body the port's expert tile (`body_n`) with the expert's weights addressed by the
loop's expert id (runtime offsets into the stacked weights). Same arithmetic, same expert order.

T = 6 * BS rows (BS=10 by default; --set BS=1).
"""
import math
import os

import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl
import ml_dtypes

import gpsimd_nki as gp
from ds41d import mk_moe as M
from ds41d import mk_mx as X

BS = int(os.environ.get("BS", "10"))
T = 6 * BS
PT = 128
TP = (T + PT - 1) // PT * PT
NL, NE, TOPK = 6, 384, 6
DIM, INTER = 5120, 2304
LIMIT = 10.0
RANK = int(os.environ.get("RANK", "0"))
OPS = ["op.cpp"]
CFLAGS = [f"-DMS_TP={TP}", f"-DMS_NL={NL}"]
PARAMS = {"T": T, "bs": BS, "local_experts": NL, "dim": DIM, "inter": INTER}


@nki.jit
def baseline_kernel(x, wexp, w1, s1, w3, s3, w2, s2):
    eacc = nl.ndarray((TP + PT, DIM), dtype=nl.float32, buffer=nl.shared_hbm)
    M.routed_experts(0, NL, TP, x, wexp, w1.view(X.FP4_X4), s1, w3.view(X.FP4_X4), s3, w2.view(X.FP4_X4), s2,
                     DIM, INTER, 0, INTER, LIMIT, eacc, zero_acc=True)
    return eacc


@nki.jit
def gpsimd_kernel(x, wexp, w1, s1, w3, s3, w2, s2):
    raise NotImplementedError


# ----------------------------------------------------------------------------- inputs, reference, check
def make_inputs(rng):
    bf = ml_dtypes.bfloat16
    x = np.zeros((TP + PT, DIM), np.float32)
    x[:T] = rng.standard_normal((T, DIM))
    wexp = np.zeros((TP + PT, NL), np.float32)
    for t in range(T):
        picks = rng.choice(NE, TOPK, replace=False)
        w = np.sqrt(np.log1p(np.exp(rng.standard_normal(TOPK))))
        w = 1.5 * w / w.sum()
        for e, v in zip(picks, w):
            if RANK * NL <= e < (RANK + 1) * NL and not os.environ.get("NOACTIVE"):
                wexp[t, e - RANK * NL] = v
    def mx(k, n):
        words = rng.integers(0, 1 << 16, size=(NL, k // 4, n), dtype=np.uint16).view(np.int16)
        scales = rng.integers(118, 122, size=(NL, k // 32, n), dtype=np.uint8)
        return words, scales
    w1, s1 = mx(DIM, INTER)
    w3, s3 = mx(DIM, INTER)
    w2, s2 = mx(INTER, DIM)
    return {"x": x.astype(bf), "wexp": wexp, "w1": w1, "s1": s1, "w3": w3, "s3": s3, "w2": w2, "s2": s2}


FP4 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def _dequant4(words, scales):
    """[K/4, N] int16 words (nibble j = k 4p + j) and [K/32, N] e8m0 scales -> [K, N] float32"""
    w = words.view(np.uint16)
    k4, n = w.shape
    nib = np.stack([(w >> (4 * j)) & 0xF for j in range(4)], 1).reshape(k4 * 4, n)
    return FP4[nib] * np.exp2(np.repeat(scales.astype(np.float32), 32, 0) - 127.0)


def _qrows(x):
    xb = x.reshape(x.shape[0], -1, 32)
    amax = np.abs(xb).max(-1, keepdims=True)
    e = np.where(amax > 0, np.floor(np.log2(np.where(amax > 0, amax, 1.0))) - 7.0, 0.0)
    q = (xb / np.exp2(e)).astype(ml_dtypes.float8_e4m3fn).astype(np.float32)
    return (q * np.exp2(e)).reshape(x.shape)


def reference(x, wexp, w1, s1, w3, s3, w2, s2):
    return [np.zeros((TP + PT, DIM), np.float32)]


_BASE = {}


def check(outs, inputs, which):
    out = outs[0]
    x = inputs["x"].astype(np.float32)
    wexp = inputs["wexp"]
    want = np.zeros((T, DIM), np.float32)
    active = [e for e in range(NL) if np.any(wexp[:T, e] > 0)]
    for e in active:
        rows = np.flatnonzero(wexp[:T, e] > 0)
        xq = _qrows(x[rows])
        g = (xq @ _dequant4(inputs["w1"][e], inputs["s1"][e])).astype(ml_dtypes.bfloat16).astype(np.float32)
        u = (xq @ _dequant4(inputs["w3"][e], inputs["s3"][e])).astype(ml_dtypes.bfloat16).astype(np.float32)
        g = np.minimum(g, LIMIT)
        u = np.clip(u, -LIMIT, LIMIT)
        y = (g / (1 + np.exp(-g)) * u * wexp[rows, e:e + 1]).astype(ml_dtypes.bfloat16).astype(np.float32)
        want[rows] += _qrows(y) @ _dequant4(inputs["w2"][e], inputs["s2"][e])
    got = out[:T]
    rel = 0.0 if not active and not np.any(got) else np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-30)
    if not rel < 3e-2 or np.any(out[T:TP] != 0):
        return False, f"routed-expert output differs from the numpy MX emulation: rel L2 {rel:.3g}"
    msg = f"{len(active)} of {NL} local experts active, {int((wexp[:T] > 0).sum())} (row, expert) pairs; output matches numpy MX emulation (rel L2 {rel:.1e})"
    if which == "baseline":
        _BASE["out"] = out.copy()
    elif "out" in _BASE:
        if not np.array_equal(out.view(np.uint32), _BASE["out"].view(np.uint32)):
            return False, msg + "; but differs from the hand-optimized NKI version's output"
        msg += "; bit-identical to the hand-optimized NKI version"
    return True, msg
