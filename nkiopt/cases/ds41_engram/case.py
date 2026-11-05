"""ds41_engram: DeepSeek-V4.1-Flash engram read-out (prefill), one NeuronCore of one TP rank.

The engram block (layers 1 and 14) reads, for every token, 24 n-gram rows (3 n-gram sizes x 8 hash heads,
256 bf16 each) from a 384M-row table that is sharded by row over the 64 ranks (~6.0M rows, 3.1 GB a rank),
projects them with the fp8 `wkv` (6144 -> 5 x 5120, MX matmul) and gates them into the 4-stream residual:

    emb[t, j]  = table[ids[t, j]] if this rank holds row ids[t, j] else 0       (this rank's part)
    emb_sum    = all_reduce(emb) over the 64 ranks                              (collective: not run)
    ekv        = emb_sum @ wkv^T                                                (fp8 MX GEMM, 25600 columns)
    stream0    = engram_gate(strm, ekv, q*k weights)                            (4 x 5120 stream)

baseline_kernel is the port's hand-optimized NKI (`r0_e_prefill/mk_common.py: engram_gather`, `mk_mx.gemm_rows`,
`engram_gate_tile`, unmodified copies in ds41k/), run on one core (the deployment splits the rows and GEMM
columns over the two cores of a logical NeuronCore; every part halves). Its gather issues one indirect DMA per
128 rows and hash column, i.e. all T x 24 rows, and masks the ~98.4 % that belong to other ranks.

gpsimd_kernel: one GPSIMD call (op.cpp `engram_hits`) scans the T x 24 ids and emits only this rank's hits
(~0.375 a token: a 6.0M-row shard overlaps at most two of the 24 ~16M-row hash columns) as compact
(table row, emb row) lists plus their chunk count; NKI zero-fills emb, gathers / scatters just those rows in a
runtime-bounded loop (`nl.fori_loop` over the count), and runs the same GEMM and gate code.

Ids are distributed as the model's hash makes them: column j of layer 1 holds global rows
[offset_j, offset_j + p_j), p_j the j-th prime above 16M (NgramHashState), a token's id uniform in it.
T=1024 (an in1024 prefill) by default; --set T=2048 for an 8k-prompt chunk. RANK picks the shard.
"""
import math
import os

import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import oob_mode
import ml_dtypes

import gpsimd_nki as gp
from ds41k import mk_common as C
from ds41k import mk_mx as X

T = int(os.environ.get("T", "1024"))
RANK = int(os.environ.get("RANK", "21"))
WORLD = 64
NCOL, HD = 24, 256
DIM, HC = 5120, 4
E_IN, E_OUT, WIDTH = NCOL * HD, (HC + 1) * DIM, HC * DIM
PT = 128
TP = (T + PT - 1) // PT * PT
NT = TP // PT
CAP = min(TP * NCOL, 4 * TP)             # hit-list capacity (a shard overlaps <= 2 columns: <= 2 hits a token)


def _primes(n, start=16_000_000 - 1):
    out, c = [], start
    while len(out) < n:
        c += 1
        if all(c % d for d in range(2, int(c ** 0.5) + 1)):
            out.append(c)
    return out


PRIMES = _primes(NCOL)                                   # layer 1's 24 columns (n-gram size major, then head)
OFFSETS = np.cumsum([0] + PRIMES[:-1]).astype(np.int64)
NROWS = int(sum(PRIMES))                                 # 384,006,168 (config.engram_num_embeddings[0])
ROWW = int(os.environ.get("ROWW", str(-(-NROWS // WORLD))))   # rows a rank holds
OPS = ["op.cpp"]
NCH = 2 * TP // PT                       # static hit capacity in 128-row DMAs (<= 2 hits a token, see op.cpp)
# gather schedule: a runtime-bounded fori_loop over the hit count (~0.375 hits a token: 3 x 128-row DMAs at
# T=1024), or NCH static DMAs with padding (no loop: a dynamic loop's block boundary drains every engine, which
# at decode sizes costs the overlap of the gather with the GEMM's weight stream). Default: loop from Tp >= 512.
DYN = int(os.environ.get("DYN", "1" if TP >= 512 else "0")) == 1
assert ROWW < min(PRIMES), "a shard must be smaller than every hash column (then <= 2 hits a token)"
CFLAGS = [f"-DEG_NT={NT}", f"-DEG_NCOL={NCOL}", f"-DEG_ROWW={ROWW}", f"-DEG_CAP={CAP}", f"-DEG_NCH={NCH}"]
PARAMS = {"T": T, "rank": RANK, "rows_per_rank": ROWW, "hash_cols": NCOL, "head_dim": HD, "wkv": f"{E_IN}x{E_OUT} fp8"}
TILES = [(i * PT, min(PT, T - i * PT)) for i in range(NT)]
DEBUG = bool(os.environ.get("DEBUG"))          # also return ekv (to check the GEMM on its own)
STAGE = os.environ.get("STAGE", "readout")    # "gather": only build emb (the stage the GPSIMD call changes)
PRIO = int(os.environ.get("PRIO", "-1"))      # trn3 DMA QoS class (0 = highest) of the GPSIMD version's retrieval DMAs
QOS = {"priority": PRIO} if PRIO >= 0 else {}


# ----------------------------------------------------------------------------- the shared part
def _project_and_gate(emb, e_wkv, e_wkv_s, e_qk, strm):
    """ekv = emb @ wkv^T (MX fp8), then the gate into the stream: the port's code, both versions."""
    if STAGE == "gather":
        return ()
    ekv = nl.ndarray((TP, E_OUT), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="ekv")
    X.gemm_rows(emb, TP, E_IN, e_wkv.view(X.FP8_X4), e_wkv_s, ekv, 0, E_OUT)
    out = nl.ndarray((T, WIDTH), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="stream0")
    for r0, nr in TILES:
        C.engram_gate_tile(r0, nr, strm, ekv, e_qk, out, HC, DIM)
    return (out, ekv) if DEBUG else (out,)


def _zero_emb(emb, **qos):
    zero = nl.ndarray((PT, E_IN), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=zero, value=0.0)
    for i in range(NT):
        nisa.dma_copy(dst=emb[i * PT:(i + 1) * PT, :], src=zero, **qos)


@nki.jit
def baseline_kernel(hash_ids, etable, erow_start, e_wkv, e_wkv_s, e_qk, strm):
    emb = nl.ndarray((TP, E_IN), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="emb")
    C.engram_gather(hash_ids, etable, erow_start, emb, TILES, NCOL, HD, ROWW)
    if T < TP:                                   # as engram_readout: the padding rows are zero
        padz = nl.ndarray((PT, E_IN), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(padz, 0.0)
        nisa.dma_copy(dst=emb[T:TP, :], src=padz[0:TP - T, :])
    return (*_project_and_gate(emb, e_wkv, e_wkv_s, e_qk, strm), emb)


@nki.jit
def gpsimd_kernel(hash_ids, etable, erow_start, e_wkv, e_wkv_s, e_qk, strm):
    # (erow_start is 0: the ids arrive local to the shard, as the model passes them)
    i32 = nl.int32
    ids_sb = nl.ndarray((PT, NT * NCOL), dtype=i32, buffer=nl.sbuf)      # [p, i*24 + j] = ids[i*128 + p, j]
    if T == TP:
        nisa.dma_copy(dst=ids_sb, src=hash_ids.ap(pattern=[[NCOL, PT], [PT * NCOL, NT], [1, NCOL]], offset=0))
    else:                                        # padding rows: id -1 (never this rank's)
        nisa.memset(dst=ids_sb, value=-1)
        for r0, nr in TILES:
            i = r0 // PT
            nisa.dma_copy(dst=ids_sb[0:nr, i * NCOL:(i + 1) * NCOL], src=hash_ids[r0:r0 + nr, :])
    emb = nl.ndarray((TP, E_IN), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="emb")
    rows = emb.reshape((TP * NCOL, HD))
    if not DYN:
        # static: NCH gathers / scatters of 128 hits each (padding entries skipped)
        src_t = nl.ndarray((PT, NCH), dtype=i32, buffer=nl.sbuf)
        dst_t = nl.ndarray((PT, NCH), dtype=i32, buffer=nl.sbuf)
        gp.call("engram_hits_t", ins=[ids_sb], outs=[src_t, dst_t])
        _zero_emb(emb, **QOS)
        for b in range(NCH):
            g = nl.ndarray((PT, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=g, src=etable.ap(pattern=[[HD, PT], [1, HD]], offset=0, vector_offset=src_t[:, b:b + 1],
                                               indirect_dim=0), **QOS)
            nisa.dma_copy(dst=rows.ap(pattern=[[HD, PT], [1, HD]], offset=0, vector_offset=dst_t[:, b:b + 1],
                                      indirect_dim=0), src=g, oob_mode=oob_mode.skip, **QOS)
        return (*_project_and_gate(emb, e_wkv, e_wkv_s, e_qk, strm), emb)
    hits_sb = nl.ndarray((2, CAP), dtype=i32, buffer=nl.sbuf)
    cnt_sb = nl.ndarray((1, 1), dtype=i32, buffer=nl.sbuf)
    gp.call("engram_hits", ins=[ids_sb], outs=[hits_sb, cnt_sb])
    hits = nl.ndarray((2, CAP), dtype=i32, buffer=nl.private_hbm)
    nisa.dma_copy(dst=hits, src=hits_sb)
    _zero_emb(emb)
    h3 = hits.reshape((2, CAP // PT, PT))
    reg = nisa.register_alloc()
    nisa.register_load(reg, cnt_sb)

    def chunk(b):
        si = nl.ndarray((PT, 1), dtype=i32, buffer=nl.sbuf)
        nisa.dma_copy(dst=si, src=h3.ap(pattern=[[1, PT], [1, 1]], offset=0, scalar_offset=b, indirect_dim=1))
        di = nl.ndarray((PT, 1), dtype=i32, buffer=nl.sbuf)
        nisa.dma_copy(dst=di, src=h3.ap(pattern=[[1, PT], [1, 1]], offset=CAP, scalar_offset=b, indirect_dim=1))
        g = nl.ndarray((PT, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=g, src=etable.ap(pattern=[[HD, PT], [1, HD]], offset=0, vector_offset=si, indirect_dim=0))
        nisa.dma_copy(dst=rows.ap(pattern=[[HD, PT], [1, HD]], offset=0, vector_offset=di, indirect_dim=0), src=g,
                      oob_mode=oob_mode.skip)

    nl.fori_loop(0, reg, chunk, step=1)
    return (*_project_and_gate(emb, e_wkv, e_wkv_s, e_qk, strm), emb)


# ----------------------------------------------------------------------------- inputs, reference, check
def make_inputs(rng):
    bf = ml_dtypes.bfloat16
    row_start = RANK * ROWW
    cols = [OFFSETS[j] + rng.integers(0, PRIMES[j], T) for j in range(NCOL)]
    ids = (np.stack(cols, 1) - row_start).astype(np.int64)
    assert np.all(np.abs(ids) < 2 ** 31)
    # the table: random bf16 of magnitude [2^-2, 2^-1), random sign (generated as raw bits, 3 GB)
    bits = rng.integers(0, 1 << 16, size=(ROWW, HD), dtype=np.uint16)
    bits &= np.uint16(0x807F)
    bits |= np.uint16(0x3E80)
    etable = bits.view(bf)
    # wkv: fp8 e4m3 [E_OUT, E_IN] with e8m0 scales per 32 x 32 block, in the MX layout (mk_model.mx_fp8)
    w8 = rng.integers(0, 256, size=(E_OUT, E_IN), dtype=np.uint8)
    w8[(w8 & 0x7F) == 0x7F] ^= 0x01                      # no NaN codes
    w8 = (w8 & 0xBF) | 0x00                              # exponents <= 7: |w| <= 1.875 * 2^0
    s8 = rng.integers(124, 128, size=(E_OUT // 32, E_IN // 32), dtype=np.uint8)   # ekv ~ 0.1-1
    e_wkv = np.ascontiguousarray(w8.reshape(E_OUT, E_IN // 4, 4).transpose(1, 0, 2)).view(np.int32).reshape(E_IN // 4, E_OUT)
    e_wkv_s = np.ascontiguousarray(np.repeat(s8.T, 32, axis=1))
    e_qk = (1.0 + 0.1 * rng.standard_normal((1, WIDTH))).astype(np.float32)
    strm = rng.standard_normal((T, WIDTH), dtype=np.float32).astype(bf)
    return {"hash_ids": ids.astype(np.int32), "etable": etable, "erow_start": np.zeros((1, 1), np.int32),
            "e_wkv": e_wkv, "e_wkv_s": e_wkv_s, "e_qk": e_qk, "strm": strm}


def _emb_ref(hash_ids, etable):
    ids = hash_ids.astype(np.int64)
    ok = (ids >= 0) & (ids < ROWW)
    emb = np.zeros((TP, NCOL, HD), np.float32)
    t, j = np.nonzero(ok)
    emb[t, j] = etable[ids[t, j]].astype(np.float32)
    return emb.reshape(TP, E_IN)


def _mx_dequant_w(e_wkv, e_wkv_s):
    w8 = e_wkv.view(np.uint8).reshape(E_IN // 4, E_OUT, 4).transpose(0, 2, 1).reshape(E_IN, E_OUT)
    w = w8.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    return w * np.exp2(np.repeat(e_wkv_s.astype(np.float32), 32, axis=0) - 127.0)      # [E_IN, E_OUT]


def _quantize_rows(x):
    """quantize_mx: per 32-element block, scale 2^(floor(log2 amax) - 7), e4m3fn round to nearest"""
    xb = x.reshape(x.shape[0], -1, 32)
    amax = np.abs(xb).max(-1, keepdims=True)
    e = np.where(amax > 0, np.floor(np.log2(np.where(amax > 0, amax, 1.0))) - 7.0, 0.0)
    q = (xb / np.exp2(e)).astype(ml_dtypes.float8_e4m3fn).astype(np.float32)
    return (q * np.exp2(e)).reshape(x.shape)


def _gate_ref(strm, ekv, e_qk):
    h = strm.astype(np.float32).reshape(T, HC, DIM)
    k = ekv[:T, :WIDTH].astype(np.float32).reshape(T, HC, DIM)
    v = ekv[:T, WIDTH:].astype(np.float32)
    w = e_qk.reshape(HC, DIM)
    rstd = 1.0 / np.sqrt((h * h).mean(-1) + 1e-20) / np.sqrt((k * k).mean(-1) + 1e-20)
    dot = (h * w * k).sum(-1) * rstd / math.sqrt(DIM)
    gate = 1.0 / (1.0 + np.exp(-np.sqrt(np.maximum(np.abs(dot), 1e-6)) * np.where(dot >= 0, 1.0, -1.0)))
    return (h + gate[..., None] * v[:, None, :]).reshape(T, WIDTH)


def reference(hash_ids, etable, erow_start, e_wkv, e_wkv_s, e_qk, strm):
    emb = _emb_ref(hash_ids, etable)
    if STAGE == "gather":
        return [emb.astype(ml_dtypes.bfloat16)]
    extra = [np.zeros((TP, E_OUT), ml_dtypes.bfloat16)] if DEBUG else []
    return [np.zeros((T, WIDTH), ml_dtypes.bfloat16)] + extra + [emb.astype(ml_dtypes.bfloat16)]


_BASE_OUT = {}


def check(outs, inputs, which):
    if STAGE == "gather":
        want_emb = _emb_ref(inputs["hash_ids"], inputs["etable"])
        ok = np.array_equal(outs[-1].astype(np.float32), want_emb)
        return ok, "emb exact" if ok else "emb differs"
    out, emb = outs[0], outs[-1]
    if DEBUG:                                  # keep the device's GEMM output for an offline comparison
        np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ekv_dev.npy"), outs[1].view(np.uint16))
    want_emb = _emb_ref(inputs["hash_ids"], inputs["etable"])
    if not np.array_equal(emb.astype(np.float32), want_emb):          # +0 / -0 compare equal
        bad = np.argwhere(emb.astype(np.float32) != want_emb)
        return False, f"emb differs from numpy at {len(bad)} elements, first {bad[:3].tolist()}"
    hits = int(np.count_nonzero(np.any(want_emb.reshape(TP, NCOL, HD) != 0, -1)))
    # independent check of the GEMM + gate: numpy MX emulation (activation quantization, fp8 weights)
    xq = _quantize_rows(want_emb[:T])
    rows = np.flatnonzero(np.any(want_emb[:T] != 0, 1))
    sel = rows[:64]
    w = _mx_dequant_w(inputs["e_wkv"], inputs["e_wkv_s"])
    ekv = np.zeros((T, E_OUT), np.float32)
    ekv[sel] = (xq[sel] @ w).astype(ml_dtypes.bfloat16).astype(np.float32)
    want = _gate_ref(inputs["strm"], ekv, inputs["e_qk"])
    got = out.astype(np.float32)
    x = inputs["strm"].astype(np.float32)
    d_ref = (want - x)[sel]
    rel = np.linalg.norm((got - want)[sel]) / max(np.linalg.norm(d_ref), 1e-30)
    if not rel < 2e-2:
        return False, f"engram output differs from the numpy MX emulation on hit rows: rel L2 {rel:.3g}"
    msg = (f"emb [{TP}, {E_IN}] exact ({hits} of {T * NCOL} lookups on this rank); output matches the numpy MX "
           f"emulation on {len(sel)} hit rows (rel L2 {rel:.1e})")
    if which == "baseline":
        _BASE_OUT["out"] = out.copy()
    elif "out" in _BASE_OUT:
        if not np.array_equal(out.view(np.uint16), _BASE_OUT["out"].view(np.uint16)):
            return False, "output differs from the hand-optimized NKI version's"
        msg += "; output bit-identical to the hand-optimized NKI version"
    return True, msg
