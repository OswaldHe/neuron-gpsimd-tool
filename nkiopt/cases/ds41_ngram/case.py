"""ds41_ngram: DeepSeek-V4.1-Flash engram n-gram hashing (both engram layers' 48 row ids per token).

NgramHashState: a token's n-gram window (its own compressed id and the 3 before it, read from a per-slot
history through the block table; pad where blocked), each shift's id times a 64-bit odd multiplier,
XOR-rolled; the 2-, 3- and 4-gram rolled values reduced modulo 8 primes each (just below 2^24), plus the
column's table offset. 64-bit integer arithmetic.

baseline_kernel: the port's NKI hash, `step_kernel._hash_rows` (unmodified copy in ds41s/), per 128-row
tile: the window gathers (token map, history write and reads through `_slot`), the products from a
[99092, 16] (hi, lo) table by indirect DMA, then the 63-bit XOR and modulo as ~250 exact int32 GpSimd
instructions: Horner's rule over 7-bit chunks, the quotient from an fp32 estimate, corrected by one prime.

gpsimd_kernel: the same window gathers (the first half of `_hash_rows`, NKI), then ONE GPSIMD call
(op.cpp `ngram_hash`) for all tiles: MULL / MULUH products, XOR, Barrett-reduced Horner over 8-bit chunks,
in registers.

The port runs this per decode step in its step prologue (T = 6 * bs rows) and hashes prefill chunks on the
host; sizes: --set B=10 --set L=6 (decode, bs10) or B=1, L=T (a prefill chunk of T rows, default 2048).
"""
import os

import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

import gpsimd_nki as gp
from ds41s import step_kernel as SK

B = int(os.environ.get("B", "1"))
L = int(os.environ.get("L", "2048"))
T = B * L
PT = 128
NT = (T + PT - 1) // PT
TILES = [(i * PT, min(PT, T - i * PT)) for i in range(NT)]
BS, H = 32, SK.H
VOCAB, VC, PAD_TOKEN, EVOCAB = 129280, 99092, 2, 16_000_000
LAYER_IDS, NGRAM, HEADS = (1, 14), 4, 8
ROW_START = (3 * 6_000_000, 3 * 6_000_000 + 11)
CTX = int(os.environ.get("CTX", "8448"))
OPS = ["op.cpp"]
CFLAGS = [f"-DNG_NT={NT}", f"-DNG_T={T}"]
PARAMS = {"spans": B, "rows_per_span": L, "T": T, "hash_cols": SK.HCOLS}


def _is_prime(n):
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


def _consts():
    primes, seen = [], set()
    for _ in LAYER_IDS:
        for _ in range(NGRAM - 1):
            cur = EVOCAB - 1
            for _ in range(HEADS):
                cur += 1
                while not _is_prime(cur) or cur in seen:
                    cur += 1
                seen.add(cur)
                primes.append(cur)
    primes = np.array(primes, np.int64).reshape(2, 24)
    offsets = np.stack([np.cumsum([0, *p[:-1]]) for p in primes])
    bound = max(1, (np.iinfo(np.int64).max // VC) // 2)
    mults = np.stack([np.random.default_rng(10007 * lid).integers(0, bound, size=(NGRAM,), dtype=np.int64) * 2 + 1
                      for lid in LAYER_IDS])
    return primes, offsets, mults


PRIMES, OFFSETS, MULTS = _consts()


# ----------------------------------------------------------------------------- kernels
def _rows(r0, nr, rpos, rhdr, ids, pm):
    """a tile's row positions, span headers, write slots (`_slot`) and token ids, as the step prologue has them"""
    row_pos = SK._i((PT, 1))
    nisa.dma_copy(dst=row_pos[0:nr, :], src=rpos[r0:r0 + nr, :])
    hdr = SK._i((PT, H))
    nisa.dma_copy(dst=hdr[0:nr, :], src=rhdr[r0:r0 + nr, :])
    ws = SK._slot(pm, hdr[0:nr, SK.C_TBASE:SK.C_TBASE + 1], row_pos, nr, BS, SK._log2(BS))
    ids_raw = SK._i((PT, 1))
    nisa.dma_copy(dst=ids_raw[0:nr, :], src=ids[r0:r0 + nr, :])
    ids_t = SK._i((PT, 1))
    SK._ts(ids_t[0:nr, :], ids_raw[0:nr, :], nl.maximum, 0)
    return row_pos, ws, hdr, ids_t


@nki.jit
def baseline_kernel(rpos, rhdr, ids, pm, history, tokmap, prod, hcol, hinv, mult):
    out = nl.ndarray((T, SK.HCOLS), dtype=nl.int32, buffer=nl.shared_hbm)
    pad_id = PAD_ID
    for r0, nr in TILES:
        row_pos, ws, hdr, ids_t = _rows(r0, nr, rpos, rhdr, ids, pm)
        hv = SK._hash_rows(nr, row_pos, ws, hdr, ids_t, pm, history, tokmap, prod, hcol, hinv, pad_id, BS, SK._log2(BS))
        nisa.dma_copy(dst=out[r0:r0 + nr, :], src=hv[0:nr, :])
    return out


def _window_tokens(T_, row_pos, ws, hdr, ids_t, pm, history, tokmap, pad_id, bs, bs_log2):
    """The first half of the port's `step_kernel._hash_rows`, verbatim: the row's compressed id written to
    the history, then the 3 earlier ids through the block table (pad from the first blocked shift on)."""
    _i, _ts, _tt, DEAD = SK._i, SK._ts, SK._tt, SK.DEAD
    T = T_
    c = SK._gather1(tokmap, ids_t, T)
    nisa.dma_copy(dst=history.ap(pattern=[[1, T], [1, 1]], vector_offset=ws[0:T, 0:1], indirect_dim=0),
                  src=c[0:T, 0:1])
    tbase = hdr[0:T, SK.C_TBASE : SK.C_TBASE + 1]
    toks = [c]
    blocked = _i((PT, 1))
    nisa.memset(blocked[0:T, :], 0)
    for s in range(1, SK.SHIFTS):
        ps = _i((PT, 1))
        _ts(ps[0:T, :], row_pos[0:T, :], nl.add, -s)
        neg = _i((PT, 1))
        _ts(neg[0:T, :], ps[0:T, :], nl.less, 0)
        _ts(ps[0:T, :], ps[0:T, :], nl.maximum, 0)
        sl = SK._slot(pm, tbase, ps, T, bs, bs_log2)
        src = SK._gather1(history, sl, T)
        dead = _i((PT, 1))
        _ts(dead[0:T, :], src[0:T, :], nl.equal, DEAD)
        _tt(blocked[0:T, :], blocked[0:T, :], neg[0:T, :], nl.maximum)
        _tt(blocked[0:T, :], blocked[0:T, :], dead[0:T, :], nl.maximum)
        d = _i((PT, 1))
        _ts(d[0:T, :], src[0:T, :], nl.multiply, -1, nl.add, pad_id)
        _tt(d[0:T, :], d[0:T, :], blocked[0:T, :], nl.multiply)
        tok = _i((PT, 1))
        _tt(tok[0:T, :], src[0:T, :], d[0:T, :], nl.add)
        toks.append(tok)
    return toks


@nki.jit
def gpsimd_kernel(rpos, rhdr, ids, pm, history, tokmap, prod, hcol, hinv, mult):
    i32 = nl.int32
    toks_all = nl.ndarray((PT, NT * SK.SHIFTS), dtype=i32, buffer=nl.sbuf)
    nisa.memset(dst=toks_all, value=0)
    for r0, nr in TILES:
        row_pos, ws, hdr, ids_t = _rows(r0, nr, rpos, rhdr, ids, pm)
        toks = _window_tokens(nr, row_pos, ws, hdr, ids_t, pm, history, tokmap, PAD_ID, BS, SK._log2(BS))
        i = r0 // PT
        for s in range(SK.SHIFTS):
            nisa.tensor_copy(dst=toks_all[0:nr, i * SK.SHIFTS + s:i * SK.SHIFTS + s + 1], src=toks[s][0:nr, :])
    hcol_sb = nl.ndarray((4, SK.HCOLS), dtype=i32, buffer=nl.sbuf)
    nisa.dma_copy(dst=hcol_sb, src=hcol)
    mult_sb = nl.ndarray((1, 16), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.dma_copy(dst=mult_sb, src=mult)
    hv = nl.ndarray((PT, NT * SK.HCOLS), dtype=i32, buffer=nl.sbuf)
    gp.call("ngram_hash", ins=[toks_all, hcol_sb, mult_sb], outs=[hv])
    out = nl.ndarray((T, SK.HCOLS), dtype=nl.int32, buffer=nl.shared_hbm)
    for r0, nr in TILES:
        i = r0 // PT
        nisa.dma_copy(dst=out[r0:r0 + nr, :], src=hv[0:nr, i * SK.HCOLS:(i + 1) * SK.HCOLS])
    return out


# ----------------------------------------------------------------------------- inputs, reference
_rng0 = np.random.default_rng(7)
_TOKMAP = _rng0.integers(0, VC, size=(VOCAB,)).astype(np.int32)
PAD_ID = int(_TOKMAP[PAD_TOKEN])


def make_inputs(rng):
    W = -(-CTX // BS) + 1
    nblocks = W * B + 8
    pool = rng.permutation(np.arange(1, nblocks))
    Hw = H + W
    pm = np.zeros((B, Hw), np.int32)
    rpos = np.zeros((T, 1), np.int32)
    rhdr = np.zeros((T, H), np.int32)
    for b in range(B):
        start = int(rng.integers(1, CTX - L - 1)) if L < CTX else 0
        if B == 1:
            start = int(os.environ.get("P0", str(max(0, CTX - L))))
        pm[b, SK.C_TBASE] = b * Hw + H
        pm[b, H:H + W] = pool[b * W:(b + 1) * W]
        rpos[b * L:(b + 1) * L, 0] = start + np.arange(L)
        rhdr[b * L:(b + 1) * L] = pm[b, :H]
    nslots = nblocks * BS
    history = rng.integers(-1, VC, size=(nslots, 1)).astype(np.int32)
    history[rng.random((nslots, 1)) < 0.05] = -1
    ids = rng.integers(0, VOCAB, size=(T, 1)).astype(np.int32)
    prod = np.zeros((VC, 16), np.int64)
    cc = np.arange(VC, dtype=np.int64)
    for li in range(2):
        for s in range(NGRAM):
            v = cc * int(MULTS[li, s])
            prod[:, li * 8 + 2 * s] = v >> 32
            prod[:, li * 8 + 2 * s + 1] = v & 0xFFFFFFFF
    prod = np.where(prod >= 2 ** 31, prod - 2 ** 32, prod).astype(np.int32)
    mu = [(1 << 64) // int(q) for q in PRIMES.reshape(-1)]           # the GPSIMD op's Barrett constants
    mu_hi = np.array([m >> 32 for m in mu], np.int64)
    mu_lo = np.array([m & 0xFFFFFFFF for m in mu], np.int64)
    mu_lo = np.where(mu_lo >= 2 ** 31, mu_lo - 2 ** 32, mu_lo)
    hcol = np.stack([PRIMES.reshape(-1), np.concatenate([OFFSETS[li] - ROW_START[li] for li in range(2)]),
                     mu_hi, mu_lo]).astype(np.int32)                   # (the port's code reads rows 0 and 1)
    hinv = (1.0 / PRIMES.reshape(1, -1).astype(np.float64)).astype(np.float32)
    mult = np.zeros((1, 16), np.uint32)
    for li in range(2):
        for s in range(NGRAM):
            m = int(MULTS[li, s])
            mult[0, li * 8 + 2 * s], mult[0, li * 8 + 2 * s + 1] = m >> 32, m & 0xFFFFFFFF
    return {"rpos": rpos, "rhdr": rhdr, "ids": ids, "pm": pm.reshape(-1, 1), "history": history,
            "tokmap": _TOKMAP.reshape(-1, 1), "prod": prod, "hcol": hcol, "hinv": hinv, "mult": mult}


def reference(rpos, rhdr, ids, pm, history, tokmap, prod, hcol, hinv, mult):
    """NgramHashState.hash_spans in exact Python integers (every span's tokens enter the history first)"""
    pmf = pm.reshape(-1)
    hist = history.reshape(-1).astype(np.int64).copy()
    tm = tokmap.reshape(-1)

    def slot(r, pos):
        return int(pmf[int(rhdr[r, SK.C_TBASE]) + pos // BS]) * BS + pos % BS

    for r in range(T):
        hist[slot(r, int(rpos[r, 0]))] = tm[max(int(ids[r, 0]), 0)]
    out = np.zeros((T, SK.HCOLS), np.int64)
    for r in range(T):
        p = int(rpos[r, 0])
        toks, blocked = [], False
        for s in range(SK.SHIFTS):
            src = int(hist[slot(r, max(p - s, 0))])
            blocked = blocked or p - s < 0 or src == -1
            toks.append(PAD_ID if blocked else src)
        for li in range(2):
            roll = 0
            for s in range(SK.SHIFTS):
                roll ^= toks[s] * int(MULTS[li, s])
                if s:
                    for h in range(HEADS):
                        c = (li * 3 + s - 1) * HEADS + h
                        out[r, c] = roll % int(PRIMES.reshape(-1)[c]) + int(hcol[1, c])
    return [out.astype(np.int32)]
