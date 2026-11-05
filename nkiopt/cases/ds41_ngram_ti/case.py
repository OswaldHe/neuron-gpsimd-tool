"""ds41_ngram_ti: ds41_ngram with the block-table (span table `pm`) lookups of the window gathers moved on chip.

LOOKUP=dma      the port's lookups: one indirect DMA (software DGE on GPSIMD) per shift, 4 per tile
LOOKUP=ngather  trn2-compatible: `pm` broadcast into SBUF once, nc_n_gather (a GPSIMD built-in) for all 4 shifts
LOOKUP=ti       trn3 only: `pm` broadcast into SBUF once, tensor indirection (NeuronCore-v4 indirect SBUF
                addressing) inside one Vector tensor_tensor; offsets are shared by each group of 16
                partitions, so the instruction gathers the 16 rows' entries and an identity mask keeps the
                row's own one (then one tensor_reduce)
TIENG=vector|gpsimd  engine of the TI tensor_tensor (the reduce is a Vector instruction)
BASE_LOOKUP     the same choice for the pure-NKI baseline (default dma: the port as written)

ds41_ngram: DeepSeek-V4.1-Flash engram n-gram hashing (both engram layers' 48 row ids per token).

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
LOOKUP = os.environ.get("LOOKUP", "dma")
BASE_LOOKUP = os.environ.get("BASE_LOOKUP", "dma")
TIENG = {"vector": nisa.engine.vector, "gpsimd": nisa.engine.gpsimd}[os.environ.get("TIENG", "vector")]
WBLK = -(-CTX // BS) + 1
HW = H + WBLK
NPM = B * HW                               # span-table entries (all spans)
PB = min(PT, (T + 15) // 16 * 16)          # partitions of the broadcast span table
MERGE = int(os.environ.get("MERGE", "0")) == 1   # form the window in the GPSIMD op (reads before the write)
NS = 4                                     # shift 0 (the write slot) and the 3 window shifts
_COMMON = ["rpos", "rhdr", "ids", "pm", "history", "tokmap", "prod", "hcol", "hinv", "mult"]
INPUTS = {"baseline": _COMMON + {"ti": ["eye", "sbase"], "ngather": ["sbase"]}.get(BASE_LOOKUP, []),
          "gpsimd": _COMMON + ["eye", "sbase"]}
assert B == 1 or B * L <= PT, "span tables are loaded for the first tile's spans only"
OPS = ["op.cpp"]
CFLAGS = [f"-DNG_NT={NT}", f"-DNG_T={T}"]  # (+ -DNG_MERGE -DNG_PAD below)
PARAMS = {"spans": B, "rows_per_span": L, "T": T, "hash_cols": SK.HCOLS, "lookup": LOOKUP,
          "base_lookup": BASE_LOOKUP, "ti_engine": os.environ.get("TIENG", "vector"), "merge": MERGE}


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
def baseline_kernel_dma(rpos, rhdr, ids, pm, history, tokmap, prod, hcol, hinv, mult):
    out = nl.ndarray((T, SK.HCOLS), dtype=nl.int32, buffer=nl.shared_hbm)
    pad_id = PAD_ID
    for r0, nr in TILES:
        row_pos, ws, hdr, ids_t = _rows(r0, nr, rpos, rhdr, ids, pm)
        hv = SK._hash_rows(nr, row_pos, ws, hdr, ids_t, pm, history, tokmap, prod, hcol, hinv, pad_id, BS, SK._log2(BS))
        nisa.dma_copy(dst=out[r0:r0 + nr, :], src=hv[0:nr, :])
    return out


baseline_kernel = baseline_kernel_dma


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


# ----------------------------------------------------------------------------- on-chip span-table lookups
def _onchip_tables(pm, eye, sbase):
    """each row's span table row in its partition (TI and nc_n_gather read within a partition: row q is span
    q // L's, the port's layout, see `_span_cols`) and the identity mask"""
    pm_b = nl.ndarray((PB, HW), dtype=nl.int32, buffer=nl.sbuf)
    if PB > T:
        nisa.memset(dst=pm_b, value=0)
    if B == 1:
        nisa.dma_copy(dst=pm_b, src=pm.ap(pattern=[[0, PB], [1, HW]], offset=0))
    else:
        for b in range(B):     # hardware DGE: issued by the Sync / Scalar queues, not by GPSIMD
            nisa.dma_copy(dst=pm_b[b * L:(b + 1) * L, :], src=pm.ap(pattern=[[0, L], [1, HW]], offset=b * HW),
                          dge_mode=nisa.dge_mode.hwdge)
    eye_sb = None
    if eye is not None:
        eye_sb = nl.ndarray((PB, 16 * NS), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=eye_sb, src=eye[0:PB, :])
    sbase_sb = nl.ndarray((PB, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sbase_sb, src=sbase[0:PB, :])
    return pm_b, eye_sb, sbase_sb


def _rows_pad(r0, nr, rpos, rhdr, ids):
    """`_rows` without the write-slot lookup; partitions nr..PP (PP = nr rounded up to 16) hold position 0, span 0"""
    PP = (nr + 15) // 16 * 16
    row_pos = SK._i((PT, 1))
    hdr = SK._i((PT, H))
    if PP > nr:
        nisa.memset(row_pos[0:PP, :], 0)
        nisa.memset(hdr[0:PP, :], 0)
    nisa.dma_copy(dst=row_pos[0:nr, :], src=rpos[r0:r0 + nr, :])
    nisa.dma_copy(dst=hdr[0:nr, :], src=rhdr[r0:r0 + nr, :])
    ids_raw = SK._i((PT, 1))
    nisa.dma_copy(dst=ids_raw[0:nr, :], src=ids[r0:r0 + nr, :])
    ids_t = SK._i((PT, 1))
    SK._ts(ids_t[0:nr, :], ids_raw[0:nr, :], nl.maximum, 0)
    return PP, row_pos, hdr, ids_t


def _slots_onchip(mode, PP, row_pos, hdr, pm_b, eye_sb, pm=None, sbase=None):
    """the cache slots of positions pos - s, s = 0..3 (`_slot` for all four shifts at once), and pos - s < 0"""
    _i, _ts, _tt = SK._i, SK._ts, SK._tt
    pos = _i((PT, NS))
    for s in range(NS):
        _ts(pos[0:PP, s:s + 1], row_pos[0:PP, :], nl.add, -s)
    neg = _i((PT, NS))
    _ts(neg[0:PP, :], pos[0:PP, :], nl.less, 0)
    _ts(pos[0:PP, :], pos[0:PP, :], nl.maximum, 0)
    blk = _i((PT, NS))
    _ts(blk[0:PP, :], pos[0:PP, :], nl.right_shift, SK._log2(BS))
    idx = _i((PT, NS))
    SK._tsc(idx[0:PP, :], blk[0:PP, :], nl.add, hdr[:, SK.C_TBASE:SK.C_TBASE + 1], PP)
    if mode != "dma":     # offset within the row's own span table row
        SK._tsc(idx[0:PP, :], idx[0:PP, :], nl.subtract, sbase[0:PP, :], PP)
    tab = _i((PT, NS))
    if mode == "dma":       # the port's lookup, but the 4 shifts' DMAs are independent of each other
        for s in range(NS):
            nisa.dma_copy(dst=tab[0:PP, s:s + 1], src=pm.ap(pattern=[[1, PP], [1, 1]], vector_offset=idx[0:PP, s:s + 1],
                                                           indirect_dim=0))
    elif mode == "ti":
        # offset i of a 16-partition group g comes from idx[16 g + i % 16, i // 16]: the instruction reads, for
        # every partition, the entries of all 16 rows of its group; the identity mask keeps the row's own one
        idx16 = nl.ndarray((PP, NS), dtype=nl.uint16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=idx16, src=idx[0:PP, :])
        g = nl.ndarray((PP, 16 * NS), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=g, data1=pm_b.indirect(idx16, num_elem=16 * NS), data2=eye_sb, op=nl.multiply,
                           engine=TIENG)
        nisa.tensor_reduce(dst=tab[0:PP, :], op=nl.add, data=g.reshape((PP, NS, 16)), axis=(2,))
    else:
        idx32 = nl.ndarray((PP, NS), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=idx32, src=idx[0:PP, :])
        nisa.nc_n_gather(dst=tab[0:PP, :], data=pm_b, indices=idx32)
    off = _i((PT, NS))
    _ts(off[0:PP, :], pos[0:PP, :], nl.bitwise_and, BS - 1)
    sl = _i((PT, NS))
    _ts(sl[0:PP, :], tab[0:PP, :], nl.multiply, BS)
    _tt(sl[0:PP, :], sl[0:PP, :], off[0:PP, :], nl.add)
    return sl, neg


def _window_onchip(mode, nr, PP, row_pos, hdr, ids_t, pm_b, eye_sb, history, tokmap, pad_id, sbase=None):
    """`_window_tokens` with the span-table lookups on chip; the history accesses stay indirect DMAs"""
    _i, _ts, _tt, DEAD = SK._i, SK._ts, SK._tt, SK.DEAD
    c = SK._gather1(tokmap, ids_t, nr)          # first: it needs only the ids (the GPSIMD DMA queue is in order)
    sl, neg = _slots_onchip(mode, PP, row_pos, hdr, pm_b, eye_sb, sbase=sbase)
    nisa.dma_copy(dst=history.ap(pattern=[[1, nr], [1, 1]], vector_offset=sl[0:nr, 0:1], indirect_dim=0),
                  src=c[0:nr, 0:1])
    toks = [c]
    blocked = _i((PT, 1))
    nisa.memset(blocked[0:nr, :], 0)
    for s in range(1, SK.SHIFTS):
        src = _i((PT, 1))
        nisa.dma_copy(dst=src[0:nr, 0:1], src=history.ap(pattern=[[1, nr], [1, 1]], vector_offset=sl[0:nr, s:s + 1],
                                                          indirect_dim=0))
        dead = _i((PT, 1))
        _ts(dead[0:nr, :], src[0:nr, :], nl.equal, DEAD)
        _tt(blocked[0:nr, :], blocked[0:nr, :], neg[0:nr, s:s + 1], nl.maximum)
        _tt(blocked[0:nr, :], blocked[0:nr, :], dead[0:nr, :], nl.maximum)
        d = _i((PT, 1))
        _ts(d[0:nr, :], src[0:nr, :], nl.multiply, -1, nl.add, pad_id)
        _tt(d[0:nr, :], d[0:nr, :], blocked[0:nr, :], nl.multiply)
        tok = _i((PT, 1))
        _tt(tok[0:nr, :], src[0:nr, :], d[0:nr, :], nl.add)
        toks.append(tok)
    return toks


def _hash_from_toks(T, toks, prod, hcol, hinv):
    """the second half of the port's `step_kernel._hash_rows`, verbatim: products, XOR, modulo"""
    _i, _f, _ts, _tt, GP = SK._i, SK._f, SK._ts, SK._tt, SK.GP
    HASH_LAYERS, SHIFTS, HCOLS, NGRAMS, HEADS = SK.HASH_LAYERS, SK.SHIFTS, SK.HCOLS, SK.NGRAMS, SK.HEADS
    # the products of every shift's token: prod[c] = [layer][shift][hi, lo]
    pw = 2 * SHIFTS
    prods = []
    for s in range(SHIFTS):
        pr = _i((PT, HASH_LAYERS * pw))
        nisa.dma_copy(dst=pr[0:T, :], src=prod.ap(pattern=[[HASH_LAYERS * pw, T], [1, HASH_LAYERS * pw]],
                                                   vector_offset=toks[s][0:T, 0:1], indirect_dim=0))
        prods.append(pr)
    # the rolled (hi, lo) of every (layer, n-gram) broadcast over the heads: (T, HCOLS)
    vhi = _i((PT, HCOLS))
    vlo = _i((PT, HCOLS))
    for l in range(HASH_LAYERS):
        rh = _i((PT, 1))
        rl = _i((PT, 1))
        b0 = l * pw
        nisa.tensor_copy(dst=rh[0:T, :], src=prods[0][0:T, b0 : b0 + 1], engine=GP)
        nisa.tensor_copy(dst=rl[0:T, :], src=prods[0][0:T, b0 + 1 : b0 + 2], engine=GP)
        for n in range(NGRAMS):
            s = n + 1
            _tt(rh[0:T, :], rh[0:T, :], prods[s][0:T, b0 + 2 * s : b0 + 2 * s + 1], nl.bitwise_xor)
            _tt(rl[0:T, :], rl[0:T, :], prods[s][0:T, b0 + 2 * s + 1 : b0 + 2 * s + 2], nl.bitwise_xor)
            c0 = (l * NGRAMS + n) * HEADS
            for h in range(HEADS):
                nisa.tensor_copy(dst=vhi[0:T, c0 + h : c0 + h + 1], src=rh[0:T, :], engine=GP)
                nisa.tensor_copy(dst=vlo[0:T, c0 + h : c0 + h + 1], src=rl[0:T, :], engine=GP)
    primes = nl.ndarray((PT, HCOLS), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=primes[0:T, :], src=hcol.ap(pattern=[[0, T], [1, HCOLS]], offset=0))
    hoff = nl.ndarray((PT, HCOLS), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=hoff[0:T, :], src=hcol.ap(pattern=[[0, T], [1, HCOLS]], offset=HCOLS))
    inv = _f((PT, HCOLS))
    nisa.dma_copy(dst=inv[0:T, :], src=hinv.ap(pattern=[[0, T], [1, HCOLS]], offset=0))

    def chunk(k):
        """bits [62 - 7k .. 56 - 7k] of the 63-bit value."""
        ch = _i((PT, HCOLS))
        hi_bit = 62 - 7 * k
        lo_bit = hi_bit - 6
        if lo_bit >= 32:
            _ts(ch[0:T, :], vhi[0:T, :], nl.right_shift, lo_bit - 32, nl.bitwise_and, 127)
        elif hi_bit < 32:
            _ts(ch[0:T, :], vlo[0:T, :], nl.right_shift, lo_bit, nl.bitwise_and, 127)
        else:            # straddles the words: hi's low bits above lo's top bits
            nhi = hi_bit - 31
            a = _i((PT, HCOLS))
            _ts(a[0:T, :], vhi[0:T, :], nl.bitwise_and, (1 << nhi) - 1, nl.left_shift, 32 - lo_bit)
            _ts(ch[0:T, :], vlo[0:T, :], nl.right_shift, lo_bit, nl.bitwise_and, (1 << (32 - lo_bit)) - 1)
            _tt(ch[0:T, :], ch[0:T, :], a[0:T, :], nl.bitwise_or)
        return ch

    r = chunk(0)
    for k in range(1, 9):
        ch = chunk(k)
        x = _i((PT, HCOLS))
        _ts(x[0:T, :], r[0:T, :], nl.left_shift, 7)
        _tt(x[0:T, :], x[0:T, :], ch[0:T, :], nl.bitwise_or)
        xf = _f((PT, HCOLS))
        nisa.tensor_copy(dst=xf[0:T, :], src=x[0:T, :])
        nisa.tensor_tensor(dst=xf[0:T, :], data1=xf[0:T, :], data2=inv[0:T, :], op=nl.multiply)
        q = _i((PT, HCOLS))
        nisa.tensor_copy(dst=q[0:T, :], src=xf[0:T, :])
        _tt(q[0:T, :], q[0:T, :], primes[0:T, :], nl.multiply)
        r = _i((PT, HCOLS))
        _tt(r[0:T, :], x[0:T, :], q[0:T, :], nl.subtract)
        # r in [-p, 2p): one p up or down
        m = _i((PT, HCOLS))
        _ts(m[0:T, :], r[0:T, :], nl.less, 0)
        _tt(m[0:T, :], m[0:T, :], primes[0:T, :], nl.multiply)
        _tt(r[0:T, :], r[0:T, :], m[0:T, :], nl.add)
        m2 = _i((PT, HCOLS))
        _tt(m2[0:T, :], r[0:T, :], primes[0:T, :], nl.greater_equal)
        _tt(m2[0:T, :], m2[0:T, :], primes[0:T, :], nl.multiply)
        _tt(r[0:T, :], r[0:T, :], m2[0:T, :], nl.subtract)
    _tt(r[0:T, :], r[0:T, :], hoff[0:T, :], nl.add)
    return r


def _baseline_onchip(rpos, rhdr, ids, pm, history, tokmap, prod, hcol, hinv, eye, sbase):
    out = nl.ndarray((T, SK.HCOLS), dtype=nl.int32, buffer=nl.shared_hbm)
    pm_b, eye_sb, sbase_sb = _onchip_tables(pm, eye, sbase)
    for r0, nr in TILES:
        PP, row_pos, hdr, ids_t = _rows_pad(r0, nr, rpos, rhdr, ids)
        toks = _window_onchip(BASE_LOOKUP, nr, PP, row_pos, hdr, ids_t, pm_b, eye_sb, history, tokmap, PAD_ID,
                              sbase_sb)
        hv = _hash_from_toks(nr, toks, prod, hcol, hinv)
        nisa.dma_copy(dst=out[r0:r0 + nr, :], src=hv[0:nr, :])
    return out


if BASE_LOOKUP == "ti":
    @nki.jit
    def baseline_kernel(rpos, rhdr, ids, pm, history, tokmap, prod, hcol, hinv, mult, eye, sbase):
        return _baseline_onchip(rpos, rhdr, ids, pm, history, tokmap, prod, hcol, hinv, eye, sbase)
elif BASE_LOOKUP == "ngather":
    @nki.jit
    def baseline_kernel(rpos, rhdr, ids, pm, history, tokmap, prod, hcol, hinv, mult, sbase):
        return _baseline_onchip(rpos, rhdr, ids, pm, history, tokmap, prod, hcol, hinv, None, sbase)


def _gpsimd_merge(rpos, rhdr, ids, pm, history, tokmap, hcol, mult, eye, sbase):
    """MERGE=1: the history reads no longer wait for this step's write; the op forms the windows. The write
    follows the call (its source is the call's input, so it cannot land between the call's markers)."""
    i32 = nl.int32
    cw_all = nl.ndarray((PT, NT * 2), dtype=i32, buffer=nl.sbuf)       # c, ws
    own_all = nl.ndarray((PT, NT * 7), dtype=i32, buffer=nl.sbuf)      # rd x3, old x3, pos
    nisa.memset(dst=cw_all, value=0)
    nisa.memset(dst=own_all, value=0)
    pm_b = eye_sb = sbase_sb = None
    if LOOKUP != "dma":
        pm_b, eye_sb, sbase_sb = _onchip_tables(pm, eye if LOOKUP == "ti" else None, sbase)
    slots = []
    for r0, nr in TILES:
        i = r0 // PT
        PP, row_pos, hdr, ids_t = _rows_pad(r0, nr, rpos, rhdr, ids)
        nisa.dma_copy(dst=cw_all[0:nr, 2 * i:2 * i + 1],
                      src=tokmap.ap(pattern=[[1, nr], [1, 1]], vector_offset=ids_t[0:nr, 0:1], indirect_dim=0))
        sl, neg = _slots_onchip(LOOKUP, PP, row_pos, hdr, pm_b, eye_sb, pm, sbase_sb)
        for s in range(1, SK.SHIFTS):      # the old values: before this step's write
            nisa.dma_copy(dst=own_all[0:nr, 7 * i + 2 + s:7 * i + 3 + s],
                          src=history.ap(pattern=[[1, nr], [1, 1]], vector_offset=sl[0:nr, s:s + 1], indirect_dim=0))
        nisa.tensor_copy(dst=cw_all[0:nr, 2 * i + 1:2 * i + 2], src=sl[0:nr, 0:1])
        nisa.tensor_copy(dst=own_all[0:nr, 7 * i:7 * i + 3], src=sl[0:nr, 1:4])
        nisa.tensor_copy(dst=own_all[0:nr, 7 * i + 6:7 * i + 7], src=row_pos[0:nr, :])
        slots.append(sl)
    hcol_sb = nl.ndarray((4, SK.HCOLS), dtype=i32, buffer=nl.sbuf)
    nisa.dma_copy(dst=hcol_sb, src=hcol)
    mult_sb = nl.ndarray((1, 16), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.dma_copy(dst=mult_sb, src=mult)
    hv = nl.ndarray((PT, NT * SK.HCOLS), dtype=i32, buffer=nl.sbuf)
    cout = nl.ndarray((PT, NT), dtype=i32, buffer=nl.sbuf)                 # the op's copy of c
    gp.call("ngram_hash", ins=[cw_all, own_all, hcol_sb, mult_sb], outs=[hv, cout])
    out = nl.ndarray((T, SK.HCOLS), dtype=nl.int32, buffer=nl.shared_hbm)
    for r0, nr in TILES:
        i = r0 // PT
        nisa.dma_copy(dst=out[r0:r0 + nr, :], src=hv[0:nr, i * SK.HCOLS:(i + 1) * SK.HCOLS])
        nisa.dma_copy(dst=history.ap(pattern=[[1, nr], [1, 1]], vector_offset=slots[i][0:nr, 0:1], indirect_dim=0),
                      src=cout[0:nr, i:i + 1])
    return out


@nki.jit
def gpsimd_kernel(rpos, rhdr, ids, pm, history, tokmap, prod, hcol, hinv, mult, eye, sbase):
    if MERGE:
        return _gpsimd_merge(rpos, rhdr, ids, pm, history, tokmap, hcol, mult, eye, sbase)
    i32 = nl.int32
    toks_all = nl.ndarray((PT, NT * SK.SHIFTS), dtype=i32, buffer=nl.sbuf)
    nisa.memset(dst=toks_all, value=0)
    if LOOKUP != "dma":
        pm_b, eye_sb, sbase_sb = _onchip_tables(pm, eye if LOOKUP == "ti" else None, sbase)
    for r0, nr in TILES:
        if LOOKUP == "dma":
            row_pos, ws, hdr, ids_t = _rows(r0, nr, rpos, rhdr, ids, pm)
            toks = _window_tokens(nr, row_pos, ws, hdr, ids_t, pm, history, tokmap, PAD_ID, BS, SK._log2(BS))
        else:
            PP, row_pos, hdr, ids_t = _rows_pad(r0, nr, rpos, rhdr, ids)
            toks = _window_onchip(LOOKUP, nr, PP, row_pos, hdr, ids_t, pm_b, eye_sb, history, tokmap, PAD_ID, sbase_sb)
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
if MERGE:
    CFLAGS += ["-DNG_MERGE", f"-DNG_PAD={PAD_ID}"]
# identity mask of the TI lookup: eye[p, 16 s + j] = (j == p % 16)
# each row's span table base in pm (row q is span q // L's; B == 1: one span)
SBASE = (((np.arange(PT) // L) * HW) * (np.arange(PT) < B * L)).astype(np.int32).reshape(PT, 1)
EYE = np.tile((np.arange(16)[None, :] == (np.arange(PT) % 16)[:, None]).astype(np.int32), (1, NS))


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
            "tokmap": _TOKMAP.reshape(-1, 1), "prod": prod, "hcol": hcol, "hinv": hinv, "mult": mult,
            "eye": EYE, "sbase": SBASE}


def reference(rpos, rhdr, ids, pm, history, tokmap, prod, hcol, hinv, mult, eye=None, sbase=None):
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
