"""bm25: top-16 BM25 retrieval over an inverted index (term-at-a-time, quantized impacts).

Index (shared by both versions): D = 2^18 documents, V = 32768 terms with Zipf document frequencies,
~31 unique terms per document; every posting is one uint32 = doc << 8 | impact, impact = BM25
(k1 = 0.9, b = 0.4) quantized to 1..255 (the usual impact-quantized index of fast BM25 engines);
postings sorted by doc within a term. Padded for the NKI baseline's static tile shapes: lists with
>= 1024 postings start 1024-aligned and are padded to a multiple of 1024, shorter lists 128-aligned
and padded to a multiple of 128 (padding postings: impact 0, distinct dummy docs >= D).
Query: Q = 8 terms of mid/high frequency. score(d) = sum of the query terms' impacts in d.

baseline (pure NKI): per query term, two nl.fori_loop loops with register bounds read from the index
    (1024-posting blocks for long lists, 128-posting blocks for short ones): DMA a block in with a
    scalar_offset, unpack doc/impact on the Vector engine, scatter-add the impacts into an fp32 score
    array in HBM with dma_compute read-modify-write (vector_offset [128, 1] per instruction: this compiler
    rejects more than one index per partition), then top-16.
gpsimd: one GPSIMD call: core c owns docs [c*32768, (c+1)*32768), streams its slice of every query
    term's postings from HBM with the iDMA (slice bounds precomputed per term: split[t][c]) and
    accumulates uint16 scores in local RAM, then writes them to SBUF ([128, 2048], doc = p*2048 + f).
    Then the same top-16.
top-16 (shared NKI code): 2 x (max8 + nc_match_replace8) per partition, candidates through an HBM
    scratch row, 2 x (max8 + nc_match_replace8) on [1, 2048], nc_n_gather for the doc ids.
"""
import os

import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

import gpsimd_nki as gp

D = 1 << 18
V = 32768
Q = 8
TOPK = 16
F = D // 128                         # scores tile [128, F], doc = p * F + f
SB, B = 1024, 128                    # NKI block sizes (postings)
DOCS_PER_TERM = int(os.environ.get("DPT", "31"))
QSEED = int(os.environ.get("QSEED", "7"))
QDF = tuple(int(x) for x in os.environ.get("QDF", "1000,30000").split(","))   # df range of query terms
OPS = ["op.cpp"]
HBM = ["post"]
BASELINE_SKIP_INPUTS = ["split", "gp_hbm"]


def _index(seed=1234):
    rng = np.random.default_rng(seed)
    n = D * DOCS_PER_TERM
    docs = np.repeat(np.arange(D, dtype=np.int64), DOCS_PER_TERM)
    terms = np.minimum(rng.zipf(1.07, n), V) - 1
    key = terms * D + docs
    uk, tf = np.unique(key, return_counts=True)
    t_of, d_of = uk // D, uk % D
    df = np.bincount(t_of, minlength=V)
    dl = np.bincount(d_of, weights=tf, minlength=D)
    avgdl = dl.mean()
    idf = np.log(1 + (D - df + 0.5) / (df + 0.5))
    k1, b = 0.9, 0.4
    w = idf[t_of] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl[d_of] / avgdl))
    imp = np.clip(np.rint(w / w.max() * 255), 1, 255).astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(df)])
    # padded layout
    post, meta, split = [], np.zeros((V, 4), np.int32), np.zeros((V, 16), np.int32)
    pos = 0
    for t in range(V):
        s, e = starts[t], starts[t + 1]
        n_t = e - s
        al = SB if n_t >= SB else B
        if pos % al:
            gap = al - pos % al
            post.append(D + (np.arange(pos, pos + gap) % SB) << 8)
            pos += gap
        npad = -(-n_t // al) * al if n_t else 0
        packed = (d_of[s:e] << 8) | imp[s:e]
        dummy = (D + (np.arange(pos + n_t, pos + npad) % SB)) << 8
        post.append(packed)
        post.append(dummy)
        if al == SB:
            meta[t] = (pos // SB, (pos + npad) // SB, 0, 0)
        else:
            meta[t] = (0, 0, pos // B, (pos + npad) // B)
        cuts = np.searchsorted(d_of[s:e], np.arange(9) * (D // 8))
        split[t, :9] = pos + cuts
        pos += npad
    tail = -pos % SB + SB
    post.append((D + (np.arange(pos, pos + tail) % SB)) << 8)
    post = np.concatenate(post).astype(np.uint32)
    return post, meta, split, df, starts, d_of, imp


_IDX = None


def make_inputs(rng):
    global _IDX
    if _IDX is None:
        _IDX = _index()
    post, meta, split, df, starts, d_of, imp = _IDX
    qrng = np.random.default_rng(QSEED)
    cand = np.nonzero((df >= QDF[0]) & (df <= QDF[1]))[0]
    q = np.sort(qrng.choice(cand, Q, replace=False)).astype(np.int32)
    return {"post": post, "meta": meta, "split": split, "qterms": q[None, :],
            "gp_hbm": np.zeros((1, 16), np.int32)}


def scores_ref(post, qterms, meta, split):
    s = np.zeros(D, np.int64)
    for t in qterms[0]:
        lo, hi = split[t, 0], split[t, 8]
        w = post[lo:hi].astype(np.int64)
        np.add.at(s, w >> 8, w & 255)
    return s


def reference(post, meta, split, qterms, gp_hbm):
    s = scores_ref(post, qterms, meta, split)
    order = np.lexsort((np.arange(D), -s))[:TOPK]           # score desc, doc asc
    return [s[order].astype(np.float32)[None, :], order.astype(np.uint32)[None, :]]


def check(outs, inputs):
    vals, docs = outs[0][0].astype(np.float64), outs[1][0].astype(np.int64)
    s = scores_ref(inputs["post"], inputs["qterms"], inputs["meta"], inputs["split"])
    want = np.sort(s)[::-1][:TOPK]
    if not np.array_equal(np.sort(vals)[::-1], want):
        return False, f"top-{TOPK} scores {vals.tolist()} != {want.tolist()}"
    if len(set(docs.tolist())) != TOPK or docs.min() < 0 or docs.max() >= D:
        return False, "doc ids not distinct / out of range"
    if not np.array_equal(s[docs], vals):
        return False, "returned docs do not have the returned scores"
    pq = int(sum(inputs["split"][t, 8] - inputs["split"][t, 0] for t in inputs["qterms"][0]))
    exact = np.array_equal(docs, reference(**inputs)[1][0])
    return True, (f"top-{TOPK} of {D} docs correct ({pq} postings scored; "
                  f"{'= reference order incl. ties' if exact else 'tie order differs'})")


PARAMS = {"D": D, "V": V, "Q": Q, "topk": TOPK, "docs_per_term": DOCS_PER_TERM, "query_df_range": QDF}


# ------------------------------------------------------------------ shared NKI top-16
def _top16(S, ):
    """S: [128, F] float32 SBUF scores (doc = p*F + f) -> (vals [1,16] f32, docs [1,16] uint32) SBUF."""
    v = nl.ndarray((128, 16), dtype=nl.float32, buffer=nl.sbuf)
    ix = nl.ndarray((128, 16), dtype=nl.uint32, buffer=nl.sbuf)
    for r in range(2):
        nisa.max8(dst=v[:, 8 * r:8 * r + 8], src=S)
        nisa.nc_match_replace8(dst=S, data=S, vals=v[:, 8 * r:8 * r + 8], imm=-1.0, dst_idx=ix[:, 8 * r:8 * r + 8])
    base = nl.ndarray((128, 16), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.iota(dst=base, pattern=[[0, 16]], offset=0, channel_multiplier=F)
    cdoc = nl.ndarray((128, 16), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=cdoc, data1=base, data2=ix, op=nl.add)
    hv = nl.ndarray((128, 16), dtype=nl.float32, buffer=nl.hbm)
    hd = nl.ndarray((128, 16), dtype=nl.uint32, buffer=nl.hbm)
    nisa.dma_copy(dst=hv, src=v)
    nisa.dma_copy(dst=hd, src=cdoc)
    rv = nl.ndarray((1, 2048), dtype=nl.float32, buffer=nl.sbuf)
    rd = nl.ndarray((1, 2048), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rv, src=hv.reshape((1, 2048)))
    nisa.dma_copy(dst=rd, src=hd.reshape((1, 2048)))
    tv = nl.ndarray((1, 16), dtype=nl.float32, buffer=nl.sbuf)
    tix = nl.ndarray((1, 16), dtype=nl.uint32, buffer=nl.sbuf)
    for r in range(2):
        nisa.max8(dst=tv[:, 8 * r:8 * r + 8], src=rv)
        nisa.nc_match_replace8(dst=rv, data=rv, vals=tv[:, 8 * r:8 * r + 8], imm=-1.0, dst_idx=tix[:, 8 * r:8 * r + 8])
    td = nl.ndarray((1, 16), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.nc_n_gather(dst=td, data=rd, indices=tix)
    return tv, td


def _outputs(tv, td):
    ov = nl.ndarray((1, TOPK), dtype=nl.float32, buffer=nl.shared_hbm)
    od = nl.ndarray((1, TOPK), dtype=nl.uint32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=ov, src=tv)
    nisa.dma_copy(dst=od, src=td)
    return ov, od


@nki.jit
def baseline_kernel(post, meta, qterms):
    NP = post.shape[0]
    scores = nl.ndarray((D + SB,), dtype=nl.float32, buffer=nl.hbm)
    zero = nl.ndarray((128, (D + SB) // 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zero, value=0.0)
    nisa.dma_copy(dst=scores.reshape((128, (D + SB) // 128)), src=zero)
    q_sb = nl.ndarray((Q, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_sb, src=qterms.reshape((Q, 1)))
    m_sb = nl.ndarray((Q, 4), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=m_sb, src=meta.ap(pattern=[[4, Q], [1, 4]], offset=0, vector_offset=q_sb, indirect_dim=0))
    post_sb = post.reshape((NP // SB, SB))
    post_b = post.reshape((NP // B, B))
    for q in range(Q):
        regs = []
        for k in range(4):
            r = nisa.register_alloc()
            nisa.register_load(r, m_sb[q:q + 1, k:k + 1])
            regs.append(r)

        def body_sb(i):
            ii = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.register_store(ii, i)
            w = nl.ndarray((128, 8), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.dma_copy(dst=w, src=post_sb.ap(pattern=[[8, 128], [1, 8]], offset=0, scalar_offset=ii, indirect_dim=0))
            doc = nl.ndarray((128, 8), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=doc, data=w, op0=nl.right_shift, operand0=8)
            imp = nl.ndarray((128, 8), dtype=nl.uint32, buffer=nl.sbuf)     # DMA compute casts to fp32
            nisa.tensor_scalar(dst=imp, data=w, op0=nl.bitwise_and, operand0=255)
            for m in range(8):           # one index per partition per dma_compute (compiler requirement)
                dst = scores.ap(pattern=[[1, 128], [1, 1]], offset=0, vector_offset=doc[:, m:m + 1], indirect_dim=0)
                nisa.dma_compute(dst=dst, srcs=[dst, imp[:, m:m + 1]], reduce_op=nl.add, unique_indices=True)

        def body_b(i):
            ii = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.register_store(ii, i)
            w = nl.ndarray((128, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.dma_copy(dst=w, src=post_b.ap(pattern=[[1, 128], [1, 1]], offset=0, scalar_offset=ii, indirect_dim=0))
            doc = nl.ndarray((128, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=doc, data=w, op0=nl.right_shift, operand0=8)
            imp = nl.ndarray((128, 1), dtype=nl.uint32, buffer=nl.sbuf)     # DMA compute casts to fp32
            nisa.tensor_scalar(dst=imp, data=w, op0=nl.bitwise_and, operand0=255)
            dst = scores.ap(pattern=[[1, B], [1, 1]], offset=0, vector_offset=doc, indirect_dim=0)
            nisa.dma_compute(dst=dst, srcs=[dst, imp], reduce_op=nl.add, unique_indices=True)

        nl.fori_loop(regs[0], regs[1], body_sb)
        nl.fori_loop(regs[2], regs[3], body_b)
    S = nl.ndarray((128, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=S, src=scores[0:D].reshape((128, F)))
    return _outputs(*_top16(S))


@nki.jit
def gpsimd_kernel(post, meta, split, qterms, gp_hbm):
    q_sb = nl.ndarray((Q, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_sb, src=qterms.reshape((Q, 1)))
    sp = nl.ndarray((Q, 16), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sp, src=split.ap(pattern=[[16, Q], [1, 16]], offset=0, vector_offset=q_sb, indirect_dim=0))
    hb = nl.ndarray((1, 16), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=hb, src=gp_hbm)
    acc = nl.ndarray((128, F), dtype=nl.uint16, buffer=nl.sbuf)
    gp.call("bm25_accum", ins=[sp, hb], outs=[acc])
    S = nl.ndarray((128, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=S, src=acc)
    return _outputs(*_top16(S))
