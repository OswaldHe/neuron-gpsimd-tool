"""ann: graph-based approximate nearest-neighbour search (MIPS, top-10) vs exact NKI brute force.

Data (prep.py, cached in data/): N = 2^19 int8 vectors, d = 128; proximity graph with R = 32 out-edges;
PQ (M = 16 subspaces x 8 dims, 256 centroids) whose codes are stored inline in the node records
(record = 32 neighbour ids + 32 x 16 code bytes = 640 B); B = 8 queries.

baseline (pure NKI): exact brute force. X^T [128, N] bf16 (int8 values, exact), 64 matmuls per
    8192-row super-chunk with the X block stationary and the 8 queries moving -> scores [128 rows, 8]
    (all 128 partitions busy for the selection), per query 2 x (max8 + nc_match_replace8) per
    partition per super-chunk, then per-partition and cross-partition (HBM bounce) top-16.
gpsimd: NKI computes each query's PQ lookup table (one matmul against the block-diagonal codebook,
    exact integers) and then one GPSIMD call: core b runs query b's beam search (beam L, visited hash
    set, one 640-byte iDMA read per expanded node, 16 table lookups per neighbour), then re-ranks the
    final beam with exact int8 dot products (vectors read with the iDMA) and returns the top-10.
Both are exact integer arithmetic: the GPSIMD output is compared bit-exactly against a Python model of
the same search (ids, scores, hop and visit counts); recall@10 is reported against the exact top-10.
"""
import os

import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl
import ml_dtypes

import gpsimd_nki as gp

N, D, R, M, KS, B, TOPK = 1 << 19, 128, 32, 16, 256, 8, 10
L = int(os.environ.get("L", "64"))
HOP_LIMIT = 384
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OPS = ["op.cpp"]
CFLAGS = [f"-DANN_L={L}", f"-DANN_HOPS={HOP_LIMIT}"]
HBM = ["idx"]
INPUTS = {"baseline": ["xt", "qt"], "gpsimd": ["qt", "cmat", "q8", "entry", "idx", "gp_hbm"]}
PARAMS = {"N": N, "d": D, "R": R, "pq": f"{M}x{KS}", "queries": B, "beam_L": L, "metric": "inner product"}
SC = 8192                     # rows per super-chunk (64 blocks of 128)


def _load():
    g = {k: np.load(f"{DATA}/{k}.npy", mmap_mode="r") for k in ("X", "Q", "books", "codes", "nbr", "gt", "entry")}
    return g


def make_inputs(rng):
    g = _load()
    X, Qv, books = np.asarray(g["X"]), np.asarray(g["Q"]), np.asarray(g["books"])
    cmat = np.zeros((D, M * KS), np.float32)                 # block-diagonal codebook: q @ cmat = LUT
    for m in range(M):
        cmat[(D // M) * m:(D // M) * (m + 1), KS * m:KS * m + KS] = books[m].T
    e = int(g["entry"][0])
    entry = np.zeros((1, 16), np.int32)
    entry[0, 0] = e
    entry[0, 1:1 + M // 4] = np.asarray(g["codes"])[e].view(np.int32)
    return {"xt": np.ascontiguousarray(X.T).astype(ml_dtypes.bfloat16),
            "qt": np.ascontiguousarray(Qv.T).astype(ml_dtypes.bfloat16),
            "cmat": cmat.astype(ml_dtypes.bfloat16),
            "q8": Qv.copy(),
            "entry": entry,
            "idx": np.load(f"{DATA}/idx.npy"),
            "gp_hbm": np.zeros((1, 16), np.int32)}


# ------------------------------------------------------------------ Python model of the GPSIMD search
def search_ref(b):
    g = _load()
    X, Qv, books, codes, nbr = g["X"], np.asarray(g["Q"]), np.asarray(g["books"]), g["codes"], g["nbr"]
    q = Qv[b].astype(np.int64)
    S = D // M
    lut = np.stack([books[m].astype(np.int64) @ q[S * m:S * m + S] for m in range(M)])       # [M, KS]
    pq = lambda c: -int(lut[np.arange(M), c].sum())                                          # distance
    e = int(g["entry"][0])
    beam = [(pq(np.asarray(codes[e])), e, False)]
    seen = {e}
    hops = 0
    while hops < HOP_LIMIT:
        k = next((i for i, x in enumerate(beam) if not x[2]), None)
        if k is None:
            break
        d0, u, _ = beam[k]
        beam[k] = (d0, u, True)
        hops += 1
        for v in nbr[u].tolist():
            if v in seen:
                continue
            seen.add(v)
            d = pq(np.asarray(codes[v]))
            if len(beam) < L or (d, v) < beam[-1][:2]:
                pos = next((i for i, x in enumerate(beam) if x[:2] > (d, v)), len(beam))
                beam.insert(pos, (d, v, False))
                del beam[L:]
    ids = np.array([x[1] for x in beam], np.int64)
    s = X[ids].astype(np.int64) @ q
    order = np.lexsort((ids, -s))[:TOPK]
    return ids[order], s[order], hops, len(seen)


def reference(**inputs):
    ids, sc = np.zeros((B, 16), np.int32), np.zeros((B, 16), np.int32)
    for b in range(B):
        i, s, h, v = search_ref(b)
        ids[b, :TOPK], sc[b, :TOPK] = i, s
        ids[b, 10], ids[b, 11] = h, v
    return [ids, sc]


def check(outs, inputs, which):
    g = _load()
    gt = np.asarray(g["gt"])
    X, Qv = g["X"], np.asarray(g["Q"])
    ids, sc = outs[0][:, :TOPK].astype(np.int64), outs[1][:, :TOPK].astype(np.int64)
    exact = np.stack([X[gt[b]].astype(np.int64) @ Qv[b].astype(np.int64) for b in range(B)])
    if which == "baseline":                      # exact search: scores must equal the exact top-10 scores
        if not np.array_equal(sc, exact):
            return False, "top-10 scores differ from exact brute force"
        for b in range(B):
            if not np.array_equal(X[ids[b]].astype(np.int64) @ Qv[b].astype(np.int64), sc[b]):
                return False, f"query {b}: returned ids do not have the returned scores"
        return True, f"exact top-{TOPK} for {B} queries over {N} vectors (recall@10 = 1.000)"
    ref = reference(**inputs)
    if not (np.array_equal(outs[0][:, :12], ref[0][:, :12]) and np.array_equal(outs[1][:, :TOPK], ref[1][:, :TOPK])):
        return False, "GPSIMD search differs from the Python model of the same search"
    rec = np.mean([len(set(ids[b]) & set(gt[b].tolist())) / TOPK for b in range(B)])
    hops = outs[0][:, 10]
    return True, (f"= Python model bit-exact (ids, scores, hops, visits); recall@10 vs exact = {rec:.3f}; "
                  f"hops/query {hops.min()}-{hops.max()}, visited {outs[0][:, 11].min()}-{outs[0][:, 11].max()}")


# ------------------------------------------------------------------ kernels
@nki.jit
def baseline_kernel(xt, qt):
    NS = N // SC
    q_sb = nl.ndarray((D, B), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_sb, src=qt)
    cv = nl.ndarray((128, B * NS * 16), dtype=nl.float32, buffer=nl.sbuf)    # per query: NS x 16 candidates
    cj = nl.ndarray((128, B * NS * 16), dtype=nl.uint32, buffer=nl.sbuf)     # block index within N/128
    for s in range(NS):                      # unrolled: offsets are compile-time integers
        x = nl.ndarray((D, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=x, src=xt[:, nl.ds(s * SC, SC)])
        ps = nl.ndarray((128, 64 * B), dtype=nl.float32, buffer=nl.psum)
        for j in range(64):
            nisa.nc_matmul(dst=ps[:, j * B:(j + 1) * B], stationary=x[:, j * 128:(j + 1) * 128], moving=q_sb)
        sc = nl.ndarray((128, 64 * B), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=sc, src=ps, engine=nisa.scalar_engine)
        for b in range(B):
            view = sc.ap(pattern=[[64 * B, 128], [B, 64]], offset=b)
            col = b * NS * 16 + s * 16
            ix = nl.ndarray((128, 16), dtype=nl.uint32, buffer=nl.sbuf)
            for r in range(2):
                nisa.max8(dst=cv[:, nl.ds(col + 8 * r, 8)], src=view)
                nisa.nc_match_replace8(dst=view, data=view, vals=cv[:, nl.ds(col + 8 * r, 8)], imm=-1e9,
                                       dst_idx=ix[:, 8 * r:8 * r + 8])
            nisa.tensor_scalar(dst=cj[:, nl.ds(col, 16)], data=ix, op0=nl.add, operand0=s * 64)
    # per query: top-16 per partition, then across partitions
    pv = nl.ndarray((128, B * 16), dtype=nl.float32, buffer=nl.sbuf)
    pid = nl.ndarray((128, B * 16), dtype=nl.uint32, buffer=nl.sbuf)
    pidx = nl.ndarray((128, 16), dtype=nl.uint32, buffer=nl.sbuf)
    part = nl.ndarray((128, 16), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.iota(dst=part, pattern=[[0, 16]], offset=0, channel_multiplier=1)
    for b in range(B):
        cvb = cv[:, b * NS * 16:(b + 1) * NS * 16]
        for r in range(2):
            nisa.max8(dst=pv[:, b * 16 + 8 * r:b * 16 + 8 * r + 8], src=cvb)
            nisa.nc_match_replace8(dst=cvb, data=cvb, vals=pv[:, b * 16 + 8 * r:b * 16 + 8 * r + 8], imm=-1e9,
                                   dst_idx=pidx[:, 8 * r:8 * r + 8])
        blk = nl.ndarray((128, 16), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.nc_n_gather(dst=blk, data=cj[:, b * NS * 16:(b + 1) * NS * 16], indices=pidx)
        rows = nl.ndarray((128, 16), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=rows, data=blk, op0=nl.multiply, operand0=128)
        nisa.tensor_tensor(dst=pid[:, b * 16:(b + 1) * 16], data1=rows, data2=part, op=nl.add)
    hv = nl.ndarray((128, B * 16), dtype=nl.float32, buffer=nl.hbm)
    hi = nl.ndarray((128, B * 16), dtype=nl.uint32, buffer=nl.hbm)
    nisa.dma_copy(dst=hv, src=pv)
    nisa.dma_copy(dst=hi, src=pid)
    fv = nl.ndarray((B, 128, 16), dtype=nl.float32, buffer=nl.sbuf)          # query on partitions
    fi = nl.ndarray((B, 128, 16), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.dma_copy(dst=fv, src=hv.reshape((128, B, 16)).ap(pattern=[[16, B], [B * 16, 128], [1, 16]], offset=0))
    nisa.dma_copy(dst=fi, src=hi.reshape((128, B, 16)).ap(pattern=[[16, B], [B * 16, 128], [1, 16]], offset=0))
    fv2 = fv.reshape((B, 2048))
    fi2 = fi.reshape((B, 2048))
    tv = nl.ndarray((B, 16), dtype=nl.float32, buffer=nl.sbuf)
    tp = nl.ndarray((B, 16), dtype=nl.uint32, buffer=nl.sbuf)
    for r in range(2):
        nisa.max8(dst=tv[:, 8 * r:8 * r + 8], src=fv2)
        nisa.nc_match_replace8(dst=fv2, data=fv2, vals=tv[:, 8 * r:8 * r + 8], imm=-1e9, dst_idx=tp[:, 8 * r:8 * r + 8])
    tid = nl.ndarray((B, 16), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.nc_n_gather(dst=tid, data=fi2, indices=tp)
    tsc = nl.ndarray((B, 16), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tsc, src=tv)
    o_id = nl.ndarray((B, 16), dtype=nl.int32, buffer=nl.shared_hbm)
    o_sc = nl.ndarray((B, 16), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=o_id, src=tid.view(nl.int32))
    nisa.dma_copy(dst=o_sc, src=tsc)
    return o_id, o_sc


@nki.jit
def gpsimd_kernel(qt, cmat, q8, entry, idx, gp_hbm):
    q_sb = nl.ndarray((D, B), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_sb, src=qt)
    c_sb = nl.ndarray((D, M * KS), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=c_sb, src=cmat)
    ps = nl.ndarray((B, M * KS), dtype=nl.float32, buffer=nl.psum)
    for j in range(M * KS // 512):
        nisa.nc_matmul(dst=ps[:, j * 512:(j + 1) * 512], stationary=q_sb, moving=c_sb[:, j * 512:(j + 1) * 512])
    lut = nl.ndarray((B, M * KS), dtype=nl.int32, buffer=nl.sbuf)            # exact integers
    nisa.tensor_copy(dst=lut, src=ps)
    q_i8 = nl.ndarray((B, D), dtype=nl.int8, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_i8, src=q8)
    ent = nl.ndarray((1, 16), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ent, src=entry)
    hb = nl.ndarray((1, 16), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=hb, src=gp_hbm)
    res_id = nl.ndarray((B, 16), dtype=nl.int32, buffer=nl.sbuf)
    res_sc = nl.ndarray((B, 16), dtype=nl.int32, buffer=nl.sbuf)
    gp.call("ann_search", ins=[lut, q_i8, ent, hb], outs=[res_id, res_sc])
    o_id = nl.ndarray((B, 16), dtype=nl.int32, buffer=nl.shared_hbm)
    o_sc = nl.ndarray((B, 16), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=o_id, src=res_id)
    nisa.dma_copy(dst=o_sc, src=res_sc)
    return o_id, o_sc
