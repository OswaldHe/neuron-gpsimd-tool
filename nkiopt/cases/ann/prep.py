#!/usr/bin/env python3
"""prep.py - synthetic ANN dataset, proximity graph and PQ index for cases/ann (cached in data/).

N = 2^19 vectors, d = 128, int8 (a nonlinear 16-dim latent manifold with 64 modes, embedded in 128-d,
plus noise -> realistic low intrinsic dimension). Graph: R = 32 out-edges per node = 24 approximate
nearest neighbours (IVF candidate lists over 1024 clusters, 8 nearest clusters probed) + 8 random
long-range links. PQ: M = 16 subspaces x 8 dims, 256 int8 centroids each (k-means on a sample).
Node record (640 B): 32 neighbour ids (int32) followed by the 32 neighbours' PQ codes (32 x 16 uint8),
so expanding a node is one contiguous read. idx.npy = N records followed by the N int8 vectors (384 MB,
one GPSIMD HBM window). Queries: B = 8 fresh samples (int8). Similarity: inner product (MIPS);
gt = exact top-10 by int8 dot product (score desc, id asc).
"""
import os
import sys
import time

import numpy as np
import torch

N, D, R, KNN, M, KS, B = 1 << 19, 128, 32, 24, 16, 256, 8
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def sample(rng, n, W, modes):
    z = modes[rng.integers(0, len(modes), n)] + 0.35 * rng.standard_normal((n, 16)).astype(np.float32)
    x = np.tanh(z @ W) + 0.03 * rng.standard_normal((n, D)).astype(np.float32)
    x = x / np.linalg.norm(x, axis=1, keepdims=True) * 280.0      # ~constant norm: MIPS ranking ~ L2 ranking
    return np.clip(np.rint(x), -127, 127).astype(np.int8)


def kmeans(x, k, iters, rng):
    c = x[rng.choice(len(x), k, replace=False)].clone()
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        s = torch.zeros_like(c).index_add_(0, a, x)
        n = torch.bincount(a, minlength=k).float()[:, None]
        c = torch.where(n > 0, s / n.clamp(min=1), c)
    return c


def main():
    torch.set_num_threads(os.cpu_count())
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(2026)
    W = (rng.standard_normal((16, D)) / 3).astype(np.float32)
    modes = (rng.standard_normal((64, 16)) * 1.2).astype(np.float32)
    t = time.time()
    X = sample(rng, N, W, modes)
    Qv = sample(np.random.default_rng(99), B, W, modes)
    xf = torch.from_numpy(X.astype(np.float32))
    # IVF clusters for candidate generation
    cen = kmeans(xf[torch.from_numpy(rng.choice(N, 65536, replace=False))], 1024, 8, rng)
    assign = torch.cat([torch.cdist(xf[i:i + 65536], cen).argmin(1) for i in range(0, N, 65536)])
    near = torch.cdist(cen, cen).topk(8, largest=False).indices          # 8 nearest clusters (incl. self)
    members = [torch.nonzero(assign == c).flatten() for c in range(1024)]   # ~512 points per cluster
    print(f"[prep] data + IVF {time.time() - t:.0f}s", flush=True)
    nbr = np.zeros((N, R), np.int32)
    for c in range(1024):
        q = members[c]
        if len(q) == 0:
            continue
        cand = torch.cat([members[j] for j in near[c].tolist()])
        dd = torch.cdist(xf[q], xf[cand])
        k = min(KNN + 1, len(cand))
        idx = cand[dd.topk(k, largest=False).indices]                     # [len(q), KNN+1]
        for r, p in enumerate(q.tolist()):
            row = [v for v in idx[r].tolist() if v != p][:KNN]
            while len(row) < KNN:
                row.append(int(rng.integers(0, N)))
            nbr[p, :KNN] = row
        if c % 128 == 0:
            print(f"[prep] knn cluster {c} {time.time() - t:.0f}s", flush=True)
    nbr[:, KNN:] = rng.integers(0, N, (N, R - KNN))
    # PQ codebooks (int8 centroids) and codes
    sub = xf[torch.from_numpy(rng.choice(N, 65536, replace=False))]
    S = D // M
    books = np.zeros((M, KS, S), np.int8)
    codes = np.zeros((N, M), np.uint8)
    for m in range(M):
        cb = kmeans(sub[:, S * m:S * m + S].contiguous(), KS, 12, rng)
        books[m] = np.clip(np.rint(cb.numpy()), -127, 127).astype(np.int8)
        cbf = torch.from_numpy(books[m].astype(np.float32))
        codes[:, m] = torch.cat([torch.cdist(xf[i:i + 262144, S * m:S * m + S], cbf).argmin(1)
                                 for i in range(0, N, 262144)]).numpy()
    rec = np.zeros((N, R * 4 + R * M), np.uint8)
    rec[:, :R * 4] = nbr.view(np.uint8).reshape(N, R * 4)
    rec[:, R * 4:] = codes[nbr].reshape(N, R * M)
    idx = np.concatenate([rec.reshape(-1), X.view(np.uint8).reshape(-1)])
    entry = int(torch.cdist(xf.mean(0, keepdim=True), xf).argmin())       # medoid-ish entry point
    # exact MIPS ground truth on the int8 data (score desc, id asc)
    sc = (X.astype(np.int64) @ Qv.astype(np.int64).T).T                  # [B, N]
    gt = np.stack([np.lexsort((np.arange(N), -sc[b]))[:10] for b in range(B)]).astype(np.int32)
    np.save(f"{OUT}/X.npy", X)
    np.save(f"{OUT}/Q.npy", Qv)
    np.save(f"{OUT}/books.npy", books)
    np.save(f"{OUT}/codes.npy", codes)
    np.save(f"{OUT}/nbr.npy", nbr)
    np.save(f"{OUT}/idx.npy", idx)
    np.save(f"{OUT}/gt.npy", gt)
    np.save(f"{OUT}/entry.npy", np.array([entry], np.int32))
    print(f"[prep] done {time.time() - t:.0f}s entry={entry}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
