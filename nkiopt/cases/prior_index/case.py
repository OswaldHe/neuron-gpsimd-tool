"""prior_index: nkilib experimental.deepseek_v32.mla_decode_attention.build_prior_index (DeepSeek-V3.2 sparse
MLA decode, trn3): resolve the Lightning Indexer's selected prior positions into paged-KV cache rows.

row = clamp(block_table[t, clamp(pos >> log2(bs), 0, maxb-1)] * bs + (pos & (bs-1)), 0, n_blocks*bs-1),
published as three [128, T, n_prior_tiles] tables (key tile*128+p of token t at [p, t, tile]): rows in the
batched gather's order (a partition/free permutation of the natural order), key positions (fp32, tail -1)
and rows in natural order (uint32). nkilib broadcasts the block table to all 128 partitions (in groups
of sequences that fit 4096 columns), resolves with nc_n_gather + Vector ops, and does the permutation
through an HBM round trip ("no compute engine can perform it in place"). GPSIMD: one call; core c
resolves tokens c, c+8, ... from the SBUF copies of gather_indices/block_table rows (a table lookup per
key in local RAM) and writes the natural-order rows with one 2D iDMA per token; the gather-order
permutation (HBM round trip) and the key positions (DMA) stay nkilib's NKI code.
"""
import os

import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

import gpsimd_nki as gp
from nkilib.experimental.deepseek_v32.mla_decode_attention.prior_index import build_prior_index

T = int(os.environ.get("T", "32"))
NP = int(os.environ.get("NPRIOR", "2048"))
BS = 64
MAXB = int(os.environ.get("MAXB", "2048"))           # 128K context / 64
NBLK = T * MAXB
NPT = -(-NP // 128)
OPS = ["op.cpp"]
CFLAGS = [f"-DPI_BS={BS}", f"-DPI_NBLK={NBLK}"]
PARAMS = {"n_tokens": T, "n_prior": NP, "block_size": BS, "max_blocks_per_seq": MAXB, "n_blocks": NBLK}


def _outs(rows, kpos, rnat):
    o = [nl.ndarray((128, T, NPT), dtype=d, buffer=nl.shared_hbm) for d in (nl.int32, nl.float32, nl.uint32)]
    for dst, src in zip(o, (rows, kpos, rnat)):
        nisa.dma_copy(dst=dst, src=src)
    return tuple(o)


@nki.jit
def baseline_kernel(gather_indices, block_table):
    kv_geom = nl.ndarray((NBLK, 1, BS, 1), dtype=nl.bfloat16, buffer=nl.shared_hbm)   # geometry only
    pi = build_prior_index(gather_indices, block_table, kv_geom)
    return _outs(pi.rows, pi.key_pos, pi.rows_natural)


@nki.jit
def gpsimd_kernel(gather_indices, block_table):
    gi = nl.ndarray((T, NP), dtype=nl.int32, buffer=nl.sbuf)
    bt = nl.ndarray((T, MAXB), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=gi, src=gather_indices)
    nisa.dma_copy(dst=bt, src=block_table)
    rnat = nl.ndarray((128, T, NPT), dtype=nl.int32, buffer=nl.sbuf)
    gp.call("prior_index", ins=[gi, bt], outs=[rnat])
    # gather order: nkilib's partition/free permutation through HBM, unchanged
    ts = 128 * NPT
    perm = nl.ndarray((T * ts,), dtype=nl.int32, buffer=nl.private_hbm)
    nisa.dma_copy(dst=perm.ap(pattern=[[NPT, 128], [ts, T], [1, NPT]], offset=0), src=rnat)
    rows = nl.ndarray((128, T, NPT), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rows, src=perm.ap(pattern=[[1, 128], [ts, T], [128, NPT]], offset=0))
    # key positions: a pure layout copy of gather_indices, loaded key-major by DMA as nkilib does
    pos = nl.ndarray((128, T, NPT), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=pos, value=-1)
    nfull = NP // 128
    gflat = gather_indices.reshape((T * NP, 1))
    nisa.dma_copy(dst=pos[:, :, 0:nfull], src=gflat.ap(pattern=[[1, 128], [NP, T], [128, nfull]], offset=0))
    if NP % 128:
        r = NP % 128
        nisa.dma_copy(dst=pos[0:r, :, nfull:nfull + 1], src=gflat.ap(pattern=[[1, r], [NP, T], [1, 1]], offset=nfull * 128))
    kpos = nl.ndarray((128, T, NPT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=kpos, src=pos)
    return _outs(rows, kpos, rnat.view(nl.uint32))


def make_inputs(rng):
    ctx = MAXB * BS
    gi = np.stack([np.sort(rng.choice(ctx, NP, replace=False)) for _ in range(T)]).astype(np.int32)
    bt = rng.permutation(NBLK).astype(np.int32).reshape(T, MAXB)
    return {"gather_indices": gi, "block_table": bt}


def reference(gather_indices, block_table):
    pos = np.full((T, NPT * 128), -1, np.int64)
    pos[:, :NP] = gather_indices
    lb = np.clip(pos >> int(np.log2(BS)), 0, MAXB - 1)
    rows = np.clip(block_table[np.arange(T)[:, None], lb] * BS + (pos & (BS - 1)), 0, NBLK * BS - 1)
    key_major = lambda a: a.reshape(T, NPT, 128).transpose(2, 0, 1)          # [p, t, tile]
    nat = key_major(rows)
    # gather order: [p, t, tile] holds natural flat entry f = tile*128 + p of the [p', tile'] = divmod(f, NPT) layout
    flat = nat.transpose(1, 0, 2).reshape(T, 128 * NPT)                      # [t, p*NPT + tile]
    gat = flat.reshape(T, NPT, 128).transpose(2, 0, 1)
    return [gat.astype(np.int32), key_major(pos).astype(np.float32), nat.astype(np.uint32)]
