"""moe_dispatch_meta: nkilib experimental.moe_block.build_all2all_dispatch_metadata (MoE all-to-all-v).

send_counts[r] = (#tokens with at least one of their K experts on rank r) * elements_per_token,
send_displs = exclusive prefix sum. nkilib unrolls ~12 dependent Vector/Scalar/PE instructions per
destination rank (range masks via relu, max over K for the de-duplication, a transpose and a sum over
tokens), then a scan. The GPSIMD version is one call: a per-token rank bitset, counts, scan.
"""
import os

import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

import gpsimd_nki as gp
from nkilib.experimental.moe_block.build_all2all_dispatch_metadata import build_all2all_dispatch_metadata

T = int(os.environ.get("T", "32"))      # nkilib: nc_transpose limits T to 32
K = int(os.environ.get("K", "8"))
E = int(os.environ.get("E", "256"))
R = int(os.environ.get("R", "64"))
NPT = int(os.environ.get("NPT", "7170"))       # elements per token (hidden + packed affinity/index columns)
OPS = ["op.cpp"]
CFLAGS = [f"-DMETA_R={R}", f"-DMETA_NLOCAL={E // R}", f"-DMETA_NPT={NPT}"]
PARAMS = {"T": T, "K": K, "E": E, "R": R, "elements_per_token": NPT}


@nki.jit
def baseline_kernel(expert_index):
    return build_all2all_dispatch_metadata(expert_index, E, NPT, R)


@nki.jit
def gpsimd_kernel(expert_index):
    out = nl.ndarray((4, R), dtype=nl.float32, buffer=nl.shared_hbm)
    ei = nl.ndarray((T, K), dtype=nl.int32, buffer=nl.sbuf)
    meta = nl.ndarray((4, R), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ei, src=expert_index)
    gp.call("dispatch_meta", ins=[ei], outs=[meta])
    nisa.dma_copy(dst=out, src=meta)
    return out


def make_inputs(rng):
    return {"expert_index": np.stack([rng.choice(E, K, replace=False) for _ in range(T)]).astype(np.int32)}


def reference(expert_index):
    nl_ = E // R
    ranks = expert_index // nl_
    mask = np.zeros((T, R), np.int64)
    mask[np.arange(T)[:, None], ranks] = 1
    counts = mask.sum(0) * NPT
    displs = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return [np.stack([counts, displs, np.zeros(R), np.zeros(R)]).astype(np.float32)]
