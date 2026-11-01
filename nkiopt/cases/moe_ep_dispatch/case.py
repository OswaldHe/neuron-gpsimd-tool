"""moe_ep_dispatch: the dispatch step of an expert-parallel MoE layer (token generation), one rank.

What one EP rank runs before the all-to-all of an MoE layer, built from nkilib's a2a-v MoE kernels:

    norm, expert_index, affinities = rmsnorm_router_topk_a2av(x, gamma, W_router, top_k=K, softmax)
    send_buffer = permute_routed_tokens(norm, expert_index, affinities)     # [T*K, H+3] rows grouped by expert
    a2a_meta    = build_all2all_dispatch_metadata(expert_index, E, H+3, R)  # [4, R] send counts / displacements
    -> all_to_all_v(send_buffer, a2a_meta)                                   (collective: not part of this demo)

Default size: Qwen3-235B-A22B MoE layer (H=4096, E=128 experts, top-8, softmax router), 16 tokens per rank
(T*K = 128 = the permute kernel's maximum), EP=128 with one expert per rank (R=E), so the per-rank row
counts of the send buffer equal the metadata's send counts.

Versions (all share nkilib's RMSNorm + router and the hidden-row gather of the send buffer):
    BASE=nkilib  baseline_kernel = the three nkilib kernels as shipped
    BASE=opt     baseline_kernel = hand-optimized pure NKI routing math (rank by comparison on one 128x128
                 tile, one-hot matmuls for the permutation/affinity gather, matmul + scan for the metadata)
    gpsimd_kernel  the routing math (argsort, affinity gather, token index, metadata) is one GPSIMD call
    BASE=router  timing reference: the shared RMSNorm + router alone
    BASE=floor   timing reference: router + the shared gather/pack/store with the routing math replaced by
                 constants (wrong output, not checked): what any implementation of the routing math adds to
Second size used: Mixtral-8x7B (T=32 K=2 E=8, EP=8): --set T=32 --set K=2 --set E=8.
"""
import math
import os

import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl
import ml_dtypes

import gpsimd_nki as gp
from nkilib.core.utils.common_types import RouterActFnType
from nkilib.experimental.moe_block.build_all2all_dispatch_metadata import build_all2all_dispatch_metadata
from nkilib.experimental.moe_block.rmsnorm_router_topk_a2av import rmsnorm_router_topk_a2av
from nkilib.experimental.subkernels.permute_routed_tokens import permute_routed_tokens

T = int(os.environ.get("T", "16"))
K = int(os.environ.get("K", "8"))
E = int(os.environ.get("E", "128"))
R = int(os.environ.get("R", str(E)))
H = int(os.environ.get("H", "4096"))
BASE = os.environ.get("BASE", "nkilib")
EPS = 1e-6
N = T * K
NPT = H + 3                      # send-buffer elements per row: hidden, affinity, token index (2 x bf16)
OPS = ["op.cpp"]
CFLAGS = [f"-DMOE_T={T}", f"-DMOE_K={K}", f"-DMOE_E={E}", f"-DMOE_R={R}", f"-DMOE_NPT={NPT}"] + (["-DMOE_TIMING"] if os.environ.get("TIMING") else [])
PARAMS = {"T": T, "K": K, "E": E, "R": R, "H": H, "baseline": BASE}
assert N <= 128 and E % R == 0 and T <= 32


def _router(hidden_states, gamma, router_weights):
    return rmsnorm_router_topk_a2av(hidden_states, gamma, router_weights, None, EPS, K, RouterActFnType.SOFTMAX)


def _pack(norm_out, tok_sb, tail_sb):
    """send buffer from the routing results: gather hidden rows (nkilib step 4.2), append the 3 tail columns"""
    packed_sb = nl.ndarray((N, NPT), dtype=nl.bfloat16, buffer=nl.sbuf)
    packed = nl.ndarray((N, NPT), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    nisa.dma_copy(src=norm_out.ap(pattern=[[H, N], [1, H]], offset=0, vector_offset=tok_sb, indirect_dim=0),
                  dst=packed_sb[:, :H])
    nisa.tensor_copy(dst=packed_sb[:, H:NPT], src=tail_sb[:, 0:3])
    nisa.dma_copy(dst=packed, src=packed_sb)
    return packed


@nki.jit
def baseline_kernel(hidden_states, gamma, router_weights):
    norm_out, ei, aff = _router(hidden_states, gamma, router_weights)
    if BASE == "opt":
        return (norm_out, ei, aff) + _opt_dispatch(norm_out, ei, aff)
    if BASE == "router":                     # the shared router alone (time floor of every version)
        return norm_out, ei, aff
    if BASE == "floor":                      # router + the shared gather/pack/store, routing math replaced by
        return (norm_out, ei, aff) + _floor_dispatch(norm_out)   # constants (wrong result; timing floor)
    packed = permute_routed_tokens(norm_out, ei, aff)
    meta = build_all2all_dispatch_metadata(ei, E, NPT, R)
    return norm_out, ei, aff, packed, meta


@nki.jit
def gpsimd_kernel(hidden_states, gamma, router_weights):
    norm_out, ei, aff = _router(hidden_states, gamma, router_weights)
    ei_sb = nl.ndarray((T, K), dtype=nl.int32, buffer=nl.sbuf)
    aff_sb = nl.ndarray((T, E), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ei_sb, src=ei)
    nisa.dma_copy(dst=aff_sb, src=aff)
    tok_sb = nl.ndarray((N, 1), dtype=nl.uint32, buffer=nl.sbuf)
    tail_sb = nl.ndarray((N, 4), dtype=nl.bfloat16, buffer=nl.sbuf)
    meta_sb = nl.ndarray((4, R), dtype=nl.float32, buffer=nl.sbuf)
    gp.call("moe_dispatch", ins=[ei_sb, aff_sb], outs=[tok_sb, tail_sb, meta_sb])
    packed = _pack(norm_out, tok_sb, tail_sb)
    meta = nl.ndarray((4, R), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=meta, src=meta_sb)
    return norm_out, ei, aff, packed, meta


def _floor_dispatch(norm_out):
    tok_sb = nl.ndarray((N, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.iota(dst=tok_sb, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    nisa.tensor_scalar(dst=tok_sb, data=tok_sb, op0=nl.right_shift, operand0=int(math.log2(K)))
    tail_sb = nl.ndarray((N, 4), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=tail_sb, value=0.0)
    packed = _pack(norm_out, tok_sb, tail_sb)
    zeros = nl.ndarray((4, R), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zeros, value=0.0)
    meta = nl.ndarray((4, R), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=meta, src=zeros)
    return packed, meta


def _opt_dispatch(norm_out, ei, aff):
    """pure-NKI routing math without argsort or indirect DMA: slot s = t*K + k has key e_s*N + s; its packed
    row is dest[s] = #{s': key_s' < key_s} (one compare+reduce over a [N, N] tile); the permutation matrix
    P[s, p] = (dest[s] == p) moves [token, affinity] of every slot to its row with one matmul."""
    f32, bf16, i32, u32 = nl.float32, nl.bfloat16, nl.int32, nl.uint32
    LK = int(math.log2(K))
    # routing results in slot-major layouts
    e_col_i = nl.ndarray((N, 1), dtype=i32, buffer=nl.sbuf)
    nisa.dma_copy(dst=e_col_i, src=ei.reshape((N, 1)))
    e_row_i = nl.ndarray((N, N), dtype=i32, buffer=nl.sbuf)                 # every partition: all N experts
    nisa.dma_copy(dst=e_row_i, src=ei.reshape((1, N)).ap(pattern=[[0, N], [1, N]], offset=0))
    aff_sb = nl.ndarray((T, E), dtype=bf16, buffer=nl.sbuf)
    nisa.dma_copy(dst=aff_sb, src=aff)
    # constants
    iota_p = nl.ndarray((N, 1), dtype=f32, buffer=nl.sbuf)                  # s
    nisa.iota(dst=iota_p, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    iota_f = nl.ndarray((N, N), dtype=f32, buffer=nl.sbuf)                  # s' (= p)
    nisa.iota(dst=iota_f, pattern=[[1, N]], offset=0)
    iota_e = nl.ndarray((N, E), dtype=f32, buffer=nl.sbuf)                  # e
    nisa.iota(dst=iota_e, pattern=[[1, E]], offset=0)
    slot_t = nl.ndarray((T, N), dtype=i32, buffer=nl.sbuf)                  # s - K*t
    nisa.iota(dst=slot_t, pattern=[[1, N]], offset=0, channel_multiplier=-K)
    tok_of_slot_T = nl.ndarray((T, N), dtype=bf16, buffer=nl.sbuf)         # [t, s] = (s // K == t)
    nisa.tensor_scalar(dst=slot_t, data=slot_t, op0=nl.right_shift, operand0=LK)
    nisa.tensor_scalar(dst=tok_of_slot_T, data=slot_t, op0=nl.equal, operand0=0)
    # keys and destination rows
    e_col = nl.ndarray((N, 1), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=e_col, src=e_col_i)
    key_col = nl.ndarray((N, 1), dtype=f32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=key_col, data=e_col, op0=nl.multiply, operand0=float(N), op1=nl.add, operand1=iota_p)
    key_row = nl.ndarray((N, N), dtype=f32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=key_row, data=e_row_i, op0=nl.multiply, operand0=float(N), op1=nl.add, operand1=iota_f)
    lt = nl.ndarray((N, N), dtype=f32, buffer=nl.sbuf)
    dest = nl.ndarray((N, 1), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_scalar_reduce(dst=lt, data=key_row, op0=nl.less, operand0=key_col, reduce_op=nl.add, reduce_res=dest)
    perm = nl.ndarray((N, N), dtype=bf16, buffer=nl.sbuf)                  # [s, p] = (dest[s] == p)
    nisa.tensor_scalar(dst=perm, data=iota_f, op0=nl.equal, operand0=dest)
    # per-slot values V[s] = [token, affinity[t, e_s]]
    aff_rows = nl.ndarray((N, E), dtype=f32, buffer=nl.psum)               # aff[s // K, :]
    nisa.nc_matmul(dst=aff_rows, stationary=tok_of_slot_T, moving=aff_sb)
    V = nl.ndarray((N, 2), dtype=bf16, buffer=nl.sbuf)
    tok_slot_i = nl.ndarray((N, 1), dtype=i32, buffer=nl.sbuf)
    nisa.iota(dst=tok_slot_i, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    nisa.tensor_scalar(dst=tok_slot_i, data=tok_slot_i, op0=nl.right_shift, operand0=LK)
    nisa.tensor_copy(dst=V[:, 0:1], src=tok_slot_i)
    onehot_aff = nl.ndarray((N, E), dtype=f32, buffer=nl.sbuf)
    aff_slot = nl.ndarray((N, 1), dtype=f32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=onehot_aff, data=iota_e, op0=nl.equal, operand0=e_col, op1=nl.multiply,
                              operand1=aff_rows, reduce_op=nl.add, reduce_res=aff_slot)
    nisa.tensor_copy(dst=V[:, 1:2], src=aff_slot)
    # move every slot's values to its packed row
    rows = nl.ndarray((N, 2), dtype=f32, buffer=nl.psum)
    nisa.nc_matmul(dst=rows, stationary=perm, moving=V)
    tok_sb = nl.ndarray((N, 1), dtype=u32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tok_sb, src=rows[:, 0:1])
    tail_sb = nl.ndarray((N, 4), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tail_sb[:, 0:1], src=rows[:, 1:2])
    nisa.tensor_copy(dst=tail_sb[:, 1:3], src=tok_sb.view(bf16))
    packed = _pack(norm_out, tok_sb, tail_sb)
    # all-to-all metadata: member[t, r] = token t has >= 1 expert on rank r
    nloc = E // R
    rank_col = nl.ndarray((N, 1), dtype=f32, buffer=nl.sbuf)
    if nloc == 1:
        nisa.tensor_copy(dst=rank_col, src=e_col)
    else:
        rk = nl.ndarray((N, 1), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=rk, data=e_col_i, op0=nl.right_shift, operand0=int(math.log2(nloc)))
        nisa.tensor_copy(dst=rank_col, src=rk)
    iota_r = nl.ndarray((N, R), dtype=f32, buffer=nl.sbuf)
    nisa.iota(dst=iota_r, pattern=[[1, R]], offset=0)
    onehot_r = nl.ndarray((N, R), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=onehot_r, data=iota_r, op0=nl.equal, operand0=rank_col)
    tok_of_slot = nl.ndarray((N, T), dtype=bf16, buffer=nl.sbuf)           # [s, t] = (s // K == t)
    slot_c = nl.ndarray((N, T), dtype=i32, buffer=nl.sbuf)
    nisa.iota(dst=slot_c, pattern=[[-K, T]], offset=0, channel_multiplier=1)
    nisa.tensor_scalar(dst=slot_c, data=slot_c, op0=nl.right_shift, operand0=LK)
    nisa.tensor_scalar(dst=tok_of_slot, data=slot_c, op0=nl.equal, operand0=0)
    tr = nl.ndarray((T, R), dtype=f32, buffer=nl.psum)                       # experts of t on rank r
    nisa.nc_matmul(dst=tr, stationary=tok_of_slot, moving=onehot_r)
    member = nl.ndarray((T, R), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=member, data=tr, op0=nl.minimum, operand0=1.0)
    ones_t = nl.ndarray((T, 1), dtype=bf16, buffer=nl.sbuf)
    nisa.memset(dst=ones_t, value=1.0)
    cnt_ps = nl.ndarray((1, R), dtype=f32, buffer=nl.psum)
    nisa.nc_matmul(dst=cnt_ps, stationary=ones_t, moving=member)
    sc = nl.ndarray((1, R), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=sc, data=cnt_ps, op0=nl.multiply, operand0=float(NPT))
    ones_r = nl.ndarray((1, R), dtype=f32, buffer=nl.sbuf)
    nisa.memset(dst=ones_r, value=1.0)
    incl = nl.ndarray((1, R), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_tensor_scan(dst=incl, data0=ones_r, data1=sc, initial=0.0, op0=nl.multiply, op1=nl.add)
    sd = nl.ndarray((1, R), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sd, data1=incl, data2=sc, op=nl.subtract)
    zeros = nl.ndarray((2, R), dtype=f32, buffer=nl.sbuf)
    nisa.memset(dst=zeros, value=0.0)
    meta = nl.ndarray((4, R), dtype=f32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=meta[0:1, :], src=sc)
    nisa.dma_copy(dst=meta[1:2, :], src=sd)
    nisa.dma_copy(dst=meta[2:4, :], src=zeros)
    return packed, meta


# ----------------------------------------------------------------------------- inputs, reference, check
def make_inputs(rng):
    x = rng.standard_normal((1, T, H)).astype(np.float32)
    gamma = (1.0 + 0.1 * rng.standard_normal((1, H))).astype(np.float32)
    w = (rng.standard_normal((H, E)) / math.sqrt(H)).astype(np.float32)
    bf = ml_dtypes.bfloat16
    return {"hidden_states": x.astype(bf), "gamma": gamma.astype(bf), "router_weights": w.astype(bf)}


def _router_ref(hidden_states, gamma, router_weights):
    x = hidden_states.reshape(T, H).astype(np.float32)
    norm = x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + EPS) * gamma.astype(np.float32)
    norm = norm.astype(ml_dtypes.bfloat16)
    logits = norm.astype(np.float32) @ router_weights.astype(np.float32)
    ei = np.argsort(-logits, axis=1, kind="stable")[:, :K].astype(np.int32)
    sel = np.take_along_axis(logits, ei, 1)
    p = np.exp(sel - sel.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    aff = np.zeros((T, E), np.float32)
    np.put_along_axis(aff, ei, p, 1)
    return norm, ei, aff.astype(ml_dtypes.bfloat16), logits


def _dispatch_ref(norm, ei, aff, order=None):
    """send buffer and metadata from routing results (stable order unless `order` is given)"""
    flat = ei.reshape(-1)
    if order is None:
        order = np.argsort(flat, kind="stable")
    tok = (order // K).astype(np.uint32)
    packed = np.zeros((N, NPT), ml_dtypes.bfloat16)
    packed[:, :H] = norm[tok]
    packed[:, H] = aff[tok, flat[order]]
    packed[:, H + 1:] = tok.view(np.uint16).reshape(N, 2).view(ml_dtypes.bfloat16)
    ranks = ei // (E // R)
    member = np.zeros((T, R), np.int64)
    member[np.arange(T)[:, None], ranks] = 1
    sc = member.sum(0) * NPT
    sd = np.concatenate([[0], np.cumsum(sc)[:-1]])
    meta = np.stack([sc, sd, np.zeros(R), np.zeros(R)]).astype(np.float32)
    return packed, meta


def reference(hidden_states, gamma, router_weights):
    norm, ei, aff, _ = _router_ref(hidden_states, gamma, router_weights)
    if BASE == "router":
        return [norm, ei, aff]
    return [norm, ei, aff] + list(_dispatch_ref(norm, ei, aff))


def check(outs, inputs, which):
    norm, ei, aff = outs[:3]
    rnorm, _, _, _ = _router_ref(**inputs)
    # 1. router outputs (nkilib, identical code in every version) against numpy, with bf16 tolerance
    d = np.abs(norm.astype(np.float32) - rnorm.astype(np.float32))
    if d.max() > 0.02 * np.abs(rnorm.astype(np.float32)).max():
        return False, f"RMSNorm output differs from numpy (max abs diff {d.max():.4g})"
    logits = norm.astype(np.float32) @ inputs["router_weights"].astype(np.float32)
    for t in range(T):
        s = set(ei[t].tolist())
        if len(s) != K or min(s) < 0 or max(s) >= E:
            return False, f"token {t}: invalid expert ids {ei[t]}"
        rest = np.delete(logits[t], ei[t])
        if logits[t, ei[t]].min() < rest.max() - 1e-2:
            return False, f"token {t}: router top-{K} is not the top-{K} of the logits"
    sel = np.take_along_axis(logits, ei, 1)
    p = np.exp(sel - sel.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    a = aff.astype(np.float32)
    if np.abs(np.take_along_axis(a, ei, 1) - p).max() > 1e-2:
        return False, "affinities differ from softmax over the selected logits"
    mask = np.zeros((T, E), bool)
    np.put_along_axis(mask, ei, True, 1)
    if np.any(a[~mask] != 0):
        return False, "affinities not masked"
    if len(outs) == 3:
        return True, "router outputs ok"
    if BASE == "floor":
        return True, "router outputs ok (timing floor: dispatch outputs not checked)"
    packed, meta = outs[3:]
    # 2. dispatch outputs: exact, from the device's own routing results
    stable, want_meta = _dispatch_ref(norm, ei, aff)
    if not np.array_equal(meta, want_meta):
        return False, f"a2a metadata differs: {np.argwhere(meta != want_meta)[:4].tolist()}"
    if np.array_equal(packed.view(np.uint16), stable.view(np.uint16)):
        return True, f"router ok; send buffer [{N}, {NPT}] and a2a metadata [4, {R}] bit-exact (stable expert order)"
    # nkilib's argsort is unstable: accept any order within an expert, rows must be exact
    tok = packed[:, H + 1:].view(np.uint16).reshape(-1, 2).copy().view(np.uint32).reshape(-1).astype(np.int64)
    if tok.min() < 0 or tok.max() >= T:
        return False, "token index column out of range"
    exp_sorted = np.sort(ei.reshape(-1))
    for r in range(N):
        t, e = tok[r], exp_sorted[r]
        if e not in ei[t]:
            return False, f"row {r}: token {t} is not routed to expert {e} (rows not grouped by expert)"
    pairs = sorted(zip(tok.tolist(), exp_sorted.tolist()))
    if pairs != sorted((t, int(e)) for t in range(T) for e in ei[t]):
        return False, "send buffer rows are not a permutation of the (token, expert) slots"
    order = np.array([np.flatnonzero((ei.reshape(-1) == e) & (np.arange(N) // K == t))[0]
                      for t, e in zip(tok, exp_sorted)])
    got, _ = _dispatch_ref(norm, ei, aff, order)
    if not np.array_equal(packed.view(np.uint16), got.view(np.uint16)):
        return False, "send buffer rows differ from [hidden[t], affinity[t, e], t]"
    return True, f"router ok; send buffer rows exact (unstable order within experts), a2a metadata [4, {R}] bit-exact"
