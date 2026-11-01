"""moe_permute: nkilib experimental.subkernels.permute_routed_tokens (MoE all_to_all_v dispatch packing).

The kernel groups T*K routed (token, expert) slots by expert and packs [hidden, affinity, token index]
rows. nkilib builds the slot order with argsort_unstable (T*K/8 dependent max8 + nc_match_replace8
passes), a PE transpose + PSUM copy, an iota written to HBM and read back through an indirect DMA,
and an indirect DMA + dma_compute to form the affinity offsets. The GPSIMD version replaces all of
that with one call (a counting sort over the expert ids that directly emits the token index and the
flattened affinity offset of every sorted slot); the data movement (hidden-row gather, affinity
gather, packing, store) is nkilib's code unchanged.
"""
import math
import os

import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl
import ml_dtypes

import gpsimd_nki as gp
from nkilib.experimental.subkernels.permute_routed_tokens import permute_routed_tokens

T = int(os.environ.get("T", "16"))
K = int(os.environ.get("K", "8"))
E = int(os.environ.get("E", "64"))
H = int(os.environ.get("H", "4096"))
OPS = ["op.cpp"]
CFLAGS = [f"-DMOE_K={K}", f"-DMOE_E={E}"]
PARAMS = {"T": T, "K": K, "E": E, "H": H, "dtype": "bfloat16"}


@nki.jit
def baseline_kernel(hidden_input, expert_index, expert_affinities_masked):
    return permute_routed_tokens(hidden_input, expert_index, expert_affinities_masked)


@nki.jit
def gpsimd_kernel(hidden_input, expert_index, expert_affinities_masked):
    TK = T * K
    out_shape = (TK, H + 1 + 2)
    expert_index_sb = nl.ndarray((1, TK), dtype=nl.int32, buffer=nl.sbuf)
    tok_sb = nl.ndarray((TK, 1), dtype=nl.uint32, buffer=nl.sbuf)       # sorted slot -> token index
    aff_sb = nl.ndarray((TK, 1), dtype=nl.uint32, buffer=nl.sbuf)       # sorted slot -> t * E + expert
    packed_sb = nl.ndarray(out_shape, dtype=hidden_input.dtype, buffer=nl.sbuf)
    packed_hbm = nl.ndarray(out_shape, dtype=hidden_input.dtype, buffer=nl.shared_hbm)

    nisa.dma_copy(expert_index_sb, expert_index.reshape((1, TK)))
    gp.call("moe_route_idx", ins=[expert_index_sb], outs=[tok_sb, aff_sb])

    # from here on: nkilib permute_routed_tokens steps 3.3, 4.2, 5, 6 (bf16 path)
    aff_flat = expert_affinities_masked.reshape((T * E, 1))
    nisa.dma_copy(
        src=aff_flat.ap(pattern=[[1, TK], [1, 1]], offset=0, vector_offset=aff_sb, indirect_dim=0,
                        dtype=hidden_input.dtype),
        dst=packed_sb[:, H:H + 1],
    )
    nisa.dma_copy(
        src=hidden_input.ap(pattern=[[H, TK], [1, H]], offset=0, vector_offset=tok_sb, indirect_dim=0),
        dst=packed_sb[:, :H],
    )
    nisa.tensor_copy(src=tok_sb.view(hidden_input.dtype), dst=packed_sb[:, H + 1:])
    nisa.dma_copy(packed_hbm, packed_sb)
    return packed_hbm


def make_inputs(rng):
    hidden = rng.standard_normal((T, H)).astype(ml_dtypes.bfloat16)
    idx = np.stack([rng.choice(E, K, replace=False) for _ in range(T)]).astype(np.int32)
    aff = np.zeros((T, E), np.float32)
    for t in range(T):
        aff[t, idx[t]] = rng.random(K) + 0.05
    return {"hidden_input": hidden, "expert_index": idx, "expert_affinities_masked": aff.astype(ml_dtypes.bfloat16)}


def _pack(hidden, idx, aff, order):
    TK = T * K
    tok = (order // K).astype(np.uint32)
    exp = idx.reshape(-1)[order]
    out = np.zeros((TK, H + 3), ml_dtypes.bfloat16)
    out[:, :H] = hidden[tok]
    out[:, H] = aff[tok, exp]
    out[:, H + 1:] = tok.view(np.uint16).reshape(TK, 2).view(ml_dtypes.bfloat16)
    return out


def reference(hidden_input, expert_index, expert_affinities_masked):
    order = np.argsort(expert_index.reshape(-1), kind="stable")
    return [_pack(hidden_input, expert_index, expert_affinities_masked, order)]


def check(outs, inputs):
    """Any expert grouping is valid (nkilib's argsort is unstable): every (token, expert) slot exactly once,
    rows ascending by expert, each row [hidden[t], affinity[t, e], t]."""
    o = outs[0]
    hidden, idx, aff = inputs["hidden_input"], inputs["expert_index"], inputs["expert_affinities_masked"]
    tok = o[:, H + 1:].view(np.uint16).reshape(-1, 2).copy().view(np.uint32).reshape(-1).astype(np.int64)
    if tok.min() < 0 or tok.max() >= T:
        return False, "token index column out of range"
    # recover each row's expert: the slots of token t, matched in row order by affinity value
    if np.bincount(tok, minlength=T).tolist() != [K] * T:
        return False, "token multiplicities differ from K"
    if not np.array_equal(o[:, :H].view(np.uint16), hidden[tok].view(np.uint16)):
        return False, "hidden rows do not match their token index"
    exp_sorted = np.sort(idx.reshape(-1))
    a_row = o[:, H].astype(np.float32)
    # expert of row r must be exp_sorted[r] and aff[t, e] must equal the packed affinity
    want = aff[tok, exp_sorted].astype(np.float32)
    if not np.array_equal(a_row, want):
        return False, "affinity column / expert order mismatch"
    pairs = sorted(zip(tok.tolist(), exp_sorted.tolist()))
    if pairs != sorted((t, int(e)) for t in range(T) for e in idx[t]):
        return False, "(token, expert) slots are not a permutation of the routing"
    stable = np.array_equal(o.view(np.uint16), reference(**inputs)[0].view(np.uint16))
    return True, f"{T * K} slots grouped by expert, rows [hidden, affinity, token] correct ({'= stable order, bit-exact' if stable else 'unstable tie order'})"
