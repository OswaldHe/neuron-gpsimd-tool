"""moe_ep_layer: a whole expert-parallel MoE layer (token generation), the device work of one rank.

    norm, ei, aff = rmsnorm_router_topk_a2av(x, gamma, W_router, top_k=K)       nkilib (all versions)
    send, meta    = dispatch routing math + hidden-row gather                   <- the part that differs
    -- all_to_all_v(send, meta) --                                              loopback (see below)
    y    = moe_tkg(recv[:, :H], local expert, POST_SCALE by recv[:, H])         nkilib expert FFN (all versions)
    -- all_to_all_v(y) back --                                                  loopback
    out  = x + sum over returned rows r of token tok[r]: y[r]                   combine (one-hot matmul, all versions)

The collectives are not run (they need EP ranks): the rank's send buffer is used as its receive buffer
and the expert output as the returned buffer. With balanced routing a rank receives T*K rows on average
(EP ranks x T*K/EP rows each), the same row count as it sends, so the local work per rank is represented
(the receiving rank applies its one local expert to every received row). Like the real collective, the
loopback copy cannot start before the metadata exists: its trip count is read from meta at run time.
Everything else is the real kernel code, in one NEFF.

Versions (dispatch step from ../moe_ep_dispatch):
    BASE=nkilib  nkilib permute_routed_tokens + build_all2all_dispatch_metadata as shipped
    BASE=opt     hand-optimized pure NKI routing math
    gpsimd       routing math in one GPSIMD call (moe_dispatch)
    BASE=floor   routing math replaced by constants (wrong routing, layer output not checked): timing floor
Default: Qwen3-235B-A22B (H=4096, I=1536, E=128, top-8), T=16 tokens per rank, EP=128, one expert per rank.
Mixtral-8x7B: --set T=32 --set K=2 --set E=8 --set I=14336.
"""
import importlib.util
import math
import os

import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl
import ml_dtypes

import gpsimd_nki as gp
from nkilib.core.moe.moe_tkg.moe_tkg import moe_tkg
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode

_spec = importlib.util.spec_from_file_location(
    "moe_ep_dispatch_case", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "moe_ep_dispatch", "case.py"))
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)

T, K, E, R, H, N, NPT, BASE = D.T, D.K, D.E, D.R, D.H, D.N, D.NPT, D.BASE
I = int(os.environ.get("I", "1536"))
OPS = ["../moe_ep_dispatch/op.cpp"]
CFLAGS = D.CFLAGS
PARAMS = dict(D.PARAMS, I=I, stage="layer")
assert BASE in ("nkilib", "opt", "floor") and E == R


# ----------------------------------------------------------------------------- kernel parts
def _dispatch_gpsimd(norm_out, ei, aff):
    ei_sb = nl.ndarray((T, K), dtype=nl.int32, buffer=nl.sbuf)
    aff_sb = nl.ndarray((T, E), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ei_sb, src=ei)
    nisa.dma_copy(dst=aff_sb, src=aff)
    tok_sb = nl.ndarray((N, 1), dtype=nl.uint32, buffer=nl.sbuf)
    tail_sb = nl.ndarray((N, 4), dtype=nl.bfloat16, buffer=nl.sbuf)
    meta_sb = nl.ndarray((4, R), dtype=nl.float32, buffer=nl.sbuf)
    gp.call("moe_dispatch", ins=[ei_sb, aff_sb], outs=[tok_sb, tail_sb, meta_sb])
    packed = D._pack(norm_out, tok_sb, tail_sb)
    meta = nl.ndarray((4, R), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=meta, src=meta_sb)
    return packed, meta


def _dispatch_baseline(norm_out, ei, aff):
    if BASE == "opt":
        return D._opt_dispatch(norm_out, ei, aff)
    if BASE == "floor":
        return D._floor_dispatch(norm_out)
    return D.permute_routed_tokens(norm_out, ei, aff), D.build_all2all_dispatch_metadata(ei, E, NPT, R)


def _loopback_all_to_all(packed, meta):
    """stand-in for all_to_all_v(packed, meta): like the collective, it cannot start before the metadata
    exists. The copy runs in a loop whose trip count is read from meta at run time (total elements sent =
    sd[R-1] + sc[R-1]), as nkilib's permute_a2av does after its collective."""
    bf16, f32, u32 = nl.bfloat16, nl.float32, nl.uint32
    recv_h = nl.ndarray((N, H), dtype=bf16, buffer=nl.shared_hbm)
    recv_aff = nl.ndarray((N, 1), dtype=bf16, buffer=nl.shared_hbm)
    last = nl.ndarray((1, 2), dtype=f32, buffer=nl.sbuf)
    nisa.dma_copy(dst=last[0:1, 0:1], src=meta[0:1, R - 1:R])
    nisa.dma_copy(dst=last[0:1, 1:2], src=meta[1:2, R - 1:R])
    tot = nl.ndarray((1, 1), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=tot, data1=last[0:1, 0:1], data2=last[0:1, 1:2], op=nl.add)
    nisa.tensor_scalar(dst=tot, data=tot, op0=nl.maximum, operand0=1.0)     # floor version: meta = 0
    tot_u = nl.ndarray((1, 1), dtype=u32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tot_u, src=tot)
    reg = nisa.register_alloc()
    nisa.register_load(reg, tot_u)

    def _copy(_):
        nisa.dma_copy(dst=recv_h, src=packed[:, 0:H])
        nisa.dma_copy(dst=recv_aff, src=packed[:, H:H + 1])

    nl.fori_loop(0, reg, _copy, step=N * NPT)
    return recv_h, recv_aff


def _expert_and_combine(hidden_states, packed, meta, w_gate_up, w_down, rank_id):
    bf16, f32, u32 = nl.bfloat16, nl.float32, nl.uint32
    # receiving rank (loopback all-to-all): hidden rows and affinity column of the received buffer
    recv_h, recv_aff = _loopback_all_to_all(packed, meta)
    zero_idx = nl.ndarray((N, 1), dtype=nl.int32, buffer=nl.sbuf)      # every received row: local expert 0
    nisa.memset(dst=zero_idx, value=0)
    y = moe_tkg(recv_h, w_gate_up, w_down, recv_aff, zero_idx, True, rank_id=rank_id,
                expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE, activation_fn=ActFnType.SiLU)
    # sending rank (loopback all-to-all back): out[t] = x[t] + sum of the returned rows of token t
    tokp = nl.ndarray((N, 2), dtype=bf16, buffer=nl.sbuf)
    nisa.dma_copy(dst=tokp, src=packed[:, H + 1:H + 3])
    tok_f = nl.ndarray((N, 1), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tok_f, src=tokp.view(u32))
    iota_t = nl.ndarray((N, T), dtype=f32, buffer=nl.sbuf)
    nisa.iota(dst=iota_t, pattern=[[1, T]], offset=0)
    sel = nl.ndarray((N, T), dtype=bf16, buffer=nl.sbuf)               # [r, t] = (tok[r] == t)
    nisa.tensor_scalar(dst=sel, data=iota_t, op0=nl.equal, operand0=tok_f)
    y_sb = nl.ndarray((N, H), dtype=bf16, buffer=nl.sbuf)
    nisa.dma_copy(dst=y_sb, src=y)
    x_sb = nl.ndarray((T, H), dtype=bf16, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_sb, src=hidden_states.reshape((T, H)))
    out_sb = nl.ndarray((T, H), dtype=bf16, buffer=nl.sbuf)
    CH = 512
    for c in range(H // CH):
        ps = nl.ndarray((T, CH), dtype=f32, buffer=nl.psum)
        nisa.nc_matmul(dst=ps, stationary=sel, moving=y_sb[:, c * CH:(c + 1) * CH])
        nisa.tensor_tensor(dst=out_sb[:, c * CH:(c + 1) * CH], data1=ps, data2=x_sb[:, c * CH:(c + 1) * CH], op=nl.add)
    out = nl.ndarray((1, T, H), dtype=bf16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out.reshape((T, H)), src=out_sb)
    return out


@nki.jit
def baseline_kernel(hidden_states, gamma, router_weights, w_gate_up, w_down, rank_id):
    norm_out, ei, aff = D._router(hidden_states, gamma, router_weights)
    packed, meta = _dispatch_baseline(norm_out, ei, aff)
    out = _expert_and_combine(hidden_states, packed, meta, w_gate_up, w_down, rank_id)
    return out, norm_out, ei, aff, packed, meta


@nki.jit
def gpsimd_kernel(hidden_states, gamma, router_weights, w_gate_up, w_down, rank_id):
    norm_out, ei, aff = D._router(hidden_states, gamma, router_weights)
    packed, meta = _dispatch_gpsimd(norm_out, ei, aff)
    out = _expert_and_combine(hidden_states, packed, meta, w_gate_up, w_down, rank_id)
    return out, norm_out, ei, aff, packed, meta


# ----------------------------------------------------------------------------- inputs, reference, check
def make_inputs(rng):
    inputs = D.make_inputs(rng)
    bf = ml_dtypes.bfloat16
    inputs["w_gate_up"] = (rng.standard_normal((1, H, 2, I), dtype=np.float32) / math.sqrt(H)).astype(bf)
    inputs["w_down"] = (rng.standard_normal((1, I, H), dtype=np.float32) / math.sqrt(I)).astype(bf)
    inputs["rank_id"] = np.zeros((1, 1), np.uint32)
    return inputs


def _ffn(h, w_gate_up, w_down):
    """the local expert on rows h [n, H] (float32 math, bf16 intermediate like the kernel's matmul inputs)"""
    wg = w_gate_up[0, :, 0, :].astype(np.float32)
    wu = w_gate_up[0, :, 1, :].astype(np.float32)
    g = h @ wg
    act = (g / (1 + np.exp(-g))) * (h @ wu)
    act = act.astype(ml_dtypes.bfloat16).astype(np.float32)
    return act @ w_down[0].astype(np.float32)


def _layer_ref(hidden_states, norm, ei, aff, w_gate_up, w_down):
    """out[t] = x[t] + sum_k aff[t, e_k] * FFN(norm[t]) (loopback: the one local expert serves every row)"""
    f = _ffn(norm.astype(np.float32), w_gate_up, w_down)
    s = np.take_along_axis(aff.astype(np.float32), ei.astype(np.int64), 1).sum(1, keepdims=True)
    return hidden_states.reshape(T, H).astype(np.float32) + s * f


def reference(hidden_states, gamma, router_weights, w_gate_up, w_down, rank_id):
    norm, ei, aff, _ = D._router_ref(hidden_states, gamma, router_weights)
    packed, meta = D._dispatch_ref(norm, ei, aff)
    out = _layer_ref(hidden_states, norm, ei, aff, w_gate_up, w_down).astype(ml_dtypes.bfloat16).reshape(1, T, H)
    return [out, norm, ei, aff, packed, meta]


def check(outs, inputs, which):
    out = outs[0]
    sub = {k: inputs[k] for k in ("hidden_states", "gamma", "router_weights")}
    ok, msg = D.check(list(outs[1:]), sub, which)
    if not ok:
        return ok, msg
    if BASE == "floor" and which == "baseline":
        return True, msg + "; layer output not checked (timing floor)"
    norm, ei, aff = outs[1:4]
    want = _layer_ref(inputs["hidden_states"], norm, ei, aff, inputs["w_gate_up"], inputs["w_down"])
    got = out.reshape(T, H).astype(np.float32)
    x = inputs["hidden_states"].reshape(T, H).astype(np.float32)
    err = np.abs(got - want).max()
    scale = np.abs(want - x).max()          # size of the MoE contribution
    rel = np.linalg.norm(got - want) / np.linalg.norm(want - x)
    if rel > 2e-2 or err > 0.1 * scale:
        return False, f"layer output differs from numpy: max abs err {err:.4g} (MoE term max {scale:.4g}), rel L2 {rel:.3g}"
    return True, msg + f"; layer output [{T}, {H}] matches numpy (rel L2 err {rel:.2e}, bf16)"
