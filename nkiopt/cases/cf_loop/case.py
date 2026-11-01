"""cf_loop: a GPSIMD call followed by NKI dynamic control flow (a fori_loop whose trip count is read at
run time), the pattern the injector supports since it relocates absolute jumps and renumbers only the
first block's semaphore waits. out = cnt * (x + 1): the call adds 1, the loop accumulates cnt times."""
import os
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl
import gpsimd_nki as gp

OPS = ["op.cpp"]


def _loop(ys, cnt):
    acc = nl.ndarray((128, 64), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0)
    c = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=c, src=cnt)
    reg = nisa.register_alloc()
    nisa.register_load(reg, c)

    def body(i):
        nisa.tensor_tensor(dst=acc, data1=acc, data2=ys, op=nl.add)

    nl.fori_loop(0, reg, body, step=1)
    out = nl.ndarray((128, 64), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=acc)
    return out


@nki.jit
def baseline_kernel(x, cnt):
    xs = nl.ndarray((128, 64), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=xs, src=x)
    ys = nl.ndarray((128, 64), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ys, data=xs, op0=nl.add, operand0=1)
    return _loop(ys, cnt)


@nki.jit
def gpsimd_kernel(x, cnt):
    xs = nl.ndarray((128, 64), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=xs, src=x)
    ys = nl.ndarray((128, 64), dtype=nl.int32, buffer=nl.sbuf)
    gp.call("addone", ins=[xs], outs=[ys])
    return _loop(ys, cnt)


def make_inputs(rng):
    return {"x": rng.integers(-1000, 1000, (128, 64)).astype(np.int32), "cnt": np.array([[int(os.environ.get("CNT", "3"))]], np.int32)}


def reference(x, cnt):
    return (int(cnt[0, 0]) * (x + 1)).astype(np.int32)
