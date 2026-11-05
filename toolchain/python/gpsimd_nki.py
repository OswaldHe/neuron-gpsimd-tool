"""gpsimd_nki - call trn3 GPSIMD custom operators from NKI kernels.

Usage inside an @nki.jit kernel:

    import gpsimd_nki as gp
    ...
    gp.call("my_op", ins=[t_a, t_b], outs=[t_c])      # t_* are SBUF tiles or tile views

`gp.call` emits *marker* instructions on the GpSimd engine that carry the call site's data
dependencies. After compilation, `gpsimd-neff inject` replaces the markers by one KERNEL_CALL to the
custom operator `my_op` (a C/C++ function in the operator library built by `gpsimd-cc`) and rewires
the synchronization, so the operator runs after all producers of its inputs (on any engine) and
before all consumers / later writers of its inputs and outputs.

Markers of call site s (all GpSimd instructions; every tensor_scalar carries a tag
    TAG(s, role, arg, chunk) = 2**20 + 4096*s + 512*role + 32*arg + chunk
as its immediate operand1, so `gpsimd-neff` identifies them independently of scheduling):
    memset(tok, TAG(s, 0, 0, 0))                                  start            (opcode 0x49)
    per input k, per partition chunk c of in_k:
      tensor_scalar(c, c + tok[0:P] + TAG(s,1,k,c))               touch: reads+writes all of c
      tensor_scalar(tok0, corner(c) + tok0 + TAG(s,2,k,c))        chain: c -> tok0
    tensor_scalar(scratch, bcast(tok) + tok0 + TAG(s,3,0,0))       scratch fill
    tensor_scalar(tok0, scratch[0,0] + tok0 + TAG(s,4,0,0))        scratch chain
    per output j, per partition chunk c of out_j:
      tensor_scalar(c, bcast(tok[0:P]) + tok[0:P] + TAG(s,5,j,c))  join: writes all of c, after the chain
    per output chunk n of the previous call site (right after the start marker):
      tensor_scalar(tok0, corner(prev_n) + tok0 + TAG(s,6,n//32,n%32)) order: call sites never interleave
The compiler tracks dependencies per element range and deletes producers of tile regions nobody reads,
so markers touch whole tiles; every chunk's corner is chained into tok0 so no marker is dead.
Compute instructions need aligned partition starts (0, 32, 64 or 96; 64 partitions at 0 or 64; 128 at 0);
tile views are split into such chunks, the first one extended down to the boundary if the view starts
between boundaries (then the output join is in place so the extra partitions keep their contents).

Each call also gets a hidden scratch tile (1 x 256 int32) in SBUF that the GPSIMD runtime uses for
cross-core synchronization (gp::barrier, reductions).

Building a NEFF: `compile(kernel, example_inputs, out_dir)` compiles for trn3 and writes
`kernel.neff` and `sites.json` (kernel names, tile shapes and dtypes per call site).
"""
import json
import os
import shutil
import tempfile

MARK = 1 << 20
SCRATCH_WORDS = 256
MAX_ARGS = 8
PARTITIONS = 128
ROLE_START, ROLE_TOUCH, ROLE_CHAIN, ROLE_FILL, ROLE_SCHAIN, ROLE_JOIN, ROLE_ORDER = range(7)

_sites = []
_prev_outs = []      # output chunks of the previous call site (ordering markers of the next site read them)


def tag(site, role, arg=0, chunk=0):
    return MARK + 4096 * site + 512 * role + 32 * arg + chunk


def _dtype_name(dt):
    s = str(dt).split(".")[-1].split("'")[0]
    for k in ("bfloat16", "float16", "float32", "uint8", "uint16", "uint32", "int8", "int16", "int32"):
        if s == k or s.endswith("." + k) or s.endswith(k) and not s.endswith("u" + k):
            return k
    return s


def _partition_start(t):
    try:
        off, pstride = int(t.offset), int(t.strides[0])
    except (AttributeError, TypeError, IndexError):
        return 0
    return off // pstride if pstride > 0 else 0


def _partition_chunks(t):
    """Split a tile (view) into partition chunks [(rel_start, size)] that compute instructions accept:
    a chunk starts at partition 0, 32, 64 or 96, and a 64-partition chunk at 0 or 64, a 128-partition
    chunk at 0. rel_start is relative to the view's first partition; a view that starts between those
    boundaries gets a first chunk extended down to the boundary (rel_start < 0)."""
    n = int(t.shape[0])
    p0 = _partition_start(t)
    a, end, out = p0 - p0 % 32, p0 + n, []
    while a < end:
        for size in (128, 64):
            if a % size == 0 and a + size <= end:
                break
        else:
            size = min(end - a, 32)
        out.append((a - p0, size))
        a += size
    return out


def _chunk_view(t, rel, size):
    """View of partitions [p0 + rel, p0 + rel + size) of t's storage with t's free-axis pattern."""
    if rel >= 0:
        return t if (rel == 0 and size == int(t.shape[0])) else t[rel:rel + size]
    pat = t.get_pattern()
    # .ap() offsets are relative to the view's own offset
    return t.ap(pattern=[[pat[0][0], size]] + [list(x) for x in pat[1:]], offset=rel * int(pat[0][0]))


def _corner(t):
    return t[tuple(slice(0, 1) for _ in t.shape)]


def _bcast(tok, t):
    """tok[0:P, 0:1] broadcast to the shape of t (free-dimension stride 0)."""
    return tok.ap(pattern=[[1, int(t.shape[0])]] + [[0, int(n)] for n in t.shape[1:]], offset=0)


def _desc(t):
    return {"shape": [int(x) for x in t.shape], "dtype": _dtype_name(t.dtype),
            "partition_start": _partition_start(t), "chunks": _partition_chunks(t)}


def call(kernel, ins, outs):
    """Call GPSIMD custom operator `kernel` with SBUF input tiles `ins` and output tiles `outs`."""
    import nki.isa as nisa
    import nki.language as nl
    if not outs:
        raise ValueError("a GPSIMD call needs at least one output tile")
    if len(ins) > MAX_ARGS or len(outs) > MAX_ARGS:
        raise ValueError(f"at most {MAX_ARGS} inputs and {MAX_ARGS} outputs per call")
    s = len(_sites)
    eng = nisa.engine.gpsimd
    scratch = nl.ndarray((1, SCRATCH_WORDS), dtype=nl.int32, buffer=nl.sbuf)
    tok = nl.ndarray((PARTITIONS, 1), dtype=nl.float32, buffer=nl.sbuf)
    tok0 = tok[0:1, 0:1]
    _sites.append({"site": s, "kernel": kernel, "ins": [_desc(t) for t in ins], "outs": [_desc(t) for t in outs],
                   "scratch": _desc(scratch)})

    def ts(dst, data, operand0, role, arg=0, chunk=0):
        nisa.tensor_scalar(dst=dst, data=data, op0=nl.add, operand0=operand0, op1=nl.add,
                           operand1=float(tag(s, role, arg, chunk)), engine=eng)

    nisa.memset(dst=tok, value=float(tag(s, ROLE_START)), engine=eng)
    # call sites must not interleave in the Pool program: read the previous site's outputs first
    for n_prev, v in enumerate(_prev_outs):
        ts(tok0, _corner(v), tok0, ROLE_ORDER, n_prev // 32, n_prev % 32)
    _prev_outs.clear()
    for k, t in enumerate(ins):
        for c, (p, n) in enumerate(_partition_chunks(t)):
            v = _chunk_view(t, p, n)
            ts(v, v, tok[0:n, 0:1], ROLE_TOUCH, k, c)
            ts(tok0, _corner(v), tok0, ROLE_CHAIN, k, c)
    ts(scratch, _bcast(tok, scratch), tok0, ROLE_FILL)
    ts(tok0, scratch[0:1, 0:1], tok0, ROLE_SCHAIN)
    for j, t in enumerate(outs):
        for c, (p, n) in enumerate(_partition_chunks(t)):
            v = _chunk_view(t, p, n)
            # a chunk extended below the view must keep the extra partitions' contents: in-place join
            ts(v, v if p < 0 else _bcast(tok, v), tok[0:n, 0:1], ROLE_JOIN, j, c)
            _prev_outs.append(v)


def compile(kernel, example_inputs, out_dir, target="trn3"):
    """Compile an @nki.jit kernel for `target` with numpy example inputs.

    Writes out_dir/kernel.neff and out_dir/sites.json. NKI's standalone path compiles and then tries
    to execute through nrtpy; the execution step is skipped if nrtpy is not installed.
    """
    os.makedirs(out_dir, exist_ok=True)
    os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", target)
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"     # forced: importing torch(-neuronx) sets it to 2
    os.environ["NKI_DISABLE_COMPILE_CACHE"] = "1"
    if shutil.which("neuronx-cc") is None:           # venv python run without activating the venv
        import sys
        os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
    art = tempfile.mkdtemp(prefix="gpsimd_nki_")
    os.environ["NKI_ARTIFACTS_DIR"] = art
    _sites.clear()
    _prev_outs.clear()
    # The NKI front end may trace the kernel more than once (e.g. when torch is imported); every trace
    # must start with empty call-site state, or markers would reference values of an earlier trace.
    orig = getattr(kernel, "func", None)
    if orig is not None:
        import dataclasses
        import functools

        @functools.wraps(orig)
        def _fresh_trace(*a, **kw):
            _sites.clear()
            _prev_outs.clear()
            return orig(*a, **kw)
        kernel = dataclasses.replace(kernel, func=_fresh_trace)     # nki Kernel is a frozen dataclass
    try:
        kernel(*example_inputs)
    except ImportError as e:
        if "nki.runtime" not in str(e):
            raise
    neff = os.path.join(art, "kernel.neff")
    if not os.path.exists(neff):
        raise RuntimeError(f"NKI did not produce {neff}")
    shutil.copy(neff, os.path.join(out_dir, "kernel.neff"))
    shutil.rmtree(art, ignore_errors=True)
    with open(os.path.join(out_dir, "sites.json"), "w") as f:
        json.dump({"mark": MARK, "scratch_words": SCRATCH_WORDS, "sites": list(_sites)}, f, indent=1)
    return os.path.join(out_dir, "kernel.neff")
