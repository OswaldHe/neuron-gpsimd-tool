# trn3 GPSIMD custom-operator toolchain

Write custom operators in C/C++ that run on the 8 GPSIMD cores of a trn3 NeuronCore (the Pool engine),
call them from NKI kernels like any other instruction, and validate them bit-exactly on the device.

```
NKI kernel (kernel.py) ──gp.call markers──► neuronx-cc ──► kernel.neff ──┐
                                                                          ├─ gpsimd-neff analyze ─► call-site tiles
operators (op.cpp) ──xt-clang (base ISA) + runtime ──► ops.so ◄── gpsimd-cc ┘                      (sites_resolved.json)
kernel.neff + ops.so ──gpsimd-neff inject──► final.neff ──gpsimd-run──► bit-exact check + per-core receipts
```

Everything was reverse-engineered on trn3 (SDK 2.33: neuronx-cc 2.28, NKI 0.7, libnrt 2.35, GPSIMD
firmware 1.25.36); see `../PROGRESS.md` for the findings this builds on.

## Quick start

```bash
cd ~/trn3-gpsimd-re/toolchain
./run_examples.sh                    # build + run all 19 examples (~4 min), prints a PASS/FAIL table
./run_examples.sh vadd_i32           # one example
bin/gpsimd-build examples/vadd_i32   # build only -> examples/vadd_i32/build/final.neff
bin/gpsimd-run-example examples/vadd_i32/build --receipts --logs
```

Requirements: trn3 instance with the Neuron driver/runtime, the torch-neuronx venv
(`NEURON_VENV`, default `/opt/aws_neuronx_venv_torch_neuronx_2_14_0`) for NKI, and
`aws-neuronx-gpsimd-tools` (`GPSIMD_TOOLS`, default `/opt/aws/neuron/gpsimd`) for xt-clang.

## Writing an operator

An example directory holds `kernel.py` (NKI kernel + test data + numpy reference) and `op.cpp`.

**NKI side** — pass SBUF tiles (or tile views: slices, strides, partition offsets, 3-D) to `gp.call`:

```python
import gpsimd_nki as gp

@nki.jit
def kernel(a, b):
    out = nl.ndarray((128, 512), dtype=nl.int32, buffer=nl.shared_hbm)
    ta, tb, tc = (nl.ndarray((128, 512), dtype=nl.int32, buffer=nl.sbuf) for _ in range(3))
    nisa.dma_copy(dst=ta, src=a); nisa.dma_copy(dst=tb, src=b)
    gp.call("vadd_i32", ins=[ta, tb], outs=[tc])        # runs on the 8 GPSIMD cores
    nisa.dma_copy(dst=out, src=tc)
    return out

OPS = ["op.cpp"]                                         # operator sources
def make_inputs(rng): return {"a": ..., "b": ...}        # test data, kernel parameter order
def reference(a, b): return [(a.astype(np.int64) + b).astype(np.int32)]   # bit-exact expectation
# optional: HBM = ["tensor", ...] (direct HBM access), CFLAGS = [...], ITERS = 2
```

Any number of calls per kernel, mixed freely with other engines' instructions; inputs and outputs
may be the same tile (in place). Every output tile must be read by something later (unread outputs
are removed by the compiler and the build stops with an explanation).

**Device side** (`include/gpsimd/gpsimd.hpp`; plain C: `gpsimd.h`):

```cpp
#include <gpsimd/gpsimd.hpp>
GPSIMD_OP(vadd_i32) {                          // extern "C" void vadd_i32(gp_call *call), runs on all 8 cores
    gp::Call c(call);
    auto a = c.in<int32_t>(0), b = c.in<int32_t>(1);
    auto o = c.out<int32_t>(0);
    const uint32_t n = o.elems();
    int32_t la[4096], lb[4096];                // staging buffers in local data RAM (stack)
    for (gp::Range blk : c.blocks(o.partitions(), 4096 / n)) {   // this core's partitions, 8 at a time
        c.fetch(a, blk.lo, blk.size(), la);                       // iDMA: one 2D descriptor per tile
        c.dma_wait(c.fetch(b, blk.lo, blk.size(), lb));
        for (uint32_t k = 0; k < blk.size() * n; ++k) la[k] += lb[k];
        c.dma_wait(c.flush(o, blk.lo, blk.size(), la));
    }
}
```

The simplest form, element access straight through the SBUF window, also works
(`for (p : c.partitions(o)) for (i ...) o.store(p, i, a.load(p, i) + b.load(p, i));`) but each SBUF load
costs ~300 cycles; staging blocks through local RAM with the iDMA is ~100x faster (see Performance).

| API | |
|---|---|
| `c.in<T>(k)`, `c.out<T>(k)` | typed tile views (`int8_t`..`uint32_t`, `gp::f32`, `gp::bf16`, `gp::f16`); `c.in_any(k)` reads any dtype as f32 |
| `t.load(p,i)`, `t.store(p,i,v)`, `t.row(p)`, `t.partitions()`, `t.elems()`, `t.contiguous()` | element i = row-major over the view's free dims |
| `c.core()`, `c.ncores()`, `c.split(n)`, `c.split_aligned(n,a)`, `c.partitions(t)`, `c.blocks(n, rows)` | work distribution |
| `c.fetch(t, p, n, local)`, `c.flush(t, p, n, local)`, `t.span()`, `t.offset(i)` | stage partitions [p, p+n) of a tile (any view) through a dense local array with the iDMA; element i of staged row r at `local[r*t.span() + t.offset(i)]` |
| `c.dma(dst, src, bytes)`, `c.dma_2d(dst, src, row_bytes, nrows, src_pitch, dst_pitch)`, `c.dma_wait(ticket)`, `c.dma_wait_all()` | asynchronous iDMA copies between local RAM, SBUF and HBM (in order; pitch `GP_PARTITION_STRIDE` steps across partitions) |
| `c.barrier()`, `c.allreduce_add/max/min(v)` (u32/i32/f32), `c.exscan_add(v,&tot)`, `c.lock()/unlock()`, `c.atomic_add(word,v)`, `c.shared(&n)` | cross-core cooperation through the call's SBUF scratch tile |
| `gp::f32` (+ - * /, compares, `sqrt fma min max floor ceil trunc rint round ldexp`), `GP_F32(1.5)` | IEEE binary32 soft-float, bit-identical to numpy/x86 |
| `gp::math::exp log tanh sigmoid sin cos gelu rsqrt` (`gpsimd/math.hpp`) | float32 elementary functions, mirrored op-for-op by `python/gpsimd_math_ref.py` |
| `gp::hbm_ptr<T>(params, k)` | direct HBM access to tensors listed in `HBM = [...]` |
| `gp_printf(...)`, `gp_receipt_user(i, v)`, `c.fail(bit)` | diagnostics (shown by `gpsimd-run --logs/--receipts`) |

Restrictions (checked by `gpsimd-cc`): no `float`/`double` (the core's FPU/FLIX instructions do not run on
trn3 — use `gp::f32`), no vector intrinsics, no heap, no static constructors, no libc beyond the
runtime's `mem*`/`strlen`, code ≤ 64 KB and data+bss ≤ 7 KB per library. Put staging buffers on the
stack: ~140 KB of data RAM below the entry stack pointer is unused (the examples use up to 64 KB).
Volatile accesses are compiled without `memw` (`-mno-serialize-volatile`), so stores to SBUF/HBM are
posted; the runtime orders memory where it matters (barrier, lock, iDMA submission, end of the call).

## Tools

| | |
|---|---|
| `bin/gpsimd-build <example> [build]` | NKI compile → analyze → gpsimd-cc → inject; writes `build/final.neff`, `build/manifest.json`, test data |
| `bin/gpsimd-run-example <build> [args]` | run `final.neff` with the example's data via gpsimd-run |
| `bin/gpsimd-neff analyze/inject/dump` | call-site resolution, KERNEL_CALL injection + synchronization rewiring, instruction dump |
| `bin/gpsimd-cc --sites <json> -o ops.so op.cpp ...` | build the operator library (generated call-site table, runtime, checks) |
| `bin/gpsimd-run <neff> --in N=f.npy --expect N=f.npy [--sites S] [--iters I] [--hbm T] [--logs] [--receipts] [--save DIR]` | libnrt runner (C++, `runner/gpsimd_run.cpp`) |
| `python/gpsimd_nki.py` | `gp.call`, `gp.compile` |
| `python/gpsimd_math_ref.py` | numpy mirror of `gpsimd/math.hpp` |
| `run_examples.sh` | build + run all examples |
| `tests/softfloat_test.c` | host test: soft-float vs x86 hardware (33 M checks) |
| `probes/bench` | cycle costs (memory spaces, soft-float, cooperation): `bin/gpsimd-build probes/bench && bin/gpsimd-run-example probes/bench/build --logs` |
| `probes/excl` | shows that exclusive-access atomics (L32EX/S32EX) hang (expect an execution timeout) |
| `probes/mem2`, `probes/idma` | window access without `memw`; iDMA discovery and the firmware-state restore experiments (`MODE=0..4`) |
| `isa/`, `re/` | earlier experiments (fixed-operand ISA probe, marker-placement studies), superseded by `examples/` |

## How it works

**Call sites in NKI.** `gp.call` emits GpSimd *marker* instructions (tensor_scalar / memset tagged with
`2^20 + 4096·site + 512·role + 32·arg + chunk` in an immediate operand). The NKI compiler tracks
dependencies per element range and deletes producers of tile regions nobody reads, so the markers read
and write whole tiles: a *touch* (in-place update) per input, a *join* (full write from a broadcast of
the token) per output, chained through a token tile; *order* markers keep call sites from interleaving.
Compute instructions must start at partition 0/32/64/96 (64-partition accesses at 0/64), so views are
split into aligned chunks (extended downwards when a view starts between boundaries).

**Analysis.** `gpsimd-neff analyze` finds the tagged markers in the Pool program and reads each tile's
SBUF partition range, byte offset, dtype and free-axis access pattern from their operands.

**Injection.** Each site's markers are replaced by one `KERNEL_CALL` (opcode 0x85, byte 12 = index into
the NEFF's `gpsimd_kernels_def`) at the position of the site's first output (join) marker (NKI may schedule
Pool work, e.g. an indirect DMA's descriptor generation, between later output markers). Markers become `EVENT_SEMAPHORE`
waits; the scratch fill marker becomes a real `MEMSET` that zeroes the call's scratch tile. The GPSIMD
cores execute Pool instructions independently, ordered only by completion semaphores, so the
KERNEL_CALL waits for that memset and signals the sum of the removed markers' increments on completion
(`SEM_ADD_IMM_COMPLETE`). Every wait on the Pool completion semaphore in all engine programs is
renumbered to wait for the same instruction as before *and* for the KERNEL_CALL of every call site with
a marker at or before it.

**Dynamic control flow** (`nl.fori_loop` / runtime trip counts from `nisa.register_load`) is supported when
every call site lies before the Pool program's first block boundary, i.e. `gp.call` precedes the first
dynamic loop. The compiler splits such programs into blocks; at each boundary every engine drains,
barriers and resets its semaphores (op 0xB0), so waits are renumbered only up to each engine's first reset.
Conditional branches go to label ids (0xA9 -> 0xCC labels, position independent); the unconditional jump
0xDD carries an absolute instruction index (byte 16), which the injector relocates. (Measured: one
`fori_loop` iteration costs ~2.4 us of drain / barrier even for a trivial body; `nkiopt/cases/cf_loop`.)

**Library.** `gpsimd-cc` generates the call-site table (no pointers in data) and one entry stub
`<op>__s<site>` per site, links with the runtime (`runtime/rt.c`: dispatch, receipts, log ring, barrier,
reductions, bakery lock; `builtins.c`: 64-bit/bit helpers, mem*; `softfloat.c`) using the pisplitload
layout (`lsp/`), adds the GNU build-id and one "Neuron" type-2 note per entry, and rejects FLIX code.

**iDMA.** Each GPSIMD core has an Xtensa integrated DMA channel (ERI registers at 0x110000, 64-byte
descriptors). The firmware uses it to load kernel libraries (a ring of descriptors in data RAM that copy
the library from HBM and broadcast it into all 8 cores' instruction/data RAM at `0x01000000 + core·0x100000`).
The runtime borrows the channel during a call: it saves the firmware's settings and last-processed
descriptor (`CURR_DESC`), runs its own descriptor ring, and at the end of the call replays a harmless
16-byte copy at the firmware's last descriptor address so `CURR_DESC` is restored exactly, then points
the fetch at the firmware's next slot (its ring ends in a JUMP). Leaving `CURR_DESC` elsewhere makes the
firmware's later descriptors unreachable (the runtime then fails at model stop).

**Running.** The firmware loads the library at execution and calls the entry stub on all 8 cores.
SBUF is reached through GPSIMD address-translation entry 38 (core address `0xD0000000 + p·0x40000 + off`),
which `gpsimd-run` programs over PCIe BAR0 after `nrt_load`; `--hbm` maps tensors' device memory through
further entries from `0xE0000000`. The runner poisons outputs, executes, compares with `memcmp`, and
reads every core's receipt (PRID, calls, call-site mask, cycles, entry PC in instruction RAM, stack in
data RAM, status) — evidence that the operator ran on the GPSIMD cores.

## Examples (all bit-exact on device)

| example | data formats | parallelism | access pattern | arithmetic |
|---|---|---|---|---|
| `vadd_i32` | int32 | partition split | 2 inputs, contiguous | add |
| `isa_alu` | uint32 | partition split | — | all 90 base-ISA integer instructions (asm) incl. branches, SAR shifts, DEPBITS |
| `isa_mem` | uint32 | partition split | SBUF + local RAM | all load/store forms (8/16/32-bit, signed, acquire/release, literal) |
| `int_subword` | int8, uint8, int16, uint16, int32 | element-interleaved over cores (8 cores store into one word) + partition split | 7 inputs, 6 outputs | saturating add, rounding average, Q15 multiply, byte swap, widening |
| `int64_bits` | 64-bit as uint32 pairs | partition split | — | 64-bit add/mul/div/mod/shifts/compare, popcount/clz/ctz/ffs/bswap/parity, mem* |
| `fp32_math` | float32, int32 | partition split | — | add sub mul div sqrt fma min max floor ceil trunc rint round, conversions, compares (incl. NaN/inf/subnormal) |
| `half_formats` | bf16, fp16, fp32, int8 | partition split | mixed-dtype call | mixed precision, RNE narrowing, quantization |
| `transcendental` | float32 | partition split | 8 outputs | exp log tanh sigmoid sin cos gelu rsqrt (≤ 1.7 ulp) |
| `softmax_rows` | float32 | partition split | row-wise | max / exp / sum / divide |
| `global_reduce` | int32, float32 | all cores → all-reduce | whole-tile reduction | sum (wrap / deterministic float order), max, min, argmax, count |
| `prefix_scan` | int32, int16 | element split + cross-core exclusive scan | 1 × 8192 row, 3-D tile | inclusive scan |
| `single_core_sort` | int32 | one core (core 0 sorts, core 7 counts) | SBUF ↔ local RAM | heap sort |
| `atomics` | int32 | dynamic work queue, mutual exclusion | shared SBUF words | software atomics (bakery lock), barrier, reductions |
| `transpose` | int32, bf16 | output-partition split | cross-partition (column reads) | transpose |
| `gather_hist` | int32, float32 | partition split + lock-protected merge | data-dependent gather over 32 partitions | histogram |
| `views` | int16, int32 | partition split | strided 3-D view, partition offset 32, unaligned 5..44 in place | scale, increment |
| `multi_site` | int32 | — | 3 call sites + Vector-engine + native GpSimd instructions between them | — |
| `hbm_direct` | float32, int32 | partition split | random reads from a 4 MB HBM table, direct HBM output writes | gather |
| `dma_tiles` | int32, int16, int8 | partition split | iDMA: 2D row blocks, strided view spans, 37-byte rows, SBUF→SBUF, column gathers (transpose), double buffering, 64 KB stack buffer, HBM blocks | elementwise |

## Performance

Cycle costs (`probes/bench`, core 0 while all 8 cores run the same loop):

| | cycles |
|---|---|
| local data RAM load / store | 12 / 7 |
| SBUF load through the window | ~296 (blocking: independent loads do not overlap) |
| SBUF / HBM store through the window | 19 / 144 (posted; no `memw`) |
| HBM load through the window | ~390 |
| iDMA SBUF → local, per KB (16 KB transfer) | 128 |
| iDMA local → SBUF, SBUF → SBUF, HBM → local, per KB | 73, 93, 174 |
| iDMA 2D, 16 partitions × 1 KB, per KB | 68 |
| iDMA fixed cost per transfer (16 B / 4 KB) | ~440 / ~600 |
| soft-float add, mul, div, sqrt, fma | 117, 97, 299, 501, 238 |
| `math::exp`, `math::tanh` | ~2600, ~3600 |
| int32 divide, multiply | 20, 3 |
| barrier, all-reduce (8 cores) | ~4600, ~5500 |
| lock + unlock, 8 cores contending | ~22000 |

Rules of thumb: move tile data with the iDMA in blocks (`c.fetch` / `c.flush`, a few KB or more per
transfer) and compute in local RAM; use direct window loads only for scattered single elements (a random
4-byte HBM read costs ~390 cycles directly vs ~560 through an iDMA round trip); stores can stay direct.
`c = a + b` on 128 × 512 int32 runs in 0.12 ms per execution (vs 7.1 ms with element access).

Effect of iDMA staging, posted stores and the `nsau`/`muluh` soft-float primitives on the examples
(core cycles summed over 8 cores and all executions; `run_examples_before_idma.log` vs `run_examples.log`):

| example | core cycles before | after | speedup | main change |
|---|---:|---:|---:|---|
| `vadd_i32` | 120,073,279 | 1,212,295 | 99.0x | iDMA staging |
| `prefix_scan` | 41,055,322 | 584,551 | 70.2x | iDMA block copies |
| `isa_alu` | 117,761,859 | 1,927,952 | 61.1x | iDMA staging (large output) |
| `multi_site` | 35,213,385 | 739,771 | 47.6x | iDMA staging |
| `transpose` | 27,646,913 | 890,960 | 31.0x | 2D column-gather descriptors |
| `views` | 6,591,790 | 695,128 | 9.5x | fetch/flush of strided spans |
| `gather_hist` | 19,507,987 | 3,309,493 | 5.9x | table staged in local RAM |
| `int64_bits` | 33,497,795 | 6,713,276 | 5.0x | iDMA staging |
| `global_reduce` | 15,433,175 | 3,767,574 | 4.1x | iDMA staging |
| `fp32_math` | 29,436,287 | 8,813,179 | 3.3x | iDMA staging + faster soft-float |
| `half_formats` | 53,512,192 | 18,258,229 | 2.9x | iDMA staging + faster soft-float |
| `isa_mem` | 70,300,496 | 27,263,327 | 2.6x | posted stores (no memw) |
| `hbm_direct` | 21,858,582 | 12,057,896 | 1.8x | posted stores |
| `single_core_sort` | 12,663,787 | 7,376,605 | 1.7x | iDMA row copy |
| `softmax_rows` | 176,744,803 | 105,297,846 | 1.7x | soft-float bound |
| `int_subword` | 145,978,907 | 101,813,610 | 1.4x | posted stores (direct sub-word access kept) |
| `transcendental` | 235,565,176 | 176,511,352 | 1.3x | soft-float bound |
| `atomics` | 301,233,283 | 299,666,713 | 1.0x | lock bound |

Float-heavy operators are now bound by soft-float arithmetic (~100–300 cycles per operation).

## Hardware facts and limitations

* Only base-ISA Xtensa code runs: the ncore2gp FLIX/vector/FPU encodings shipped with the tools do not
  execute on the trn3 GPSIMD cores (hang). Floating point is soft-float.
* `L32EX/S32EX/GETEX` (exclusive access) hang on local RAM and SBUF; `S32C1I`, MAC16, `POPC`, `BSWAP` are
  not in the core configuration. Cross-core atomics use the runtime's bakery lock.
* 8- and 16-bit stores through the SBUF window are byte-exact even when 8 cores write the same word.
* SBUF/HBM windows are programmed by the host (`gpsimd-run`) after `nrt_load`; other runners must do the
  same (translation entry 38 / 39, 35, 34, ... — see `runner/gpsimd_run.cpp`).
* The data RAM budget per library is ~8 KB (`--log-size` controls the log ring, default 1 KB).
* The iDMA channel is shared with the firmware (library loading); only the runtime's gp_dma_* functions
  restore its state correctly — do not program the iDMA registers directly.
* Pool programs with dynamic control flow are supported only when every call site precedes the first block
  boundary (see "Dynamic control flow"). If the scheduler puts a native GpSimd instruction inside a call site
  that depends on it, injection stops with an error.
* The NEFF must have a single subgraph; one NeuronCore (`NEURON_RT_VISIBLE_CORES`).
