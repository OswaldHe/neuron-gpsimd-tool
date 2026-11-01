# GPSIMD operation coverage (trn3)

What the trn3 GPSIMD cores can execute through this toolchain, and which example validates it on the
device (bit-exact against numpy / Python references). "No" entries were probed and do not work on trn3.

## Base-ISA integer instructions (inline asm, `examples/isa_alu`, 2048 operand triples each + edge pairs)

| group | instructions | status |
|---|---|---|
| add/sub | `add sub addx2 addx4 addx8 subx2 subx4 subx8 addi addmi neg abs` | validated |
| logic | `and or xor` | validated |
| multiply | `mull muluh mulsh mul16u mul16s` | validated |
| divide | `quou quos remu rems` (incl. INT_MIN / -1 → INT_MIN, rem 0) | validated |
| min/max/compare | `min max minu maxu salt saltu` | validated |
| shifts | `slli srli srai`, SAR-based `ssl+sll`, `ssr+srl`, `ssr+sra`, funnel `src` with `ssr ssa8l ssa8b ssai` | validated |
| bit fields | `extui sext clamps nsa nsau depbits` | validated |
| conditional moves | `moveqz movnez movltz movgez` | validated |
| branches | `beq bne blt bge bltu bgeu bany bnone ball bnall bbc bbs beqz bnez bltz bgez beqi bnei blti bgei bltui bgeui bbci bbsi` | validated |
| loops / calls | zero-overhead `loop`, `call8/callx8`, windowed ABI, `l32r` literals | exercised by all compiled code |
| special registers | `rsr.prid`, `rsr.ccount` | used by the runtime (core id, cycle counter) |
| barriers | `memw` (`gp_memw`), `extw`, `isync`, `dsync` | `memw` used by barrier/lock |
| boolean registers | `andb movt movf` | assemble; no use without FP compares (not validated) |
| exclusive access | `l32ex s32ex getex clrex` | **No** — hangs on local RAM and SBUF |
| `s32c1i`, MAC16 (`mul.aa.*`), `popc`, `bswap`, `clz`, `salti` | — | **No** — not in the core configuration |
| FLIX / FPU / vector (IVP) | `add.s`, `mul.s`, vector TIE, ... | **No** — ncore2gp encodings do not execute on trn3 |

## Memory instructions (`examples/isa_mem`, `examples/int_subword`)

| | SBUF (window) | local data RAM | HBM (window) |
|---|---|---|---|
| `l8ui l16ui l16si l32i` (all offsets) | validated | validated | `l32i` validated (`hbm_direct`) |
| `l32ai` (acquire), `s32ri` (release) | validated | validated | — |
| `s8i s16i s32i` | validated, byte-exact with 8 cores writing one word | validated | `s32i` validated |
| `l32r` | — | literal pool in instruction RAM | — |

## Compiler-runtime helpers (`runtime/builtins.c`, `examples/int64_bits`)

64-bit `+ - * / %` (signed/unsigned), shifts, compares, 64x64→128 high product, `popcount clz ctz ffs
parity bswap` (32/64-bit), `memcpy memmove memset memcmp strlen` — validated.

## Floating point (soft-float, `runtime/softfloat.c`, `gp::f32`)

| | status | example |
|---|---|---|
| binary32 `+ - * /`, `sqrt`, `fma` (single rounding) | bit-exact vs numpy/x86 incl. NaN propagation, ±0, ±inf, subnormals | `fp32_math` (+ 33 M host checks in `tests/`) |
| `min max` (x86 `minss/maxss` semantics), compares, `floor ceil trunc rint round` | validated | `fp32_math` |
| int32/uint32 ↔ float32 (truncation, x86 out-of-range → INT32_MIN) | validated | `fp32_math` |
| bf16 ↔ f32 (RNE, NaN → ±0x7FC0 like ml_dtypes), fp16 ↔ f32 (RNE, subnormals, numpy NaN payloads) | validated | `half_formats` |
| `exp log tanh sigmoid sin cos gelu rsqrt` (`gpsimd/math.hpp`) | bit-exact vs `python/gpsimd_math_ref.py`; 0.6–1.7 ulp vs float64 | `transcendental`, `softmax_rows` |

## Data formats on tiles

int8, uint8, int16, uint16, int32, uint32, bfloat16, float16, float32 (any mix per call); 64-bit values as
uint32 pairs. Views: partition offsets (aligned or not), free-axis slices with steps, up to 3 free dims.

## Parallelism and cooperation

| pattern | API | example |
|---|---|---|
| partition blocks per core | `c.partitions(t)`, `c.split(n)` | most |
| element split / interleaved | `c.split(n)`, `c.split_aligned`, `i = core; i += 8` | `prefix_scan`, `int_subword` |
| single core / designated cores | `if (c.core() == k)` | `single_core_sort` |
| barrier | `c.barrier()` | `atomics`, `gather_hist` |
| all-reduce (sum/max/min, u32/i32/f32 deterministic order), exclusive scan | `c.allreduce_*`, `c.exscan_add` | `global_reduce`, `prefix_scan`, `atomics` |
| mutual exclusion, atomic add, dynamic work queue | `c.lock()`, `c.atomic_add()` | `atomics`, `gather_hist` |
| shared scratch words (192 per call, zeroed) | `c.shared()` | `atomics` |

## Memory spaces and data movement

| space | access | cost |
|---|---|---|
| SBUF (all 128 partitions) | core address `0xD0000000 + p·0x40000 + offset` (translation entry 38, mapped by gpsimd-run) | load ~296 cycles (blocking), store 19 (posted) |
| HBM tensors | `gp::hbm_ptr` (entries 39, 35, ... from 0xE0000000, mapped by gpsimd-run) | load ~390, store ~144 |
| local data RAM (per core) | stack (~140 KB free below the entry SP), `.data/.bss` (≤ ~7 KB per library, persists across executions) | ~7–12 |
| iDMA (per core) | `c.fetch/flush`, `c.dma/dma_2d` between any of the above; 1D and 2D (row pitch), asynchronous, in order | ~70–175 cycles per KB + ~440 per transfer |
| instruction RAM | code ≤ 64 KB | — |

iDMA validated by `examples/dma_tiles` (contiguous/strided/odd-width/2D/SBUF→SBUF/HBM, double buffering) and
used by most examples. Other cores' local memories are visible to the iDMA at `0x01000000 + core·0x100000`
(the firmware broadcasts libraries that way); not used by the runtime yet.
