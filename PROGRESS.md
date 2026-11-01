# trn3 GPSIMD custom-kernel reverse engineering — progress log

Goal: write a custom operator for the trn3 GPSIMD (Pool engine, "Q7" Xtensa cores), build it with
the gpsimd-tools Xtensa toolchain, call it from an NKI-generated NEFF, and have it print a debug
message containing an instruction operand.

Environment: trn3.48xlarge, Ubuntu 24.04, Neuron SDK 2.33 (neuronx-cc 2.28.7631, NKI 0.7.0,
libnrt 2.35.30, nrtucode FW 1.25.36.0), aws-neuronx-gpsimd-tools 0.25.0.0 (Xtensa RI-2022.9 "14.09",
core config `ncore2gp`), aws-neuronx-gpsimd-customop-lib 0.25.12.0.

Status legend: **[V]** verified on device, **[S]** read from disassembly/strings, **[?]** hypothesis.

## Layout of this directory

| Path | What |
|---|---|
| `kl/` | kernel-library build kit: `build.sh <src.c> <out.so>`, custom LSP `lsp_kl/`, `neuron_kl.h` (kernel note macro), `klpatch.py` (fills note entry addresses), `tchan.h` (timing side channel), `dbg_print.c` (demo kernel) |
| `tools/` | `inject_kl.py` (put a kernel library + KERNEL_CALL into the NKI NEFF), `run_neff.py` (ctypes libnrt runner, checks output), `bench_neff.py`, `tchan_run.py` (decode words), `tchan_print.py` (decode a text message), `xdis.sh`, `annotate.py`, legacy helpers |
| `probes/` | every probe kernel used so far (see the log below) |
| `nki/` | the NKI test kernel and its compiled trn3 NEFF (HBM->SBUF DMA, SBUF->HBM DMA on the Pool queue) |
| `re/` | dumps and reference artifacts |

Typical loop (run from this directory):
```bash
kl/build.sh probes/probeN.c /tmp/pN.so
PY=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/python
$PY -I tools/inject_kl.py nki/kernel_trn3.neff /tmp/pN.so dbg_kernel /tmp/pN.neff [--arg-hex HEX]
NEURON_RT_VISIBLE_CORES=0 NEURON_RT_LOG_LEVEL=ERROR NEURON_RT_EXEC_TIMEOUT=30 $PY -I tools/tchan_run.py /tmp/pN.neff NWORDS 3
```

## Established facts

### Legacy custom ops (trn1 only)
- [V] neuronx-cc refuses `AwsNeuronCustomOp` for trn2/trn3 (`NCC_EVRF030`); trn1 works.
  backend_config: `version=1.0;name=FN;path=LIB;hash=MD5;ulib_to_ucode_version=1.25.36.0;ulib_to_isa_version=1.0.2520.0;`
- [V] Legacy NEFF: `sgNN/ucode_lib.json` + the `.so`, def.json `ucode_lib` + `neff_feature_custom_ops`,
  header feature bit 2, Pool stream `0x85 KERNEL_CALL` + `0x86 CUSTOM_OP_PAYLOAD` per tensor.
- [V] On trn3 the runtime ignores `ucode_lib.json` and rejects `0x86` ("Custom ops only supported on v2 arch").

### trn3 kernel libraries (the working path)
- [V] Engine json `sgNN/Pool0.json` key `"gpsimd_kernels_def": "<file>.json"`; file = JSON array of
  `{"name": <kernel>, "lib": "/<in-NEFF path>"}` (or `"external_lib": <filename>`, resolvable through
  `NEURON_RT_GPSIMD_LIB_OVERRIDES`). Parsed by `parse_one_engine_instr` (libnrt), logged at DEBUG.
- [V] Library = ELF32 Xtensa, segments: text `0x01000000` R+X, data `0x02000000` RW(X),
  `0x03000000` W+X (.dynamic/.rela.got) + PT_DYNAMIC; PT_NOTE with GNU build-id (required) and
  one note per kernel: owner `"Neuron"`, type 2, desc `{u32 entry_link_addr; char name[]}`.
  Type 1 notes (`{u8,u8,sub_opcode,opcode,u32 entry}`) are ISA-opcode kernels and are rejected for
  user libraries. Validator: `nrtucode_kl_create_from_buffer` in libnrtucode_extisa.so.
- [V] Build: copy of Cadence `pisplitload` LSP + notes segment; link through `-mlsp=<dir>` with
  `-Wl,--shared-pagesize=128 -Wl,-pie -Wl,--build-id` (raw `-T` + `-pie` fails).
- [V] `KERNEL_CALL` = opcode 0x85, 64 bytes; byte 12 = index into gpsimd_kernels_def, replaced by
  the runtime with the global kernel id (`nrtucode_ll_builder_add_kl_call`); other bytes passed on.
- [V] Only the kernel libraries a NEFF uses are assembled (budget IRAM 64 KB, DRAM 8 KB).
- [V] The firmware calls the library `_start(ctx)` once per load; ctx = `0x00080878`, a table:
  `[0x10, self, fn 0x3628, fn 0x3644, 0x82100 ("UCPL" block), 0x10000, 0, fn 0x3658 (noreturn,
  calls 0xd888 with 1), fn 0x3660, fn 0x309c, 0, 1, 2, 2, 0xc496, 0xc4a6, ...]`.
- [V] Kernels run on all 8 cores (PRID 0..7) via `callx12` from firmware PC 0x1468; arguments
  a2=0x92b00 (data RAM, zeros), a3=0, a4=core index, a5=a2+0x3c, a6=0x80, a7=0.
- [V] `.bss` persists across executions of a loaded model; data RAM 0x80000-0xBFFFF readable;
  instruction RAM readable with l32i; kernel stack near 0xB9A50.
- [V] Only base-ISA code from the ncore2gp toolchain runs on trn3: ncore2gp FLIX/IVP (Vision TIE)
  code hangs. Builtin trn3 kernels use FLIX bundles ncore2gp cannot decode (`?f` first byte;
  `bf`/`4f` look 8-byte, `3f` 16-byte). Build kernels with `-mno-flix -mno-generate-flix -mno-coproc`.
- [V] No GPSIMD stdout on trn3 (`aws_hal_q7_swap_file_io_table_mariana` is a stub).
- [V] Output channel used so far: timing side channel (`kl/tchan.h`): 4-bit symbol = busy-wait,
  host times `nrt_execute` (16-level calibration, 3x repeat, ~680 us/level, error-free).
- [V] Demo: `kl/dbg_print.c` prints `dbg_print cpu0 a2=0x00092b00 a4=0x00000000 a5=0x00092b3c a6=0x00000080 ctx=0x00080878`.

### Not yet found
- The KERNEL_CALL argument bytes 13..63 are not anywhere in data RAM 0x80000-0xBFFFF.
- SBUF access: `0x80000000` (trn1 fixed SBUF window) reads zeros; `0x00100000` (trn1 local regs) hangs;
  full 32-bit sweep for the tile hangs somewhere. SoC SBUF bases: libnrt `mariana_sbuf_base_addresses`
  = 0x20_0000_0000, 0x30_..., 0x60_..., 0x70_..., 0x8020_..., ... ; HAL Q7 params (mariana):
  base 0x3100000, 0x20000, 0x40000, 0x60000, 0x40000, 8 cores.

### Session 2 additions (2026-11-01)
- [V] **Host access to on-chip memory over PCIe BAR0** (`tools/devmem.py`): driver ioctl
  `NEURON_IOCTL_BAR_READ/WRITE` (`'N'` 11/12), `/dev/neuronN` is world-rw, max 100 registers per
  call. BAR0 PA from `/sys/bus/pci/devices/<bdf>/resource` (device 0 = `0x2c4000000000`).
  TPB *i* window = BAR0 + `0xd0000000 + i*0x4000000` (64 MB). Our NEFFs run on **TPB4** of device 0.
  - SBUF: TPB + `p*0x40000 + off` (verified: input tile at `+0x3FE00`).
  - GPSIMD core *k* local address space: TPB + `0x3100000 + k*0x100000` + local addr
    (IRAM local 0x0.., data RAM local 0x80000-0xBFFFF). Unmapped space reads `0xdeadbeef`.
  - Pool engine runtime instruction ring: TPB + `0x3000000`; the in-flight KERNEL_CALL with all
    operand bytes was at TPB+`0x3001000` (copies also at `+0x2609000`, `+0x2611000`).
  - Firmware dumps: `re/fw/tpb4_core{0,1}_{iram,dram}*.bin` (all 8 cores run identical firmware).
  - After a hang the runtime does not scrub GPSIMD memory, so logs can be read post-mortem;
    after a clean `nrt_close` memory is cleared, so read logs in-process (`run_neff.py --klog`).
- [V] **Real printf for kernels**: `kl/kprint.h` (`kprintf`, `%d %u %x %X %p %s %c`, width/zero-pad)
  writes a per-core ring in the library `.bss`; `tools/klog.py` finds and prints them;
  `tools/run_neff.py <neff> --klog` runs and prints. Demo `probes/kprint_demo.c`:
  `hello from GPSIMD core N: a2=0x00092b00 a3=0 a4=N a5=0x00092b3c a6=0x80 a7=0 ctx=0x00080878` x8 cores.
  The timing channel (`kl/tchan.h`) is no longer needed; it also produced occasional ±1 nibble
  errors that misled earlier decoding (e.g. firmware returns are standard `retw.n` = `1d f0`, and
  the dispatcher call at 0x1468 is `callx8 a2`, so kernel a2..a7 = dispatcher a10..a15).
- [V] **Core-local `0x80000000`-`0x83FFFFFF` is a 64 MB window onto device HBM** (contains device
  pointers like `0x8008c01bc420`, i.e. HBM `0x8008c0000000 + x` at local `0x80000000 + x`
  [S: inferred from self-referencing pointers]); the NKI input tensor `a` was found at local
  `0x80599000`. `0x84000000` and `0x00100000` hang the core. No SBUF in this window
  (host-planted SBUF magic not found).
- [V] Bundle lengths: trn3 shares ncore2gp's length encoding in the builtin library
  (`?e` = 16 B; `?f` = 16 B if byte3 bit0 else 8 B: all 92 `.xt.prop` instruction segments of
  `re/all_stripped_emb47.elf` tile exactly), but firmware has exceptions (bundle `5f42f229...` at
  0x3647 is 8 B) and per-slot decoding with ncore2gp's ISA tables (`tools/xslot`) yields nonsense:
  trn3 FLIX formats/slots differ. Static decoding of trn3 bundles is not feasible with the
  shipped config; `tools/fwdis.py` keeps base-ISA instructions readable between bundles.

### Session 3 additions (2026-11-01): SBUF access from GPSIMD kernels — WORKING
- [V] **GPSIMD address-translation block** at TPB window + `0x3062000` (BAR0-writable): 40 entries
  of 7 u32 `{ctrl, mask_lo, mask_hi, local_lo, local_hi, soc_lo, soc_hi}`; ctrl bit0 = valid; a core
  address `a` with `(a & mask) == local` goes to `soc + (a & ~mask)`. Identical blocks for the other
  engines at `+0x2462000`, `+0x2662000`, `+0x2862000`, `+0x2b62000`. Found by BAR0-scanning for the
  HBM window's SoC address. Runtime-programmed entries: 0 (`0x80000000` 16 MB -> HBM 0x8008c0000000),
  1 (`0x84000000` -> 0x801002000000), 4 (`0x90000000` 32 MB -> HBM 0x8008c0000000), 6, 7, and pairs
  8..37 (`0x2N000000` ctrl 0xNc; `0x1N000000` ctrl 5 -> 0x0?00..0803000000). 38, 39 are free.
- [V] The firmware keeps a software window table at data RAM `0x81b9c` (15 x 40 B:
  `{id, valid, ?, soc(64), mask(64), local, size, ?}`, locals 0x84000000 + k*0x4000000) and the SBUF
  SoC base of every TPB at `0x81f80` (= libnrt `mariana_sbuf_base_addresses`: 0x20_0000_0000,
  0x30_.., 0x60_.., 0x70_.., 0x8020_.., 0x8030_.., 0x8060_.., 0x8070_..). The firmware's software DGE
  computes SBUF SoC addresses itself (0x8020_0003fe00 for our tile seen on its stack).
- [V] **Mapping SBUF**: `tools/sbufwin.py` writes entry 38 = core `0xD0000000`, 32 MB -> TPB SBUF
  base; core address `0xD0000000 + p*0x40000 + off` is SBUF partition p, byte off. Done by the host
  after `nrt_load`, before `nrt_execute` (`run_neff.py --map-sbuf [TPB]`). Survives repeated executions.
- [V] **Demo `kl/examples/sbuf_add.c`**: int32 NKI tile (128 x 64 at SBUF 0x3FE00); each of the 8 cores
  adds 1000 to its 16 partitions in place; the NEFF's DMA copies the tile out; all 8192 outputs equal
  input + 1000 (5 consecutive executions), per-core kprintf shows partitions 16k..16k+15.
  ```bash
  cd ~/trn3-gpsimd-re
  NEFF=kernel_trn3_i32.neff KERNEL=sbuf_add RUN_ARGS="--map-sbuf 4 --expect add:1000" \
      tools/probe_run.sh kl/examples/sbuf_add.c
  ```
- [V] `nki/kernel_i32.py` -> `nki/kernel_trn3_i32.neff` (same Pool stream shape, dtype int32).
- [?] Candidate route to KERNEL_CALL operands: the GPSIMD sequencer fetches from HBM (timeout dump
  showed the GPSIMD instruction pointer 0x8008c0591040, inside the 16 MB HBM window at core
  `0x80000000`), and the prefetch ring is at TPB+0x3000000 (mappable via a free entry, e.g. 39).
  Needs the current instruction pointer (sequencer register) to locate "our" KERNEL_CALL.
- Floating point: not yet tested (ncore2gp FP/Vision TIE encodings may not run on trn3; int32 used).

### Session 4 (2026-11-01): end-to-end demo package `demo/` — WORKING
Self-contained: `demo/README.md`, `demo/run_demo.sh` (one command, ~20 s), NKI kernel
`demo/nki/sbuf_add_nki.py`, C++ operator `demo/gpsimd/sbuf_add.cpp` (xt-clang++,
`demo/gpsimd/build_kernel_lib.sh`), NEFF modifier `demo/tools/neff_tool.py` (`info` decodes the SBUF
tile from the DMA instructions; `inject` adds library + KERNEL_CALL and re-wires semaphores), C++
runner `demo/runner/run_neff.cpp` (libnrt; SBUF mapping via driver ioctls DEVICE_INFO/BAR_WRITE;
seeded full-range int32 input; memcmp bit-exact check; reads per-core receipts + kprintf logs over
BAR0; unmaps). [V] control (unmodified NEFF) bit-exact identity, no receipts; modified NEFF bit-exact
input+ADDEND (ADDEND 1000 and -123456789, seeds 12345/7, 1-5 executions). GPSIMD proof: receipt in
core k's data RAM has PRID k for k = 0..7, execution count = nrt_execute calls, per-core checksums of
values read from SBUF equal the host's checksums of the random input, code at GPSIMD IRAM 0x10000,
stack in GPSIMD DRAM, ccount advanced (~600 k cycles), caller = firmware dispatcher 0x146b; runtime
DEBUG log shows `Initialized kernel library 'sbuf_add.so'` and `Q7_POOL[0]: 1 kernel libraries`.
[V] `rsr.ccount` works on trn3 GPSIMD; C++ (no exceptions/RTTI/static ctors) works.

### Toolchain status (for the new trn3 GPSIMD custom-operator toolchain)
| Stage | Tool | State |
|---|---|---|
| Build kernel library | `kl/build.sh` (+ `lsp_kl/`, `neuron_kl.h`, `klpatch.py`) | works (base ISA only) |
| Device printf | `kl/kprint.h` + `tools/klog.py` | works |
| Inject into NKI NEFF | `tools/inject_kl.py` | works for the test NEFF shape (b1/d4/d4/a4 Pool stream) |
| Run + check + logs | `tools/run_neff.py --klog` | works |
| Inspect device memory | `tools/devmem.py`, `tools/scan_magic.py` | works |
| Disassemble firmware | `tools/fwdis.py`, `tools/xslot` | partial (trn3 FLIX opaque) |
| Read/write SBUF from a kernel | `kl/gpsimd.h` (`sbuf_ptr`, `sbuf_i32`) + `run_neff.py --map-sbuf` / `tools/sbufwin.py` | **works** (demo `kl/examples/sbuf_add.c`) |
| Read HBM tensors from a kernel | core 0x80000000 window (runtime entry 0) | found; needs device-address helper |
| Read KERNEL_CALL operands from a kernel | — | open (see Session 3 candidate route) |
| Generic build/inject/run CLI | `tools/probe_run.sh` (env NEFF, KERNEL, RUN_ARGS, INJECT_ARGS) | works for the test NEFFs |

### Next steps
1. KERNEL_CALL operands: find the GPSIMD sequencer's current-instruction register (BAR0 scan for
   the instruction pointer printed on timeouts), then read the instruction from HBM via core
   0x80000000 or map the ring (TPB+0x3000000) with entry 39.
2. Make the SBUF mapping self-contained (kernel-side or injector-side instead of a host step), and
   auto-detect the TPB of the loaded model instead of assuming TPB4.
3. Float support: test scalar FP / soft-float on trn3; vector ops need the trn3 TIE (unavailable).
4. Generalize `inject_kl.py` (arbitrary NKI Pool streams, semaphore re-balancing, multiple calls,
   operands passed in KERNEL_CALL bytes 13..63).

## Log
- Session 1 (2026-11-01): kernel-library format, NEFF wiring, timing channel, dbg_print demo.
- Session 2 (2026-11-01): BAR0 device access, kprintf, firmware dumps, instruction ring located,
  HBM window at local 0x80000000, FLIX length study.
- Session 4 (2026-11-01): `demo/` end-to-end package with C++ operator, NEFF tool, C++ runner,
  bit-exact validation and GPSIMD execution receipts.
- Session 3 (2026-11-01): firmware window table + SBUF base table, hardware translation block at
  TPB+0x3062000, SBUF mapped at core 0xD0000000, sbuf_add demo verified (8 cores, in-place, exact).
- Session 5 (2026-11-01): generalized toolchain in `toolchain/` (README.md, OPS.md, run_examples.sh,
  18 examples all bit-exact on device). Findings:
  * NKI compiler tracks dependencies per element range and DCEs producers of unread tile regions ->
    gp.call markers read+write whole tiles (touch = in-place tensor_scalar, join = full write from
    broadcast tok), each chunk's corner chained into tok0; markers tagged via tensor_scalar op1=add
    imm1=TAG (byte 37 op1, byte 44 imm1 f32). Order markers (read previous site's outputs) stop the
    scheduler from interleaving call sites (the compiler serializes all Pool instructions on S3, so
    interleaved sites look mutually dependent).
  * Compute instructions: partition start must be 0/32/64/96 for <=32 partitions, 0/64 for 64, 0 for
    128 (even 1-partition accesses). Unaligned views: chunk extended down to the 32-boundary
    (in-place join to keep the extra partitions). NkiTensor.ap(offset) is relative to the view.
  * TENSOR_SCALAR 0x43: src@16 steps@20(3xs16, innermost first, elements) nums@26, in_dt@32 out_dt@33
    channels@34 imm0_ptr@35 op0@36 op1@37 imm0@40 imm1@44 dst@48 steps@52 nums@58.
    MEMSET 0x49: elems@28 dtype@32(!) channels@34 value@40 (dst dtype) dst@44 steps@48 nums@56.
    IOTA 0x7e: offset@12 dtype@33 channels@34 dst@44 steps@48 nums@56.
  * KERNEL_CALL with SEM_ADD_IMM_COMPLETE (mode 21, value @8, no wait) works.
  * GPSIMD cores execute Pool instructions independently (each takes its partitions); Pool->Pool
    ordering is by completion semaphores -> injected scratch memset increments S3 and KC waits on it.
  * 8/16-bit loads/stores via SBUF window are byte-exact even with 8 cores storing into one word.
  * L32EX/S32EX/GETEX (exclusive access) hang on local DRAM and SBUF -> no HW atomics; runtime has a
    Lamport bakery lock in the zeroed per-call scratch tile (gp_lock / gp_atomic_add_u32).
  * xt-clang always emits FPU/FLIX for float (no soft-float switch) -> gp::f32 soft-float class,
    GP_F32() constexpr literal; NaN conventions follow numpy/ml_dtypes for bf16/fp16 conversions.
  * all 90 base-ISA integer instructions and all load/store forms validated bit-exact.
  * HBM: tensors' device addresses via nrt_tensor_get_device_allocation_info; one translation window
    per tensor (entries 39, 35, 34, ... local 0xE0000000+, >= 16 MB, aligned) works for read and write.
    Runtime output tensors live near 0x8008c0000000 (= entry 0's region), inputs e.g. 0x800001000000.
  * Costs: SBUF ld/st ~300 cycles (blocking, no overlap), HBM ld ~390 / st ~306, local RAM ~11,
    soft-float add ~142, div ~300, exp ~3400, barrier ~5000, contended lock ~20700.
  * Stack: 4 KB local arrays are fine. Library data+bss budget ~8 KB (keep log ring <= 2 KB).

### Toolchain status (session 5)
| Piece | Path | Status |
|---|---|---|
| NKI call API | `toolchain/python/gpsimd_nki.py` (`gp.call`, `gp.compile`) | works: many sites, views, in-place |
| Analyzer / injector | `toolchain/bin/gpsimd-neff` | works (sync renumbering, scratch memset) |
| Library builder | `toolchain/bin/gpsimd-cc` (+ runtime/, lsp/, include/gpsimd/) | works |
| Runner | `toolchain/bin/gpsimd-run` (runner/gpsimd_run.cpp) | works (.npy I/O, SBUF + HBM windows, receipts, logs) |
| Driver | `toolchain/bin/gpsimd-build`, `gpsimd-run-example`, `run_examples.sh` | works |

- Session 6 (2026-11-01): memory performance.
  * The ~300-cycle stores were the compiler's `memw` before every volatile access: with
    `-mno-serialize-volatile` SBUF stores are posted (19 cycles); loads stay ~296 (blocking, no overlap).
  * Each GPSIMD core has an Xtensa iDMA (ERI 0x110000: SETTINGS 0x00, TIMEOUT 0x04, DESC_START 0x08,
    NUM_DESC 0x0C, DESC_INC 0x10, CONTROL 0x14, STATUS 0x40 (state bits 2:0, errors from bit 12),
    CURR_DESC 0x44 = last processed descriptor; SETTINGS bit 7 = fetch start; 64-byte descriptors
    {control = 1 | subtype<<4 (0 = 1D, 1 = 2D), src, dst, row bytes, src_pitch, dst_pitch, nrows, ...}).
    It reaches local RAM, the SBUF window (0xD0000000) and HBM windows: 4 KB SBUF->local in ~600 cycles
    (~500x faster than word loads), ~70-175 cycles/KB streaming.
  * The firmware uses each core's iDMA to load kernel libraries: ring of 64 x 64-byte descriptors at
    data RAM 0x809c0 (header at 0x80980: next_add, start, desc size, end 0x819c0, type 4, count), ending in
    a JUMP; copies HBM (core 0x90000000 window) -> staging 0x82100.. -> broadcast with a 2D descriptor
    (dst pitch 0x100000, 8 rows) to all cores' IRAM/DRAM at 0x01000000 + core*0x100000 + local address.
  * Borrowing the channel: restoring DESC_START is not enough - the firmware needs CURR_DESC to be its
    own last descriptor (otherwise "dlr_wait_model_stop_request: failed to kickoff model stop request"
    at unload). Fix: replay a 16-byte local copy written over the firmware's last descriptor (then
    restore its bytes) so CURR_DESC is exact, and set DESC_START = last + 64 (runtime gp_dma_end).
  * ~140 KB of data RAM below the entry SP (0x95c00-0xb8c00) is unused: 64 KB stack buffers work.
  * Soft-float uses nsau / mull+muluh on the device (add 117, mul 97 cycles; exp ~2600).
  * Toolchain: gp_dma_* runtime + Tile::fetch/flush/span, Call::dma*/blocks; dense free dims merged by the
    analyzer; 15 examples converted to iDMA staging; new examples/dma_tiles. Speedups up to 99x
    (vadd_i32 7.1 ms -> 0.12 ms per execution). All 19 examples bit-exact.

### Possible next steps
1. Faster soft-float (inline fast paths for normal operands; division/sqrt without bit loops) - now the
   bottleneck of float-heavy operators. Cross-core exchange through other cores' local RAM
   (0x01000000 + core*0x100000 via iDMA) instead of SBUF scratch.
2. Self-contained NEFFs (SBUF/HBM window programming without the host runner, e.g. from the operator's
   `_start` or by the injector) so other runtimes (torch-neuronx) can run them.
3. Control flow in Pool programs (loops in NKI kernels) for the injector; multi-subgraph NEFFs.
4. Operator arguments in the KERNEL_CALL instruction (scalar parameters without a params tile).
