# trn3 GPSIMD custom operator demo: NKI kernel + C++ op on the GPSIMD cores

This demo runs a **user-written C++ operator on the 8 GPSIMD (Pool engine) cores of a trn3
NeuronCore**, called from a kernel written in **NKI**. The operator reads a tensor tile from
**SBUF**, adds a constant to every element **in parallel on all 8 GPSIMD cores**, and writes the
result back to **SBUF**; the NKI kernel's DMA then copies the tile to HBM. A C++ runner checks the
output **bit-exactly** and **proves the operator ran on the GPSIMD cores** (not on the host CPU).

```
HBM input ──DMA──► SBUF tile ──KERNEL_CALL──► 8 GPSIMD cores: x[p][i] += ADDEND (core k: partitions 16k..16k+15)
                                    │                                    │
                                    └──────────── SBUF tile (in place) ◄─┘ ──DMA──► HBM output
```

Neither neuronx-cc nor NKI supports GPSIMD custom operators on trn2/trn3 (`NCC_EVRF030`). The demo
uses the **GPSIMD kernel-library mechanism of the trn3 runtime/firmware**, which was reverse
engineered on this machine; see "How it works" and `../PROGRESS.md` for the evidence.

## Quick start

```bash
cd ~/trn3-gpsimd-re/demo
./run_demo.sh                      # ~20 s: builds everything, runs the control and the demo
ADDEND=-123456789 SEED=7 ITERS=2 ./run_demo.sh   # other parameters
```

Exit status 0 means every check passed. Expected output (abridged):

```
== 6. control: unmodified NKI NEFF (expect output == input)
  execution 1: 340 us, 8192/8192 int32 elements bit-exact
== control run: no GPSIMD custom-operator receipts (as expected for the unmodified NEFF)
RESULT: PASS

== 7. NKI NEFF + GPSIMD custom operator (expect output == input + 1000, on the GPSIMD cores)
  execution 1: 720 us, 8192/8192 int32 elements bit-exact
  ...
== GPSIMD execution evidence (receipts read from each GPSIMD core's data RAM over PCIe BAR0)
  TPB4  core prid  execs  partitions  input-sum   output-sum  cycles   kernel-pc   stack-ptr   called-from
           0    0      3    0..15    0x32e0b929  0x32f05929   646046  0x00010000  0x000b9a20  0x0000146b
           ...
           7    7      3  112..127   0x1bf6d467  0x1c067467   670301  0x00010000  0x000b9a20  0x0000146b
  -> 8 distinct GPSIMD cores (PRID 0..7) each read and rewrote their own SBUF partitions;
     the checksums of the values they read match the host's random input.
== GPSIMD kprintf logs (read from each core's data RAM)
  [TPB4 core 0] sbuf_add core 0: partitions 0..15 (+1000): x[0][0] -958040966 -> -958039966, ...
RESULT: PASS
```

### Environment variables of `run_demo.sh`

| Variable | Default | Meaning |
|---|---|---|
| `ADDEND` | `1000` | int32 constant added by the operator (compiled into the operator) |
| `ITERS` | `3` | executions of the modified NEFF |
| `SEED` | `12345` | seed of the pseudo-random full-range int32 input |
| `NEURON_RT_VISIBLE_CORES` | `0` | NeuronCore used by the runtime |
| `DEVICE` | `0` | Neuron device (`/dev/neuron$DEVICE`) whose GPSIMD cores the runner maps/inspects; must contain the NeuronCore above |
| `NEURON_VENV` | `/opt/aws_neuronx_venv_torch_neuronx_2_14_0` | venv with `nki` and `neuronx-cc` |
| `GPSIMD_TOOLS` | `/opt/aws/neuron/gpsimd` | Xtensa toolchain from `aws-neuronx-gpsimd-tools` |
| `NEURON_RT_LOG_LEVEL` | `ERROR` | set `DEBUG` to see the runtime load the kernel library |

## Requirements

* trn3 instance (tested: trn3.48xlarge, Ubuntu 24.04, kernel 7.0).
* Neuron SDK 2.33: `aws-neuronx-runtime-lib` 2.35.30 (libnrt + headers), `aws-neuronx-dkms` 2.31.7,
  the torch-neuronx venv with `nki` 0.7.0 and `neuronx-cc` 2.28.7631.
* `aws-neuronx-gpsimd-tools` 0.25.0.0 (Xtensa toolchain RI-2022.9 "14.09", core config `ncore2gp`).
  On Ubuntu 24.04 it needs `libtinfo5` and `libncursesw5` from the 22.04 archive.
* `g++` (C++17), `python3`.
* Read/write access to `/dev/neuron0` (default: world read/write).

The trn3-specific interfaces used here are undocumented; they were verified with exactly these
versions (GPSIMD firmware 1.25.36.0). Other versions may differ.

## Files

| File | Role |
|---|---|
| `nki/sbuf_add_nki.py` | **NKI kernel**: `[128, 64]` int32 HBM -> SBUF -> HBM, both DMAs on the GpSimd queue. `python sbuf_add_nki.py out.neff` compiles it for trn3. |
| `gpsimd/sbuf_add.cpp` | **GPSIMD custom operator** (C++): in-place `x += ADDEND` on SBUF, 16 partitions per core, writes an execution receipt and a log line per core. |
| `gpsimd/include/gpsimd.h` | device helpers: `gpsimd_core_id()`, `sbuf_ptr(partition, offset)`, `sbuf_u32/sbuf_i32`. |
| `gpsimd/include/kprint.h` | `kprintf()` for GPSIMD code (per-core ring buffer in data RAM, read by the host). |
| `gpsimd/include/neuron_kl.h` | `NEURON_KERNEL(name)`: registers a kernel entry point in the library. |
| `gpsimd/build_kernel_lib.sh` | builds a `.cpp`/`.c` file into a trn3 GPSIMD **kernel library** (`xt-clang++`, `lsp/`, `klpatch.py`). |
| `tools/neff_tool.py` | **NEFF modification**: `info` (find the SBUF tile), `inject` (add the library + KERNEL_CALL). |
| `runner/run_neff.cpp` | **runner executable**: libnrt load/execute, SBUF mapping, bit-exact validation, GPSIMD receipts and logs. |
| `run_demo.sh` | builds and runs everything (steps 0-7). |
| `build/` | outputs: `sbuf_add_nki.neff`, `sbuf_add.so`, `sbuf_add_gpsimd.neff`, `run_neff`. |

The steps can be run individually:

```bash
V=/opt/aws_neuronx_venv_torch_neuronx_2_14_0
PATH=$V/bin:$PATH $V/bin/python nki/sbuf_add_nki.py build/sbuf_add_nki.neff         # 1
python3 tools/neff_tool.py info build/sbuf_add_nki.neff                               # 2
gpsimd/build_kernel_lib.sh gpsimd/sbuf_add.cpp build/sbuf_add.so \
    -DTILE_SBUF_OFFSET=0x3fe00u -DTILE_PARTITIONS=128u -DTILE_ELEMS=64u -DADDEND=1000    # 3
python3 tools/neff_tool.py inject build/sbuf_add_nki.neff build/sbuf_add.so sbuf_add \
    build/sbuf_add_gpsimd.neff                                                           # 4
g++ -std=c++17 -O2 -o build/run_neff runner/run_neff.cpp -I/opt/aws/neuron/include \
    -L/opt/aws/neuron/lib -lnrt -Wl,-rpath,/opt/aws/neuron/lib                          # 5
NEURON_RT_VISIBLE_CORES=0 build/run_neff build/sbuf_add_nki.neff --expect identity     # 6
NEURON_RT_VISIBLE_CORES=0 build/run_neff build/sbuf_add_gpsimd.neff --expect add:1000 --iters 3 --logs  # 7
```

## How it works

### 1. NKI kernel -> NEFF
The NKI kernel only moves data. neuronx-cc places the tile at SBUF offset `0x3FE00` of
partitions 0..127 and emits both DGE DMAs on the Pool (GpSimd) engine; its program is
`SET_ORDERING_MODE, DMA(HBM->SBUF), DMA(SBUF->HBM), NOP`. `neff_tool.py info` decodes the DMA
instructions to find the tile, and `run_demo.sh` compiles that location into the operator.

### 2. GPSIMD custom operator -> kernel library
The trn3 runtime/firmware loads GPSIMD code as *kernel libraries*
(validated by `nrtucode_kl_create_from_buffer` in `libnrtucode_extisa.so`):
* Xtensa ELF linked with a position-independent split-load layout (`gpsimd/lsp`, derived from
  Cadence's `pisplitload`): code at `0x01000000`, data at `0x02000000`, dynamic information at
  `0x03000000`; the loader relocates them into GPSIMD instruction/data RAM.
* a GNU build-id note, and one ELF note per kernel: owner `"Neuron"`, type 2,
  `{u32 entry; char name[]}` (`NEURON_KERNEL(sbuf_add)`; `klpatch.py` fills in the entry address).
* `_start(ctx)` is called once per load by the firmware; the kernel itself is called on all 8
  GPSIMD cores for every KERNEL_CALL (`rsr.prid` gives the core id).
* Code must be **plain base-ISA code**: the shipped `ncore2gp` toolchain describes a different
  Xtensa configuration than the trn3 GPSIMD core; its FLIX/vector (Vision TIE) and FP instructions
  do not run on trn3, so `build_kernel_lib.sh` compiles with `-mno-flix -mno-coproc` and rejects
  any FLIX bundle. No C++ runtime (exceptions, RTTI, static constructors, heap) is available.

### 3. NEFF modification (`tools/neff_tool.py inject`)
* `sg00/Pool0.json` gets `"gpsimd_kernels_def": "gpsimd_kernels_def.json"`.
* `sg00/gpsimd_kernels_def.json` = `[{"name": "sbuf_add", "lib": "/sg00/sbuf_add.so"}]`, and the
  library is added to the NEFF.
* A `KERNEL_CALL` instruction (opcode `0x85`, 64 bytes, byte 12 = index into
  `gpsimd_kernels_def`) is inserted after the load DMA. It waits for the load DMA's semaphore
  (taken over from the store DMA) and increments semaphore 3 on completion; the store DMA now
  waits for semaphore 3. The NEFF header and tar payload are rewritten (sizes, MD5).

### 4. SBUF access from the GPSIMD cores
The GPSIMD cores reach SoC memory through an address-translation block (TPB + `0x3062000`,
40 entries `{ctrl, mask, local, soc}`). After `nrt_load` the runner writes **entry 38** over PCIe
BAR0 (driver ioctl `NEURON_IOCTL_BAR_WRITE`) so that GPSIMD core address
`0xD0000000 + p * 0x40000 + offset` maps SBUF partition `p` (SoC `0x8020_0000_0000` for TPB 4).
It maps every TPB of the device whose GPSIMD firmware is running and clears the entry at the end.

### 5. Validation and proof of GPSIMD execution (`runner/run_neff.cpp`)
* Input: seeded pseudo-random full-range int32 values; reference: `input + ADDEND` (two's
  complement wrap-around) computed on the host; the output is compared with `memcmp` after every
  execution.
* Each GPSIMD core writes a 64-byte **receipt** into its own data RAM. The runner reads all 8
  receipts over PCIe BAR0 (GPSIMD core `k` memory = TPB + `0x3100000 + k*0x100000`) and checks:
  - the receipt found in core `k`'s memory carries processor id `k` (hardware `PRID` register),
    and all 8 cores are present;
  - the execution counter equals the number of `nrt_execute` calls;
  - the partitions processed by the cores cover the tile exactly once;
  - the checksum of the values each core **read from SBUF** equals the host's checksum of the
    random input for those partitions, and likewise for the values it wrote;
  - the code address lies in GPSIMD instruction RAM, the stack in GPSIMD data RAM, and the core
    cycle counter advanced. The caller address (the GPSIMD firmware dispatcher, `0x146b` with
    this firmware) is printed in the table for reference.
* **Control**: the unmodified NKI NEFF must return the input unchanged and leave no receipts.
* Independent evidence from the Neuron runtime: run with `NEURON_RT_LOG_LEVEL=DEBUG` and look for
  ```
  gpsimd_kernels_def: engine json 'Pool0.json' key found, points to 'gpsimd_kernels_def.json'
  Initialized kernel library 'sbuf_add.so': 1904B IRAM, 1264B DRAM
  nrtucode: ll_builder[ND[0]NC[4].Q7_POOL[0]: 1 kernel libraries, IRAM 1879/65536 bytes ...
  ```

## Writing your own operator

* Start from `gpsimd/sbuf_add.cpp`. Declare entry points `extern "C" void name()` and register
  them with `NEURON_KERNEL(name);`; keep `extern "C" void *_start(void *)`.
* Every kernel runs on all 8 cores: split the work with `gpsimd_core_id()`
  (`SBUF_PARTITIONS_PER_CORE` = 16).
* Access SBUF through `sbuf_ptr(partition, byte_offset)` (needs the runner's mapping step);
  `kprintf()` output appears with `run_neff --logs`.
* Integer arithmetic only for now (no FP, no vector instructions); data RAM for the library is
  limited to 8 KB (the log buffer uses 1 KB), instruction RAM to 64 KB shared with builtins.
* The KERNEL_CALL's operand bytes are not yet readable by the kernel, so tile locations and
  parameters are compile-time constants (`run_demo.sh` derives them from the NEFF).

## Limitations

* Relies on undocumented, reverse-engineered runtime/firmware/hardware interfaces of the versions
  listed above; not supported by AWS.
* `neff_tool.py inject` handles NKI kernels whose Pool program has exactly one HBM->SBUF DMA
  followed by one SBUF->HBM DMA of the same tile (the shape produced by `sbuf_add_nki.py`).
* The SBUF mapping is a host-side step between `nrt_load` and `nrt_execute` that writes a GPSIMD
  address-translation entry over PCIe BAR0; run one demo process at a time.
* Performance is not the goal: each core accesses SBUF with scalar 32-bit loads/stores through the
  translation window (~600 k core cycles for 32 KB).

## Troubleshooting

* `nrt_execute failed: NRT_STATUS=5` (timeout): the GPSIMD kernel hung, e.g. it touched an
  unmapped address. Without `--no-map` this should not happen; check that `DEVICE` matches
  `NEURON_RT_VISIBLE_CORES` and that the runner printed `[map] ... on TPB 4 5` (or other TPBs).
  After a hang the cores' logs can still be read with `../tools/klog.py --tpb 4`.
* `error: ... contains FLIX bundles`: the compiler emitted bundled (FLIX) instructions that the trn3
  GPSIMD core cannot execute. Remove floating-point or vector code; inspect the library with
  `xt-objdump -d build/sbuf_add.so` (with the environment set by `build_kernel_lib.sh`).
* `License checkout failed`: `LM_LICENSE_FILE` must point to the gpsimd-tools license
  (`build_kernel_lib.sh` sets it).
* The NKI step prints nothing about execution: NKI's standalone path compiles the NEFF and then
  tries to run it through `nrtpy`, which is not installed; `sbuf_add_nki.py` only keeps the NEFF.
