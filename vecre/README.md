# vecre: reverse engineering the trn3 GPSIMD vector (FLIX / IVP) instructions

Goal: find the encodings and semantics of the Xtensa vector instructions of the trn3 GPSIMD cores, so that
custom operators can use the 512-bit vector unit instead of base-ISA scalar code only.

## Findings

| # | Finding | How |
|---|---|---|
| 1 | `ncore2gp`, the only Xtensa config in the SDK (Vision Q7, `XCHAL_VISION_TYPE 7`, NX1.1.4, config ID `0xC4019686_2908E4E3`), is the **trn1** (NeuronCore-v2) GPSIMD core. trn2 and trn3 use different, unshipped configs. | the trn1 built-in library decodes cleanly with it; trn2/trn3 kernels do not |
| 2 | `libnrtucode_extisa.so` embeds 48 Xtensa ELFs: the built-in Pool kernels (iota, cross_lane_reduce, pool, tensor_tensor, tensor_reduce, get_sequence_bounds, nonzero_with_count, tensor_dequantize, topk, gather, tensor_scalar_affine_select, sort, tensor_scalar, tensor_scalar_addr, rdma, activate) for arch ids 0x0d (15 kernels, presumably trn2), 0x15 and 0x1d (16 kernels each, trn3 and trn3-pre; nearly identical), plus one large trn1 library (`iota_kernel`, `tensor_tensor_arith`, ...). Selected by `nrtucode_get_ext_isa(arch)`. | `re/ucode_elfs/`, `modules.json`; the arch 0x1d code bytes are what the trn3 IRAM dump contains |
| 3 | The vector coprocessor is enabled while our operators run: `CPENABLE = 0x7f`. ncore2gp vector code fails on trn3 because the **encodings differ**, not because the unit is off. | `toolchain/probes/vec` phase 0 |
| 4 | Bundle layouts are partly shared with trn1: forcing ncore2gp format N0 on trn3 `2f45..5032` bundles decodes slots 0..2 as `nop`, but the vector opcodes in slot 3 do not decode. Vector opcodes are renumbered. | `xforce` |
| 5 | **trn3 instruction length rule** (device-measured, byte-granular; 274/295 functions then tile from `entry` to `retw`): op0 0..7: 3 B; 8..13: 2 B; 0xE: 16 B; 0xF: byte 3 bits [1:0] = `01` -> 8 B, `11` -> 10 B, `00`/`10` -> 16 B. (trn1: op0 0xF -> 8 B if byte 3 bit 0 clear, else 16 B.) | `lenprobe.py`, `lenprobe2.py` |
| 6 | trn3 code: 4885 distinct bundles in the built-in kernels + firmware (2711 x 8 B, 226 x 10 B, 1948 x 16 B). | `all_bundles.json` |
| 7 | **Vector loads/stores** (8-byte format `?f <v><a> 40 2d ...`; byte 1 = vector register (high nibble) and address register (low nibble)): `1f va 40 2d 22 e8 12 41` load with `a += 64`, `1f va 40 2d 44 e8 12 41` store with `a += 64`, `0f va 40 2d 00 ee 0a 41` load, `0f va 40 2d 00 fa 0a 41` store. 64 bytes; work on local data RAM and directly on the SBUF window. **Addresses must be 64-byte aligned**: misaligned ones raise EXCCAUSE 0x2102 (load) / 0x3102 (store); NKI tiles are only 32-byte aligned. | sweep of real bundles; load/store round trips |
| 8 | **ALU format `2f 45 b2 b3 b4 b5 50 32`** (slots 0..2 nop, slot 3 vector op): va = bits 17..21, vd = bits 22,23,26,27,28, vb = bits 35..39 (5-bit register fields), opcode = b3[7:5], b4[2:0], b5. 31 operations decoded and verified with random registers (add/sub 8/16/32, and/or/xor, min/max signed/unsigned 8/16/32, unsigned average with/without rounding, absolute difference, shifts by immediates); `toolchain/include/gpsimd/vec.h` | `opsweep.py`, `verify_vec_h.py` |
| 9 | Some instructions read hidden state left behind by earlier ones (e.g. `0ff239a1e98dc030` looked like a 32x32 unsigned high multiply in one context and differently in another) -> state registers (shift amount, alignment, vector booleans) still to be identified. | determinism checks |
| 10 | **Compiler pitfall**: xt-clang aligns loop bodies with ncore2gp FLIX nop bundles (`2f354008006fa502`), which trn3 decodes as 16-byte bundles -> garbage. `gpsimd-cc` now builds with `-mno-target-align` and rejects FLIX nop padding. (Builds without vector code were protected by the FLIX check already.) | `vec_add_i32` hang |
| 11 | **First vector operator** `toolchain/examples/vec_add_i32`: c = a + b (int32, 128 x 512), staged through 64-byte-aligned local RAM: add loop 2,585 vs 49,170 cycles per core (19x), whole operator 12,232 vs 58,954 cycles (4.8x), bit-exact. | device |

## Encodings (trn3, verified on the device)

```text
load  v<v> <- [a<a>], a += 64     1f <v><a> 40 2d 22 e8 12 41
store v<v> -> [a<a>], a += 64     1f <v><a> 40 2d 44 e8 12 41
load  v<v> <- [a<a>]              0f <v><a> 40 2d 00 ee 0a 41
store v<v> -> [a<a>]              0f <v><a> 40 2d 00 fa 0a 41
ALU   vd = va OP vb               2f 45 b2 b3 b4 b5 50 32
      b2 = va<<1 | vd[0]<<6 | vd[1]<<7;  b3 = 1 | vd[4:2]<<2 | op3<<5;  b4 = vb<<3 | lo;  b5 = op
      (op3, lo, op): add8/16/32 (5,1,8b/8c/8d)  sub8/16/32 (7,1,84)/(6,1,85)/(7,1,85)  and (1,1,a0)
      or (6,1,84)  xor (6,1,87)  maxs8/16/32 (1,1,a9/ad/af)  maxu8/16/32 (1,1,b3/b4/b5)
      mins8/16/32 (1,1,b7/bb/bd)  minu8/16/32 (7,1,80)/(6,1,81)/(7,1,81)  avgu8/16 (1,1,a7/a8)
      avgru8/16 (1,1,a5/a6)  absdu8/16 (5,1,89/8a)  srli32/srai32 (2,1,a6/a4; imm in va)
      slli32 (3,1,a0; imm in va)  srli16 (2,1,ac; va = imm[1:0] | imm[3:2]<<3)
```

## Next

- the remaining opcode groups of the ALU format (b4[2:0] != 1, other b6/b7), the ~250 consistent but
  unexplained results (saturating ops, multiplies, shuffles), the 16-byte formats;
- state registers (vector booleans, alignment registers, shift amount) and unaligned loads;
- moves between address registers and vector registers (`broadcast` seen: `0ff2f70d012d5643`);
- trace the built-in kernels (topk, iota, tensor_tensor) with the decoded instructions.

## The sandbox (`toolchain/probes/vec`, host side `vrun.py`)

A GPSIMD operator that executes arbitrary instruction bytes on core 0 and reports what they did:

- the test bytes (up to 56) are written into a stub in instruction RAM (our `.text`, 32-bit stores + `isync`);
  the stub loads a3..a15 from a context block (pointers into a 16 KB scratch area unless set), moves a0 and a2
  into scratch too, runs the bytes, stores a3..a15 back, clears `LCOUNT`, and returns;
- **exceptions are survived**: the operator installs its own handler in the firmware's XTOS exception table
  (data RAM `0x81b40`, 9 entries, entry 0 `xtos_default_exc_handler` at `0xd99c`, 1..8 the firmware handler
  at `0x30c0`); the handler records EXCCAUSE/EPC/EXCVADDR and resumes at the stub's epilogue;
- **hangs are survived** on the host: each test row is marked "started" in SBUF before it runs; after an
  `NRT_TIMEOUT` the host reads the output tile over BAR0, blames the started row (or the last completed one),
  and re-queues the rest; `NEURON_RT_EXEC_TIMEOUT` is lowered for sandbox runs;
- **integrity checks** after every test: firmware vectors (IRAM 0..0x2000), the harness code, the exception
  table and 1 KB guard zones around the scratch must be unchanged, else the run stops with status 5.

Result per test: status (ok / exception / hang / corrupts), EXCCAUSE, exception pc, register changes,
scratch bytes changed (first/last offset and 128 bytes of content).

## Tools

| Tool | Purpose |
|---|---|
| `t3dis <bin> <start> <end>` | trn3 disassembler skeleton: base ISA decoded, FLIX bundles split by the learned rule |
| `t3split.py` | same split in Python |
| `xforce <fmt> <hex>` | decode bytes with a forced ncore2gp format, slot by slot |
| `isa_formats` | list ncore2gp FLIX formats, lengths, slot bit maps, format-id bits |
| `vrun.py <hex> ...` | run test bundles in the sandbox and print the effects |
| `lenprobe.py`, `lenprobe2.py` | measure bundle lengths on the device |
| `sweep.py <lens> <out.jsonl>` | run all distinct bundles of the given lengths (resumable) |
| `sweepv.py`, `infer.py` | the same in vector mode (v0..v15 preloaded, dumped after), and the operation inference |
| `fieldmap.py <hex>` | flip single bits of a bundle and report the effect |
| `opsweep.py` | sweep the opcode space of the `2f45..5032` ALU format |
| `verify_vec_h.py` | verify every ALU macro of `gpsimd/vec.h` on the device |
