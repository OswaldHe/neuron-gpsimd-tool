// gpsimd/vec.h - trn3 GPSIMD vector (512-bit) instructions as raw bundles.
//
// The SDK ships no Xtensa configuration for the trn3 core, so the vector instructions were reverse
// engineered on the device (vecre/README.md). Each macro expands to a string of `.byte` directives for
// one 8-byte FLIX bundle; register numbers are assembler expressions (compile-time constants).
// Use the macros inside __asm__ volatile(...) and bind every address register you name to a C variable
// with `register T *p __asm__("aN")`. Build the operator with CFLAGS = ["--allow-vector"].
//
// Registers: 512 bits = 64 bytes = 16 x int32 = 32 x int16 = 64 x int8. The ALU fields are 5 bits wide
// (v0..v31), the load/store fields 4 bits (v0..v15). Vector loads/stores move 64 bytes and work on local
// data RAM and on the SBUF window (0xD0000000 + partition * 0x40000 + offset).
//
// Verified on trn3 (Neuron SDK 2.33, GPSIMD firmware 1.25.36) in the sandbox (toolchain/probes/vec,
// vecre/verify_vec_h.py): every ALU macro with random registers in v0..v15 (4 cases each, every shift
// amount for GPV_SRLI16) against numpy; loads/stores round-trip local RAM and SBUF. v16..v31 untested.
#pragma once

// ---- loads / stores: "1f <v><a> 40 2d ..": v and a are 4-bit fields (v0..v15, a0..a15) ----
// v <- [a]; a += 64
#define GPV_LD_IP(v, a) ".byte 0x1f, ((" #v ")<<4)|(" #a "), 0x40, 0x2d, 0x22, 0xe8, 0x12, 0x41\n"
// [a] <- v; a += 64
#define GPV_ST_IP(v, a) ".byte 0x1f, ((" #v ")<<4)|(" #a "), 0x40, 0x2d, 0x44, 0xe8, 0x12, 0x41\n"
// [a] <- v (no update)
#define GPV_ST(v, a) ".byte 0x0f, ((" #v ")<<4)|(" #a "), 0x40, 0x2d, 0x00, 0xfa, 0x0a, 0x41\n"
// v <- [a] (no update)
#define GPV_LD(v, a) ".byte 0x0f, ((" #v ")<<4)|(" #a "), 0x40, 0x2d, 0x00, 0xee, 0x0a, 0x41\n"

// ---- two-operand ALU ops in the 8-byte "2f 45 .. .. .. .. 50 32" format (slots 0..2 are nop) ----
// fields: va = bits 17..21, vd = bits 22,23,26,27,28, vb = bits 35..39 (5 bits each: v0..v31);
// opcode = byte3[7:5] (op3), byte4[2:0] (lo), byte5.
#define GPV__ALU(d, a, b, op3, lo, op)                                                                    \
    ".byte 0x2f, 0x45, (((" #a ")&31)<<1)|(((" #d ")&1)<<6)|((((" #d ")>>1)&1)<<7), "                    \
    "1|((((" #d ")>>2)&7)<<2)|((" #op3 ")<<5), ((((" #b ")&31)<<3)|(" #lo ")), " #op ", 0x50, 0x32\n"
// lane-wise d = a OP b  (suffix = lane width in bits)
#define GPV_ADD8(d, a, b)   GPV__ALU(d, a, b, 5, 1, 0x8b)
#define GPV_ADD16(d, a, b)  GPV__ALU(d, a, b, 5, 1, 0x8c)
#define GPV_ADD32(d, a, b)  GPV__ALU(d, a, b, 5, 1, 0x8d)
#define GPV_SUB8(d, a, b)   GPV__ALU(d, a, b, 7, 1, 0x84)
#define GPV_SUB16(d, a, b)  GPV__ALU(d, a, b, 6, 1, 0x85)
#define GPV_SUB32(d, a, b)  GPV__ALU(d, a, b, 7, 1, 0x85)
#define GPV_AND(d, a, b)    GPV__ALU(d, a, b, 1, 1, 0xa0)
#define GPV_OR(d, a, b)     GPV__ALU(d, a, b, 6, 1, 0x84)
#define GPV_XOR(d, a, b)    GPV__ALU(d, a, b, 6, 1, 0x87)
#define GPV_MAXS8(d, a, b)  GPV__ALU(d, a, b, 1, 1, 0xa9)
#define GPV_MAXS16(d, a, b) GPV__ALU(d, a, b, 1, 1, 0xad)
#define GPV_MAXS32(d, a, b) GPV__ALU(d, a, b, 1, 1, 0xaf)
#define GPV_MAXU8(d, a, b)  GPV__ALU(d, a, b, 1, 1, 0xb3)
#define GPV_MAXU16(d, a, b) GPV__ALU(d, a, b, 1, 1, 0xb4)
#define GPV_MAXU32(d, a, b) GPV__ALU(d, a, b, 1, 1, 0xb5)
#define GPV_MINS8(d, a, b)  GPV__ALU(d, a, b, 1, 1, 0xb7)
#define GPV_MINS16(d, a, b) GPV__ALU(d, a, b, 1, 1, 0xbb)
#define GPV_MINS32(d, a, b) GPV__ALU(d, a, b, 1, 1, 0xbd)
#define GPV_MINU8(d, a, b)  GPV__ALU(d, a, b, 7, 1, 0x80)
#define GPV_MINU16(d, a, b) GPV__ALU(d, a, b, 6, 1, 0x81)
#define GPV_MINU32(d, a, b) GPV__ALU(d, a, b, 7, 1, 0x81)
#define GPV_AVGU8(d, a, b)  GPV__ALU(d, a, b, 1, 1, 0xa7)   // (a + b) >> 1, unsigned
#define GPV_AVGU16(d, a, b) GPV__ALU(d, a, b, 1, 1, 0xa8)
#define GPV_AVGRU8(d, a, b) GPV__ALU(d, a, b, 1, 1, 0xa5)   // (a + b + 1) >> 1, unsigned
#define GPV_AVGRU16(d, a, b) GPV__ALU(d, a, b, 1, 1, 0xa6)
#define GPV_ABSDU8(d, a, b) GPV__ALU(d, a, b, 5, 1, 0x89)   // |a - b|, unsigned
#define GPV_ABSDU16(d, a, b) GPV__ALU(d, a, b, 5, 1, 0x8a)
// shifts by an immediate: d = b SHIFT imm (the immediate sits in the va field, 0..31)
// 16-bit shift: the va field holds imm[1:0] in bits 0..1 and imm[3:2] in bits 3..4 (bit 2 selects other ops)
#define GPV_SRLI16(d, b, imm) GPV__ALU(d, (((imm)&3)|((((imm)>>2)&3)<<3)), b, 2, 1, 0xac)
#define GPV_SRLI32(d, b, imm) GPV__ALU(d, imm, b, 2, 1, 0xa6)
#define GPV_SRAI32(d, b, imm) GPV__ALU(d, imm, b, 2, 1, 0xa4)
#define GPV_SLLI32(d, b, imm) GPV__ALU(d, imm, b, 3, 1, 0xa0)
