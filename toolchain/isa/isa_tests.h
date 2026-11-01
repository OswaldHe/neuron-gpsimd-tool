/* isa_tests.h - operation tests compiled both for the trn3 GPSIMD core and for the host (x86),
 * so results can be compared bit-for-bit. Each test fills out[] from fixed inputs. */
#pragma once
#include <stdint.h>
#ifdef __XTENSA__
#define DEV 1
#else
#define DEV 0
#endif
static const uint32_t A[8] = {0x00000007u, 0xFFFFFFF9u, 0x7FFFFFFFu, 0x80000000u, 0x12345678u, 0xDEADBEEFu, 0x00010001u, 0x0000FFFFu};
static const uint32_t B[8] = {0x00000003u, 0x00000005u, 0xFFFFFFFFu, 0x00000002u, 0x0000007Bu, 0x00000010u, 0xFFFF0003u, 0x00000011u};
typedef void (*test_fn)(uint32_t *out);   /* writes 8 words */
#define T(name) static void __attribute__((unused)) name(uint32_t *o)
#define FOR8 for (int i = 0; i < 8; i++)
static inline uint32_t opq(uint32_t x) { __asm__ volatile("" : "+r"(x)); return x; }   /* defeat constant folding */

/* ---- group 1: integer */
T(t_add)  { FOR8 o[i] = opq(A[i]) + opq(B[i]); }
T(t_sub)  { FOR8 o[i] = opq(A[i]) - opq(B[i]); }
T(t_mul)  { FOR8 o[i] = opq(A[i]) * opq(B[i]); }
T(t_mulhu){ FOR8 o[i] = (uint32_t)(((uint64_t)opq(A[i]) * opq(B[i])) >> 32); }
T(t_mulhs){ FOR8 o[i] = (uint32_t)(((int64_t)(int32_t)opq(A[i]) * (int32_t)opq(B[i])) >> 32); }
T(t_mul16){ FOR8 o[i] = (uint32_t)((int32_t)(int16_t)opq(A[i]) * (int32_t)(int16_t)opq(B[i])); }
T(t_divu) { FOR8 o[i] = opq(A[i]) / (opq(B[i]) | 1u); }
T(t_divs) { FOR8 { int32_t b = (int32_t)(opq(B[i]) | 1u); int32_t a = (int32_t)opq(A[i]); o[i] = (b == -1 && a == INT32_MIN) ? 0u : (uint32_t)(a / b); } }
T(t_remu) { FOR8 o[i] = opq(A[i]) % (opq(B[i]) | 1u); }
T(t_rems) { FOR8 { int32_t b = (int32_t)(opq(B[i]) | 1u); int32_t a = (int32_t)opq(A[i]); o[i] = (b == -1) ? 0u : (uint32_t)(a % b); } }
T(t_shl)  { FOR8 o[i] = opq(A[i]) << (opq(B[i]) & 31); }
T(t_shru) { FOR8 o[i] = opq(A[i]) >> (opq(B[i]) & 31); }
T(t_shrs) { FOR8 o[i] = (uint32_t)((int32_t)opq(A[i]) >> (opq(B[i]) & 31)); }
T(t_rotl) { FOR8 { uint32_t s = opq(B[i]) & 31; o[i] = s ? (opq(A[i]) << s) | (opq(A[i]) >> (32 - s)) : opq(A[i]); } }
T(t_logic){ FOR8 o[i] = (opq(A[i]) & opq(B[i])) ^ (opq(A[i]) | ~opq(B[i])); }
T(t_minmax){ FOR8 { int32_t a = (int32_t)opq(A[i]), b = (int32_t)opq(B[i]); o[i] = (uint32_t)(a < b ? a : b) ^ (opq(A[i]) > opq(B[i]) ? opq(A[i]) : opq(B[i])); } }
T(t_abs)  { FOR8 { int32_t a = (int32_t)opq(A[i]); o[i] = a == INT32_MIN ? (uint32_t)a : (uint32_t)(a < 0 ? -a : a); } }
T(t_clamp){ FOR8 { int32_t a = (int32_t)opq(A[i]); o[i] = (uint32_t)(a > 32767 ? 32767 : a < -32768 ? -32768 : a); } }
T(t_sext) { FOR8 o[i] = (uint32_t)(int32_t)(int8_t)opq(A[i]) + (uint32_t)(int32_t)(int16_t)opq(B[i]); }
T(t_clz)  { FOR8 { uint32_t a = opq(A[i]); int n = 0; if (!a) n = 32; else while (!(a & 0x80000000u)) { a <<= 1; n++; } o[i] = (uint32_t)n; } }
T(t_cmp)  { FOR8 o[i] = ((int32_t)opq(A[i]) < (int32_t)opq(B[i])) | ((opq(A[i]) < opq(B[i])) << 1) | ((opq(A[i]) == opq(B[i])) << 2); }
T(t_sel)  { FOR8 o[i] = (opq(A[i]) & 1) ? opq(B[i]) : ~opq(B[i]); }
T(t_bitf) { FOR8 o[i] = ((opq(A[i]) >> 5) & 0x3FFu) | ((opq(B[i]) & 0xFFu) << 20); }
T(t_add64){ FOR8 { uint64_t a = ((uint64_t)opq(A[i]) << 32) | opq(B[i]), b = ((uint64_t)opq(B[i]) << 32) | opq(A[i]); uint64_t r = a + b; o[i] = (uint32_t)(r >> 32) ^ (uint32_t)r; } }
T(t_mul64){ FOR8 { uint64_t a = ((uint64_t)opq(A[i]) << 32) | opq(B[i]), b = ((uint64_t)opq(B[i]) << 32) | opq(A[i]); uint64_t r = a * b; o[i] = (uint32_t)(r >> 32) ^ (uint32_t)r; } }
T(t_sh64) { FOR8 { uint64_t a = ((uint64_t)opq(A[i]) << 32) | opq(B[i]); uint32_t s = opq(B[i]) & 63; o[i] = (uint32_t)(a << s) ^ (uint32_t)((a >> s) >> 7); } }
T(t_loop) { uint32_t acc = 0; for (uint32_t k = 0; k < opq(1000u); k++) acc = acc * 1664525u + 1013904223u + k; FOR8 o[i] = acc + (uint32_t)i; }

#define GROUP1 {"add", t_add}, {"sub", t_sub}, {"mul", t_mul}, {"mulhu", t_mulhu}, {"mulhs", t_mulhs}, {"mul16", t_mul16}, \
  {"divu", t_divu}, {"divs", t_divs}, {"remu", t_remu}, {"rems", t_rems}, {"shl", t_shl}, {"shru", t_shru}, {"shrs", t_shrs}, \
  {"rotl", t_rotl}, {"logic", t_logic}, {"minmax", t_minmax}, {"abs", t_abs}, {"clamp", t_clamp}, {"sext", t_sext}, {"clz", t_clz}, \
  {"cmp", t_cmp}, {"select", t_sel}, {"bitfield", t_bitf}, {"add64", t_add64}, {"mul64", t_mul64}, {"shift64", t_sh64}, {"loop", t_loop}

/* ---- group 2: single-precision floating point (bit patterns compared exactly) */
static const uint32_t FA[8] = {0x3FC00000u /*1.5*/, 0xC0100000u /*-2.25*/, 0x000116C2u /*denormal*/, 0x7F7FFFFFu /*max*/,
                               0x7F800000u /*inf*/, 0x3EAAAAABu /*1/3*/, 0x40490FDBu /*pi*/, 0xB892CCF7u /*-7e-5*/};
static const uint32_t FB[8] = {0x40000000u /*2*/, 0x3F800000u /*1*/, 0x00800000u /*min normal*/, 0x7F7FFFFFu /*max*/,
                               0x7F800000u /*inf*/, 0x41200000u /*10*/, 0xBF000000u /*-0.5*/, 0x501502F9u /*1e10*/};
static inline float uf(uint32_t u) { union { uint32_t u; float f; } x = {opq(u)}; return x.f; }
static inline uint32_t fu(float f) { union { float f; uint32_t u; } x = {f}; return x.u; }
T(t_fadd) { FOR8 o[i] = fu(uf(FA[i]) + uf(FB[i])); }
T(t_fsub) { FOR8 o[i] = fu(uf(FA[i]) - uf(FB[i])); }
T(t_fmul) { FOR8 o[i] = fu(uf(FA[i]) * uf(FB[i])); }
T(t_fdiv) { FOR8 o[i] = fu(uf(FA[i]) / uf(FB[i])); }
T(t_fma)  { FOR8 o[i] = fu(__builtin_fmaf(uf(FA[i]), uf(FB[i]), uf(0x3F800000u))); }
T(t_fsqrt){ FOR8 o[i] = fu(__builtin_sqrtf(uf(FA[i] & 0x7FFFFFFFu))); }
T(t_fneg) { FOR8 o[i] = fu(-uf(FA[i])) ^ fu(__builtin_fabsf(uf(FB[i]))); }
T(t_fcmp) { FOR8 o[i] = (uf(FA[i]) < uf(FB[i])) | ((uf(FA[i]) <= uf(FB[i])) << 1) | ((uf(FA[i]) == uf(FB[i])) << 2) | ((uf(FA[i]) != uf(FA[i])) << 3); }
T(t_f2i)  { FOR8 { float f = uf(FA[i]) * 1000.0f; o[i] = (f > 2e9f || f < -2e9f || f != f) ? 0x7FFFFFFFu : (uint32_t)(int32_t)f; } }
T(t_i2f)  { FOR8 o[i] = fu((float)(int32_t)opq(A[i])) ^ fu((float)opq(B[i])); }
T(t_fminmax){ FOR8 { float a = uf(FA[i]), b = uf(FB[i]); o[i] = fu(a < b ? a : b) ^ fu(a > b ? a : b); } }
T(t_fnan) { o[0] = fu(uf(0x7F800000u) - uf(0x7F800000u)); o[1] = fu(uf(0) / uf(0)); o[2] = fu(__builtin_sqrtf(uf(0xBF800000u)));
            o[3] = fu(uf(0x7FC00001u) + 1.0f); o[4] = fu(uf(0xFFC00002u) * 2.0f); o[5] = fu(uf(0x80000000u) + uf(0)); o[6] = fu(uf(0x80000000u) * 3.0f); o[7] = fu(1.0f / uf(0x80000000u)); }
#define GROUP2 {"fadd", t_fadd}, {"fsub", t_fsub}, {"fmul", t_fmul}, {"fdiv", t_fdiv}, {"fma", t_fma}, {"fsqrt", t_fsqrt}, {"fneg_abs", t_fneg}, \
  {"fcmp", t_fcmp}, {"f2i", t_f2i}, {"i2f", t_i2f}, {"fminmax", t_fminmax}, {"fnan", t_fnan}
#define GROUP2A {"fadd", t_fadd}, {"fsub", t_fsub}, {"fmul", t_fmul}, {"fneg_abs", t_fneg}, {"fcmp", t_fcmp}, {"f2i", t_f2i}, \
  {"i2f", t_i2f}, {"fminmax", t_fminmax}
