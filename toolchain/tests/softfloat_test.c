/* Host validation of runtime/softfloat.c against x86 SSE/FMA/F16C hardware (bit-exact). */
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "../include/gpsimd/softfloat.h"

static inline float F(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static inline uint32_t U(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng >> 16); }
static const uint32_t SPECIAL[] = {0, 0x80000000u, 0x3F800000u, 0xBF800000u, 0x7F800000u, 0xFF800000u, 0x7FC00000u, 0xFFC00000u,
    0x7F800001u, 0xFFA00005u, 0x7FC12345u, 0x00000001u, 0x80000001u, 0x007FFFFFu, 0x00800000u, 0x80800000u, 0x7F7FFFFFu,
    0xFF7FFFFFu, 0x3F000000u, 0x3FFFFFFFu, 0x40000000u, 0x4B000000u, 0x4AFFFFFFu, 0x4B7FFFFFu, 0x3EFFFFFFu, 0x3F7FFFFFu,
    0x00400000u, 0x34000000u, 0x33800000u, 0x4F000000u, 0xCF000000u, 0x5F000000u, 0xDF000000u, 0x4EFFFFFFu, 0x3F800001u};
#define NS (sizeof(SPECIAL) / sizeof(SPECIAL[0]))
static uint32_t pick(int mode) {
    switch (mode) {
        case 0: return SPECIAL[rnd() % NS];
        case 1: return rnd();                                            /* any bit pattern */
        case 2: return (rnd() & 0x807FFFFFu) | ((100 + rnd() % 56) << 23); /* normal, moderate exponents */
        default: { uint32_t x = (rnd() & 0x807FFFFFu) | ((120 + rnd() % 8) << 23); return x; }  /* close exponents: cancellation */
    }
}
static long fails;
static void report(const char *op, uint32_t a, uint32_t b, uint32_t c, uint32_t got, uint32_t exp) {
    if (got != exp && fails++ < 12) printf("  %s(%08x, %08x, %08x): soft %08x  x86 %08x\n", op, a, b, c, got, exp);
}
int main(void) {
    long n = 0;
    const char *names[] = {"add", "sub", "mul", "div", "sqrt", "fma", "min", "max", "cmp", "i32", "i64", "u32", "f_i32", "f_u32",
                           "f_i64", "floor", "ceil", "trunc", "rint", "round", "f16", "from_f16"};
    long per[22] = {0};
    for (long it = 0; it < 1500000; it++) {
        uint32_t a = pick(it % 4), b = pick((it / 4) % 4), c = pick((it / 16) % 4);
        long f0 = fails;
        #define CHK(i, op, got, exp) do { long before = fails; report(op, a, b, c, got, exp); per[i] += fails - before; n++; } while (0)
        CHK(0, "add", sf_add(a, b), U(_mm_cvtss_f32(_mm_add_ss(_mm_set_ss(F(a)), _mm_set_ss(F(b))))));
        CHK(1, "sub", sf_sub(a, b), U(_mm_cvtss_f32(_mm_sub_ss(_mm_set_ss(F(a)), _mm_set_ss(F(b))))));
        CHK(2, "mul", sf_mul(a, b), U(_mm_cvtss_f32(_mm_mul_ss(_mm_set_ss(F(a)), _mm_set_ss(F(b))))));
        CHK(3, "div", sf_div(a, b), U(_mm_cvtss_f32(_mm_div_ss(_mm_set_ss(F(a)), _mm_set_ss(F(b))))));
        CHK(4, "sqrt", sf_sqrt(a), U(_mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(F(a))))));
        CHK(5, "fma", sf_fma(a, b, c), U(_mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(F(a)), _mm_set_ss(F(b)), _mm_set_ss(F(c))))));
        CHK(6, "min", sf_min(a, b), U(_mm_cvtss_f32(_mm_min_ss(_mm_set_ss(F(a)), _mm_set_ss(F(b))))));
        CHK(7, "max", sf_max(a, b), U(_mm_cvtss_f32(_mm_max_ss(_mm_set_ss(F(a)), _mm_set_ss(F(b))))));
        CHK(8, "cmp", (uint32_t)(sf_lt(a, b) | sf_le(a, b) << 1 | sf_eq(a, b) << 2 | sf_unordered(a, b) << 3),
            (uint32_t)((F(a) < F(b)) | (F(a) <= F(b)) << 1 | (F(a) == F(b)) << 2 | (isnan(F(a)) || isnan(F(b))) << 3));
        CHK(9, "i32", (uint32_t)sf_to_i32(a), (uint32_t)_mm_cvttss_si32(_mm_set_ss(F(a))));
        CHK(10, "i64", (uint32_t)(sf_to_i64(a) >> 32) ^ (uint32_t)sf_to_i64(a), (uint32_t)(_mm_cvttss_si64(_mm_set_ss(F(a))) >> 32) ^ (uint32_t)_mm_cvttss_si64(_mm_set_ss(F(a))));
        CHK(11, "u32", sf_to_u32(a), (uint32_t)(int64_t)_mm_cvttss_si64(_mm_set_ss(F(a))));
        CHK(12, "f_i32", sf_from_i32((int32_t)a), U((float)(int32_t)a));
        CHK(13, "f_u32", sf_from_u32(a), U((float)a));
        { int64_t v = (int64_t)(((uint64_t)a << 32) | b) >> (c & 31);
          CHK(14, "f_i64", sf_from_i64(v), U((float)v)); }
        CHK(15, "floor", sf_floor(a), U(floorf(F(a))));
        CHK(16, "ceil", sf_ceil(a), U(ceilf(F(a))));
        CHK(17, "trunc", sf_trunc(a), U(truncf(F(a))));
        CHK(18, "rint", sf_rint(a), U(nearbyintf(F(a))));
        CHK(19, "round", sf_round(a), U(roundf(F(a))));
        {   /* NaN: numpy convention (payload kept, not quieted) instead of F16C's quieting */
            uint32_t nan_h = ((a >> 16) & 0x8000u) | (0x7C00u + ((a & 0x7FFFFFu) >> 13));
            if ((nan_h & 0x7FFFu) == 0x7C00u) nan_h |= 1;
            int an = (a & 0x7FFFFFFFu) > 0x7F800000u, hn = (a & 0x7C00u) == 0x7C00u && (a & 0x3FFu);
            CHK(20, "f16", sf_to_f16(a), an ? nan_h : (uint32_t)_cvtss_sh(F(a), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            CHK(21, "from_f16", sf_from_f16((uint16_t)a), hn ? (((a & 0x8000u) << 16) | 0x7F800000u | ((a & 0x3FFu) << 13)) : U(_cvtsh_ss((uint16_t)a)));
        }
        (void)f0;
    }
    printf("%ld checks, %ld mismatches\n", n, fails);
    for (int i = 0; i < 22; i++) if (per[i]) printf("  %-8s %ld mismatches\n", names[i], per[i]);
    return fails != 0;
}
