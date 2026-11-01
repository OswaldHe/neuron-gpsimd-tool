/* softfloat.h - IEEE-754 binary32 operations on raw bit patterns (see softfloat.c). */
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
uint32_t sf_add(uint32_t a, uint32_t b);
uint32_t sf_sub(uint32_t a, uint32_t b);
uint32_t sf_mul(uint32_t a, uint32_t b);
uint32_t sf_div(uint32_t a, uint32_t b);
uint32_t sf_sqrt(uint32_t a);
uint32_t sf_fma(uint32_t a, uint32_t b, uint32_t c);
int sf_lt(uint32_t a, uint32_t b);
int sf_le(uint32_t a, uint32_t b);
int sf_eq(uint32_t a, uint32_t b);
int sf_unordered(uint32_t a, uint32_t b);
uint32_t sf_min(uint32_t a, uint32_t b);
uint32_t sf_max(uint32_t a, uint32_t b);
uint32_t sf_from_i32(int32_t v);
uint32_t sf_from_u32(uint32_t v);
uint32_t sf_from_i64(int64_t v);
int32_t sf_to_i32(uint32_t a);
int64_t sf_to_i64(uint32_t a);
uint32_t sf_to_u32(uint32_t a);
uint32_t sf_trunc(uint32_t a);
uint32_t sf_floor(uint32_t a);
uint32_t sf_ceil(uint32_t a);
uint32_t sf_rint(uint32_t a);
uint32_t sf_round(uint32_t a);
uint16_t sf_to_bf16(uint32_t a);
uint32_t sf_from_bf16(uint16_t h);
uint16_t sf_to_f16(uint32_t a);
uint32_t sf_from_f16(uint16_t h);
#ifdef __cplusplus
}
#endif
