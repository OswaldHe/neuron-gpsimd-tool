/* builtins.c - compiler-runtime helpers for trn3 GPSIMD kernel libraries.
 *
 * The libgcc shipped with the ncore2gp toolchain is compiled with FLIX bundles that the trn3 GPSIMD
 * core cannot execute, so kernel libraries link against these base-ISA implementations instead.
 * Every routine is written with 32-bit operations only (no calls into other helpers).
 * Covers: 64-bit shifts / multiply / divide / modulo, bit counting, byte swaps, mem* and str*.
 */
#include <stddef.h>
#include <stdint.h>

typedef union { uint64_t u; int64_t s; struct { uint32_t lo, hi; } w; } du;

/* ------------------------------------------------------------------ 64-bit shifts */
uint64_t __ashldi3(uint64_t a, int b) {
    du x = {.u = a}, r;
    b &= 63;
    if (b == 0) return a;
    if (b >= 32) { r.w.lo = 0; r.w.hi = x.w.lo << (b - 32); }
    else { r.w.lo = x.w.lo << b; r.w.hi = (x.w.hi << b) | (x.w.lo >> (32 - b)); }
    return r.u;
}
uint64_t __lshrdi3(uint64_t a, int b) {
    du x = {.u = a}, r;
    b &= 63;
    if (b == 0) return a;
    if (b >= 32) { r.w.hi = 0; r.w.lo = x.w.hi >> (b - 32); }
    else { r.w.hi = x.w.hi >> b; r.w.lo = (x.w.lo >> b) | (x.w.hi << (32 - b)); }
    return r.u;
}
int64_t __ashrdi3(int64_t a, int b) {
    du x = {.s = a}, r;
    b &= 63;
    if (b == 0) return a;
    if (b >= 32) { r.w.hi = (uint32_t)((int32_t)x.w.hi >> 31); r.w.lo = (uint32_t)((int32_t)x.w.hi >> (b - 32)); }
    else { r.w.hi = (uint32_t)((int32_t)x.w.hi >> b); r.w.lo = (x.w.lo >> b) | (x.w.hi << (32 - b)); }
    return r.s;
}

/* ------------------------------------------------------------------ 64-bit multiply */
static inline uint32_t mulhu32(uint32_t a, uint32_t b) {   /* high word of a*b via 16-bit pieces */
    uint32_t al = a & 0xFFFF, ah = a >> 16, bl = b & 0xFFFF, bh = b >> 16;
    uint32_t ll = al * bl, lh = al * bh, hl = ah * bl, hh = ah * bh;
    uint32_t mid = (ll >> 16) + (lh & 0xFFFF) + (hl & 0xFFFF);
    return hh + (lh >> 16) + (hl >> 16) + (mid >> 16);
}
uint64_t __muldi3(uint64_t a, uint64_t b) {
    du x = {.u = a}, y = {.u = b}, r;
    r.w.lo = x.w.lo * y.w.lo;
    r.w.hi = mulhu32(x.w.lo, y.w.lo) + x.w.lo * y.w.hi + x.w.hi * y.w.lo;
    return r.u;
}

/* ------------------------------------------------------------------ 64-bit divide / modulo */
static inline int lt64(du a, du b) { return a.w.hi < b.w.hi || (a.w.hi == b.w.hi && a.w.lo < b.w.lo); }
static uint64_t udivmod64(uint64_t num, uint64_t den, uint64_t *rem) {
    du n = {.u = num}, d = {.u = den}, q = {.u = 0}, r = {.u = 0};
    if (d.w.hi == 0 && d.w.lo == 0) { if (rem) *rem = num; return ~(uint64_t)0; }
    for (int i = 63; i >= 0; i--) {
        /* r = (r << 1) | bit i of n */
        r.w.hi = (r.w.hi << 1) | (r.w.lo >> 31);
        r.w.lo = (r.w.lo << 1) | ((i >= 32 ? n.w.hi >> (i - 32) : n.w.lo >> i) & 1);
        if (!lt64(r, d)) {
            uint32_t borrow = r.w.lo < d.w.lo;
            r.w.lo -= d.w.lo; r.w.hi -= d.w.hi + borrow;
            if (i >= 32) q.w.hi |= 1u << (i - 32); else q.w.lo |= 1u << i;
        }
    }
    if (rem) *rem = r.u;
    return q.u;
}
uint64_t __udivdi3(uint64_t a, uint64_t b) { return udivmod64(a, b, 0); }
uint64_t __umoddi3(uint64_t a, uint64_t b) { uint64_t r; udivmod64(a, b, &r); return r; }
uint64_t __udivmoddi4(uint64_t a, uint64_t b, uint64_t *r) { return udivmod64(a, b, r); }
static inline uint64_t neg64(uint64_t a) { du x = {.u = a}; x.w.lo = ~x.w.lo; x.w.hi = ~x.w.hi; x.w.lo += 1; if (x.w.lo == 0) x.w.hi += 1; return x.u; }
int64_t __divdi3(int64_t a, int64_t b) {
    du x = {.s = a}, y = {.s = b};
    int neg = (int)((x.w.hi ^ y.w.hi) >> 31);
    uint64_t ua = (x.w.hi >> 31) ? neg64(x.u) : x.u, ub = (y.w.hi >> 31) ? neg64(y.u) : y.u;
    uint64_t q = udivmod64(ua, ub, 0);
    return (int64_t)(neg ? neg64(q) : q);
}
int64_t __moddi3(int64_t a, int64_t b) {
    du x = {.s = a}, y = {.s = b};
    uint64_t ua = (x.w.hi >> 31) ? neg64(x.u) : x.u, ub = (y.w.hi >> 31) ? neg64(y.u) : y.u, r;
    udivmod64(ua, ub, &r);
    return (int64_t)((x.w.hi >> 31) ? neg64(r) : r);
}

/* ------------------------------------------------------------------ bit counting / swaps */
int __popcountsi2(uint32_t a) {
    a = a - ((a >> 1) & 0x55555555u);
    a = (a & 0x33333333u) + ((a >> 2) & 0x33333333u);
    a = (a + (a >> 4)) & 0x0F0F0F0Fu;
    return (int)((a * 0x01010101u) >> 24);
}
int __popcountdi2(uint64_t a) { du x = {.u = a}; return __popcountsi2(x.w.lo) + __popcountsi2(x.w.hi); }
int __paritysi2(uint32_t a) { return __popcountsi2(a) & 1; }
int __paritydi2(uint64_t a) { return __popcountdi2(a) & 1; }
int __clzsi2(uint32_t a) {
    int n = 0;
    if (!a) return 32;
    if (!(a & 0xFFFF0000u)) { n += 16; a <<= 16; }
    if (!(a & 0xFF000000u)) { n += 8; a <<= 8; }
    if (!(a & 0xF0000000u)) { n += 4; a <<= 4; }
    if (!(a & 0xC0000000u)) { n += 2; a <<= 2; }
    if (!(a & 0x80000000u)) { n += 1; }
    return n;
}
int __ctzsi2(uint32_t a) {
    int n = 0;
    if (!a) return 32;
    if (!(a & 0xFFFFu)) { n += 16; a >>= 16; }
    if (!(a & 0xFFu)) { n += 8; a >>= 8; }
    if (!(a & 0xFu)) { n += 4; a >>= 4; }
    if (!(a & 0x3u)) { n += 2; a >>= 2; }
    if (!(a & 0x1u)) { n += 1; }
    return n;
}
int __clzdi2(uint64_t a) { du x = {.u = a}; return x.w.hi ? __clzsi2(x.w.hi) : 32 + __clzsi2(x.w.lo); }
int __ctzdi2(uint64_t a) { du x = {.u = a}; return x.w.lo ? __ctzsi2(x.w.lo) : 32 + __ctzsi2(x.w.hi); }
int __ffssi2(uint32_t a) { return a ? __ctzsi2(a) + 1 : 0; }
int __ffsdi2(uint64_t a) { return a ? __ctzdi2(a) + 1 : 0; }
uint32_t __bswapsi2(uint32_t a) { return (a >> 24) | ((a >> 8) & 0xFF00u) | ((a << 8) & 0xFF0000u) | (a << 24); }
uint64_t __bswapdi2(uint64_t a) { du x = {.u = a}, r; r.w.lo = __bswapsi2(x.w.hi); r.w.hi = __bswapsi2(x.w.lo); return r.u; }
int __cmpdi2(int64_t a, int64_t b) { du x = {.s = a}, y = {.s = b}; if ((int32_t)x.w.hi != (int32_t)y.w.hi) return (int32_t)x.w.hi < (int32_t)y.w.hi ? 0 : 2; return x.w.lo == y.w.lo ? 1 : x.w.lo < y.w.lo ? 0 : 2; }
int __ucmpdi2(uint64_t a, uint64_t b) { du x = {.u = a}, y = {.u = b}; if (x.w.hi != y.w.hi) return x.w.hi < y.w.hi ? 0 : 2; return x.w.lo == y.w.lo ? 1 : x.w.lo < y.w.lo ? 0 : 2; }
int64_t __negdi2(int64_t a) { return (int64_t)neg64((uint64_t)a); }

/* ------------------------------------------------------------------ memory / strings */
void *memcpy(void *dst, const void *src, size_t n) {
    volatile uint8_t *d = (volatile uint8_t *)dst; const volatile uint8_t *s = (const volatile uint8_t *)src;
    if ((((uint32_t)d | (uint32_t)s | n) & 3) == 0) {
        volatile uint32_t *d4 = (volatile uint32_t *)dst; const volatile uint32_t *s4 = (const volatile uint32_t *)src;
        for (size_t i = 0; i < n / 4; i++) d4[i] = s4[i];
        return dst;
    }
    for (size_t i = 0; i < n; i++) d[i] = s[i];
    return dst;
}
void *memmove(void *dst, const void *src, size_t n) {
    volatile uint8_t *d = (volatile uint8_t *)dst; const volatile uint8_t *s = (const volatile uint8_t *)src;
    if (d < s) for (size_t i = 0; i < n; i++) d[i] = s[i];
    else for (size_t i = n; i > 0; i--) d[i - 1] = s[i - 1];
    return dst;
}
void *memset(void *dst, int c, size_t n) {
    volatile uint8_t *d = (volatile uint8_t *)dst;
    if ((((uint32_t)d | n) & 3) == 0) {
        uint32_t v = (uint8_t)c; v |= v << 8; v |= v << 16;
        volatile uint32_t *d4 = (volatile uint32_t *)dst;
        for (size_t i = 0; i < n / 4; i++) d4[i] = v;
        return dst;
    }
    for (size_t i = 0; i < n; i++) d[i] = (uint8_t)c;
    return dst;
}
int memcmp(const void *a, const void *b, size_t n) {
    const volatile uint8_t *x = (const volatile uint8_t *)a, *y = (const volatile uint8_t *)b;
    for (size_t i = 0; i < n; i++) if (x[i] != y[i]) return x[i] < y[i] ? -1 : 1;
    return 0;
}
size_t strlen(const char *s) { const volatile char *p = s; size_t n = 0; while (p[n]) n++; return n; }
