/* softfloat.c - IEEE-754 binary32 arithmetic in integer C for trn3 GPSIMD kernels.
 *
 * The trn3 GPSIMD core cannot execute the ncore2gp toolchain's floating-point instructions (they
 * live in FLIX bundles on vector registers), so float math is done in software. All operations
 * round to nearest, ties to even, support subnormals, and follow x86-64 SSE conventions for NaN
 * results so that device results are bit-identical to host (x86) references:
 *   - invalid operation (inf-inf, 0*inf, 0/0, inf/inf, sqrt(<0), fma invalid) -> 0xFFC00000
 *   - NaN operand(s): the first NaN operand (in operand order) is returned, quieted
 *   - float -> int32 / int64 conversion of NaN or out-of-range values -> INT_MIN (0x80000000...)
 * The same file compiles on the host (tests/softfloat_test.c) for validation.
 */
#include <stdint.h>

#include <gpsimd/softfloat.h>

#define SIGN 0x80000000u
#define EXPM 0x7F800000u
#define FRACM 0x007FFFFFu
#define QNAN 0x00400000u
#define DEFAULT_NAN 0xFFC00000u

static inline int is_nan(uint32_t a) { return (a & 0x7FFFFFFFu) > EXPM; }
static inline int is_inf(uint32_t a) { return (a & 0x7FFFFFFFu) == EXPM; }
static inline uint32_t quiet(uint32_t a) { return a | QNAN; }
static inline int clz32(uint32_t a) {
#ifdef __XTENSA__
    uint32_t r;
    __asm__("nsau %0, %1" : "=r"(r) : "r"(a));      /* hardware count-leading-zeros (32 for 0) */
    return (int)r;
#endif
    int n = 0;
    if (!a) return 32;
    if (!(a & 0xFFFF0000u)) { n += 16; a <<= 16; }
    if (!(a & 0xFF000000u)) { n += 8; a <<= 8; }
    if (!(a & 0xF0000000u)) { n += 4; a <<= 4; }
    if (!(a & 0xC0000000u)) { n += 2; a <<= 2; }
    if (!(a & 0x80000000u)) { n += 1; }
    return n;
}
/* 64-bit helpers built from 32-bit operations (keep the library free of helper calls) */
typedef struct { uint32_t lo, hi; } u64p;
static inline u64p mul32x32(uint32_t a, uint32_t b) {
#ifdef __XTENSA__
    u64p p;
    p.lo = a * b;                                    /* mull */
    __asm__("muluh %0, %1, %2" : "=r"(p.hi) : "r"(a), "r"(b));
    return p;
#endif
    uint32_t al = a & 0xFFFF, ah = a >> 16, bl = b & 0xFFFF, bh = b >> 16;
    uint32_t ll = al * bl, lh = al * bh, hl = ah * bl, hh = ah * bh;
    uint32_t mid = (ll >> 16) + (lh & 0xFFFF) + (hl & 0xFFFF);
    u64p r;
    r.lo = (ll & 0xFFFF) | (mid << 16);
    r.hi = hh + (lh >> 16) + (hl >> 16) + (mid >> 16);
    return r;
}

/* Round and pack: sign, biased exponent `e` of the value sig * 2^(e - 127 - 30), where `sig` has
 * its leading 1 at bit 30 (or is subnormal-aligned). Bits below bit 7 are round/sticky bits. */
static uint32_t round_pack(uint32_t sign, int32_t e, uint32_t sig) {
    if (e <= 0) {                      /* subnormal: shift right, keeping sticky */
        uint32_t sh = (uint32_t)(1 - e);
        sig = sh >= 31 ? (sig != 0) : (sig >> sh) | ((sig & ((1u << sh) - 1)) != 0);
        e = 0;
    }
    uint32_t round = sig & 0x7F;
    sig >>= 7;
    if (round > 0x40 || (round == 0x40 && (sig & 1))) {
        sig++;
        if (e == 0 && sig == 0x00800000u) e = 1;            /* subnormal rounded up to normal */
        if (sig == 0x01000000u) { sig >>= 1; e++; }          /* mantissa overflow */
    }
    if (e >= 255) return sign | EXPM;                        /* overflow -> inf */
    if (e == 0) return sign | sig;                           /* subnormal / zero */
    return sign | ((uint32_t)e << 23) | (sig & FRACM);
}
/* Normalize so the leading 1 is at bit 30, adjusting e. */
static inline uint32_t norm30(uint32_t sig, int32_t *e) {
    int s = clz32(sig) - 1;
    *e -= s;
    return s >= 0 ? sig << s : sig >> -s;
}

static inline void unpack(uint32_t a, int32_t *e, uint32_t *m) {
    *e = (int32_t)((a >> 23) & 0xFF);
    *m = a & FRACM;
    if (*e) *m |= 0x00800000u; else *e = 1;   /* subnormal: exponent 1, no hidden bit */
}

uint32_t sf_add(uint32_t a, uint32_t b) {
    if (is_nan(a)) return quiet(a);
    if (is_nan(b)) return quiet(b);
    if (is_inf(a)) return (is_inf(b) && ((a ^ b) & SIGN)) ? DEFAULT_NAN : a;
    if (is_inf(b)) return b;
    if (!(a & 0x7FFFFFFFu) && !(b & 0x7FFFFFFFu)) return a & b;   /* +0 + -0 = +0, -0 + -0 = -0 */
    if (!(a & 0x7FFFFFFFu)) return b;
    if (!(b & 0x7FFFFFFFu)) return a;
    int32_t ea, eb; uint32_t ma, mb;
    unpack(a, &ea, &ma); unpack(b, &eb, &mb);
    /* align to bit 30 with 7 guard bits */
    ma <<= 7; mb <<= 7;
    uint32_t sa = a & SIGN, sb = b & SIGN;
    if (ea < eb || (ea == eb && ma < mb)) { int32_t te = ea; ea = eb; eb = te; uint32_t tm = ma; ma = mb; mb = tm; uint32_t ts = sa; sa = sb; sb = ts; }
    uint32_t d = (uint32_t)(ea - eb);
    if (d) mb = d >= 31 ? (mb != 0) : (mb >> d) | ((mb & ((1u << d) - 1)) != 0);
    uint32_t m;
    int32_t e = ea;
    if (sa == sb) m = ma + mb;
    else {
        m = ma - mb;
        if (!m) return 0;                                   /* exact cancellation -> +0 */
    }
    if (m & 0x80000000u) { m = (m >> 1) | (m & 1); e++; }
    else m = norm30(m, &e);
    return round_pack(sa, e, m);
}
uint32_t sf_sub(uint32_t a, uint32_t b) {
    if (is_nan(a)) return quiet(a);
    if (is_nan(b)) return quiet(b);
    return sf_add(a, b ^ SIGN);
}

uint32_t sf_mul(uint32_t a, uint32_t b) {
    if (is_nan(a)) return quiet(a);
    if (is_nan(b)) return quiet(b);
    uint32_t sign = (a ^ b) & SIGN;
    int za = !(a & 0x7FFFFFFFu), zb = !(b & 0x7FFFFFFFu);
    if (is_inf(a) || is_inf(b)) return (za || zb) ? DEFAULT_NAN : sign | EXPM;
    if (za || zb) return sign;
    int32_t ea, eb; uint32_t ma, mb;
    unpack(a, &ea, &ma); unpack(b, &eb, &mb);
    /* product of two 24-bit significands: up to 48 bits */
    u64p p = mul32x32(ma << 8, mb << 8);          /* = ma*mb << 16, leading 1 at bit 62 or 63 (or lower if subnormal) */
    int32_t e = ea + eb - 127 + 1;                /* exponent for leading bit at bit 63 of p */
    /* normalize p so that the leading 1 is at bit 63 */
    if (!p.hi) { p.hi = p.lo; p.lo = 0; e -= 32; }
    int s = clz32(p.hi);
    if (s) { p.hi = (p.hi << s) | (p.lo >> (32 - s)); p.lo <<= s; e -= s; }
    uint32_t sig = (p.hi >> 1) | ((p.hi & 1) | (p.lo != 0));   /* leading 1 at bit 30, sticky */
    return round_pack(sign, e, sig);
}

uint32_t sf_div(uint32_t a, uint32_t b) {
    if (is_nan(a)) return quiet(a);
    if (is_nan(b)) return quiet(b);
    uint32_t sign = (a ^ b) & SIGN;
    int za = !(a & 0x7FFFFFFFu), zb = !(b & 0x7FFFFFFFu);
    if (is_inf(a)) return is_inf(b) ? DEFAULT_NAN : sign | EXPM;
    if (is_inf(b)) return sign;
    if (zb) return za ? DEFAULT_NAN : sign | EXPM;
    if (za) return sign;
    int32_t ea, eb; uint32_t ma, mb;
    unpack(a, &ea, &ma); unpack(b, &eb, &mb);
    /* normalize both significands to have the leading 1 at bit 23 */
    while (!(ma & 0x00800000u)) { ma <<= 1; ea--; }
    while (!(mb & 0x00800000u)) { mb <<= 1; eb--; }
    int32_t e = ea - eb + 127;
    if (ma < mb) { ma <<= 1; e--; }
    /* long division: 26 quotient bits (leading 1 at bit 25), remainder -> sticky */
    uint32_t q = 0, r = ma;
    for (int i = 0; i < 26; i++) {
        q <<= 1;
        if (r >= mb) { r -= mb; q |= 1; }
        r <<= 1;
    }
    uint32_t sig = (q << 5) | (r != 0);           /* leading 1 at bit 30 */
    return round_pack(sign, e, sig);
}

uint32_t sf_sqrt(uint32_t a) {
    if (is_nan(a)) return quiet(a);
    if (!(a & 0x7FFFFFFFu)) return a;             /* +-0 */
    if (a & SIGN) return DEFAULT_NAN;
    if (is_inf(a)) return a;
    int32_t e; uint32_t m;
    unpack(a, &e, &m);
    while (!(m & 0x00800000u)) { m <<= 1; e--; }
    int32_t k = e - 150;                          /* value = m * 2^k, m in [2^23, 2^24) */
    if (k & 1) { m <<= 1; k--; }                  /* make k even: m in [2^23, 2^25) */
    /* integer sqrt of M = m << 30 (64-bit, hi:lo); root in [2^26.5, 2^27.5), remainder < 2^29 */
    uint32_t hi = m >> 2, lo = m << 30, root = 0, rem = 0;
    for (int i = 0; i < 32; i++) {
        rem = (rem << 2) | (hi >> 30);
        hi = (hi << 2) | (lo >> 30); lo <<= 2;
        uint32_t trial = (root << 2) | 1;
        root <<= 1;
        if (rem >= trial) { rem -= trial; root |= 1; }
    }
    /* sqrt(value) = root * 2^(k/2 - 15) (+ fraction if rem != 0) */
    int L = 31 - clz32(root);                     /* 26 or 27 */
    uint32_t sig = (root << (30 - L)) | (rem != 0);
    return round_pack(0, 127 + k / 2 - 15 + L, sig);
}

/* ---- 128-bit helpers for fma (w[3] is the most significant word) */
typedef struct { uint32_t w[4]; } u128;
static inline int is_zero128(const u128 *x) { return !(x->w[0] | x->w[1] | x->w[2] | x->w[3]); }
static void shl128(u128 *x, int s) {
    while (s >= 32) { x->w[3] = x->w[2]; x->w[2] = x->w[1]; x->w[1] = x->w[0]; x->w[0] = 0; s -= 32; }
    if (s) { for (int i = 3; i > 0; i--) x->w[i] = (x->w[i] << s) | (x->w[i - 1] >> (32 - s)); x->w[0] <<= s; }
}
static uint32_t shr128_sticky(u128 *x, int s) {   /* returns 1 if any 1 bit was shifted out */
    uint32_t st = 0;
    if (s >= 128) { st = !is_zero128(x); x->w[0] = x->w[1] = x->w[2] = x->w[3] = 0; return st; }
    while (s >= 32) { st |= x->w[0] != 0; x->w[0] = x->w[1]; x->w[1] = x->w[2]; x->w[2] = x->w[3]; x->w[3] = 0; s -= 32; }
    if (s) { st |= (x->w[0] & ((1u << s) - 1)) != 0; for (int i = 0; i < 3; i++) x->w[i] = (x->w[i] >> s) | (x->w[i + 1] << (32 - s)); x->w[3] >>= s; }
    return st;
}
static int ge128(const u128 *x, const u128 *y) {
    for (int i = 3; i >= 0; i--) if (x->w[i] != y->w[i]) return x->w[i] > y->w[i];
    return 1;
}
static void add128(u128 *r, const u128 *x, const u128 *y) {
    uint32_t c = 0;
    for (int i = 0; i < 4; i++) { uint32_t t = x->w[i] + c; uint32_t c1 = t < c; r->w[i] = t + y->w[i]; c = c1 | (r->w[i] < t); }
}
static void sub128(u128 *r, const u128 *x, const u128 *y) {   /* r = x - y, x >= y */
    uint32_t b = 0;
    for (int i = 0; i < 4; i++) { uint32_t t = x->w[i] - y->w[i]; uint32_t b1 = x->w[i] < y->w[i]; r->w[i] = t - b; b = b1 | (t < b); }
}
static int clz128(const u128 *x) {
    for (int i = 3; i >= 0; i--) if (x->w[i]) return (3 - i) * 32 + clz32(x->w[i]);
    return 128;
}

uint32_t sf_fma(uint32_t a, uint32_t b, uint32_t c) {   /* a*b + c, single rounding (x86 vfmadd semantics) */
    if (is_nan(a)) return quiet(a);
    if (is_nan(b)) return quiet(b);
    if (is_nan(c)) return quiet(c);
    uint32_t ps = (a ^ b) & SIGN, cs = c & SIGN;
    int za = !(a & 0x7FFFFFFFu), zb = !(b & 0x7FFFFFFFu), zc = !(c & 0x7FFFFFFFu);
    if (is_inf(a) || is_inf(b)) {
        if (za || zb) return DEFAULT_NAN;
        if (is_inf(c) && cs != ps) return DEFAULT_NAN;
        return ps | EXPM;
    }
    if (is_inf(c)) return c;
    if (za || zb) return zc ? ((ps == cs) ? c : 0) : c;          /* exact zero product */
    int32_t ea, eb, ec; uint32_t ma, mb, mc;
    unpack(a, &ea, &ma); unpack(b, &eb, &mb);
    while (!(ma & 0x00800000u)) { ma <<= 1; ea--; }
    while (!(mb & 0x00800000u)) { mb <<= 1; eb--; }
    u64p p = mul32x32(ma, mb);                    /* exact, < 2^48 */
    u128 X = {{p.lo, p.hi, 0, 0}}, Y = {{0, 0, 0, 0}};
    int32_t xe = ea + eb - 300;                   /* X = P * 2^xe */
    shl128(&X, 77); xe -= 77;                     /* bit 47 -> bit 124 */
    int32_t ye = 0;
    if (!zc) {
        unpack(c, &ec, &mc);
        while (!(mc & 0x00800000u)) { mc <<= 1; ec--; }
        Y.w[0] = mc; ye = ec - 150;               /* Y = mc * 2^ye */
        shl128(&Y, 101); ye -= 101;               /* bit 23 -> bit 124 */
    } else {
        ye = xe;                                  /* zero addend */
    }
    uint32_t sticky = 0, rs;
    int32_t exp;
    int x_shifted;
    if (zc) { exp = xe; x_shifted = 0; }
    else if (xe >= ye) { sticky = shr128_sticky(&Y, xe - ye); exp = xe; x_shifted = 0; }
    else { sticky = shr128_sticky(&X, ye - xe); exp = ye; x_shifted = 1; }
    u128 R;
    if (zc || ps == cs) { add128(&R, &X, &Y); rs = ps; }
    else {
        int xge = ge128(&X, &Y);
        if (xge) { sub128(&R, &X, &Y); rs = ps; } else { sub128(&R, &Y, &X); rs = cs; }
        if (sticky) {                             /* the shifted (smaller) operand had extra bits */
            int small_is_subtrahend = xge ? !x_shifted : x_shifted;
            if (small_is_subtrahend) { u128 one = {{1, 0, 0, 0}}; sub128(&R, &R, &one); }
        }
        if (is_zero128(&R) && !sticky) return 0;  /* exact cancellation -> +0 */
    }
    int L = 127 - clz128(&R);                     /* leading one bit index */
    uint32_t sig;
    if (L >= 30) { sticky |= shr128_sticky(&R, L - 30); sig = R.w[0]; }
    else { shl128(&R, 30 - L); sig = R.w[0]; }
    return round_pack(rs, exp + L + 127, sig | sticky);
}

/* ------------------------------------------------------------------ comparisons */
int sf_lt(uint32_t a, uint32_t b) {
    if (is_nan(a) || is_nan(b)) return 0;
    if (!((a | b) & 0x7FFFFFFFu)) return 0;
    if ((a & SIGN) != (b & SIGN)) return (a & SIGN) != 0;
    return (a & SIGN) ? a > b : a < b;
}
int sf_eq(uint32_t a, uint32_t b) {
    if (is_nan(a) || is_nan(b)) return 0;
    return a == b || !((a | b) & 0x7FFFFFFFu);
}
int sf_le(uint32_t a, uint32_t b) { return sf_lt(a, b) || sf_eq(a, b); }
int sf_unordered(uint32_t a, uint32_t b) { return is_nan(a) || is_nan(b); }
/* x86 minss/maxss semantics: returns the second operand if either is NaN or both are zero */
uint32_t sf_min(uint32_t a, uint32_t b) { return sf_lt(a, b) ? a : b; }
uint32_t sf_max(uint32_t a, uint32_t b) { return sf_lt(b, a) ? a : b; }

/* ------------------------------------------------------------------ conversions */
uint32_t sf_from_i32(int32_t v) {
    if (!v) return 0;
    uint32_t sign = v < 0 ? SIGN : 0, m = v < 0 ? 0u - (uint32_t)v : (uint32_t)v;
    int32_t e = 127 + 30;
    m = norm30(m, &e);
    return round_pack(sign, e, m);
}
uint32_t sf_from_u32(uint32_t m) {
    if (!m) return 0;
    int32_t e = 127 + 30;
    if (m & 0x80000000u) { m = (m >> 1) | (m & 1); e++; }
    else m = norm30(m, &e);
    return round_pack(0, e, m);
}
uint32_t sf_from_i64(int64_t v) {
    uint64_t u = v < 0 ? 0u - (uint64_t)v : (uint64_t)v;
    uint32_t hi = (uint32_t)(u >> 32), lo = (uint32_t)u, sign = v < 0 ? SIGN : 0;
    if (!hi) { uint32_t r = sf_from_u32(lo); return r ? r | sign : 0; }
    int s = clz32(hi);                        /* leading 1 of the 64-bit value at bit 63-s */
    int32_t e = 127 + 63 - s;
    uint32_t top = s ? (hi << s) | (lo >> (32 - s)) : hi, rest = s ? lo << s : lo;
    uint32_t sig = (top >> 1) | ((top & 1) | (rest != 0));
    return round_pack(sign, e, sig);
}
int32_t sf_to_i32(uint32_t a) {               /* truncation toward zero (C cast semantics) */
    if (is_nan(a)) return INT32_MIN;
    int32_t e = (int32_t)((a >> 23) & 0xFF) - 127;
    if (e < 0) return 0;
    if (e >= 31) return INT32_MIN;            /* includes -2^31 exactly */
    uint32_t m = (a & FRACM) | 0x00800000u;
    uint32_t v = e >= 23 ? m << (e - 23) : m >> (23 - e);
    return (a & SIGN) ? -(int32_t)v : (int32_t)v;
}
int64_t sf_to_i64(uint32_t a) {
    if (is_nan(a)) return INT64_MIN;
    int32_t e = (int32_t)((a >> 23) & 0xFF) - 127;
    if (e < 0) return 0;
    if (e >= 63) return INT64_MIN;
    uint32_t m = (a & FRACM) | 0x00800000u;
    uint64_t v;
    if (e >= 23) {
        int32_t sh = e - 23;
        uint32_t hi = sh >= 32 ? m << (sh - 32) : sh ? m >> (32 - sh) : 0, lo = sh >= 32 ? 0 : m << sh;
        v = ((uint64_t)hi << 32) | lo;
    } else v = m >> (23 - e);
    return (a & SIGN) ? (int64_t)(0u - v) : (int64_t)v;
}
uint32_t sf_to_u32(uint32_t a) { return (uint32_t)sf_to_i64(a); }   /* == x86-64 (uint32_t)f */
/* round to integral value in floating point */
static uint32_t round_int(uint32_t a, int mode) {   /* 0 trunc, 1 floor, 2 ceil, 3 nearest-even, 4 half-away */
    if (is_nan(a)) return quiet(a);
    int32_t e = (int32_t)((a >> 23) & 0xFF) - 127;
    if (e >= 23) return a;                    /* already integral (or inf) */
    uint32_t sign = a & SIGN;
    if (e < 0) {
        if (!(a & 0x7FFFFFFFu)) return a;
        switch (mode) {
            case 1: return sign ? 0xBF800000u : 0;
            case 2: return sign ? SIGN : 0x3F800000u;
            case 3: return (e == -1 && (a & FRACM)) ? sign | 0x3F800000u : sign;
            case 4: return e == -1 ? sign | 0x3F800000u : sign;
            default: return sign;
        }
    }
    uint32_t frac_mask = FRACM >> e, frac = a & frac_mask;
    if (!frac) return a;
    uint32_t t = a & ~frac_mask, half = 0x00400000u >> e, one = 0x00800000u >> e;
    switch (mode) {
        case 1: return sign ? sf_add(t, 0xBF800000u) : t;
        case 2: return sign ? t : sf_add(t, 0x3F800000u);
        case 3: if (frac > half || (frac == half && (t & one))) return sf_add(t, sign | 0x3F800000u); return t;
        case 4: if (frac >= half) return sf_add(t, sign | 0x3F800000u); return t;
        default: return t;
    }
}
uint32_t sf_trunc(uint32_t a) { return round_int(a, 0); }
uint32_t sf_floor(uint32_t a) { return round_int(a, 1); }
uint32_t sf_ceil(uint32_t a) { return round_int(a, 2); }
uint32_t sf_rint(uint32_t a) { return round_int(a, 3); }
uint32_t sf_round(uint32_t a) { return round_int(a, 4); }

/* ------------------------------------------------------------------ bf16 / fp16 */
/* NaN conventions follow the numpy ecosystem, so results compare bit-exactly with numpy references:
 * bf16: canonical quiet NaN with the sign kept (ml_dtypes); fp16: payload kept, not quieted (numpy). */
uint16_t sf_to_bf16(uint32_t a) {             /* round to nearest even; NaN -> +-0x7FC0 */
    if (is_nan(a)) return (uint16_t)(((a >> 16) & 0x8000u) | 0x7FC0u);
    uint32_t lsb = (a >> 16) & 1;
    return (uint16_t)((a + 0x7FFFu + lsb) >> 16);
}
uint32_t sf_from_bf16(uint16_t h) { return (uint32_t)h << 16; }
uint16_t sf_to_f16(uint32_t a) {              /* IEEE binary16, round to nearest even, with subnormals */
    uint32_t sign = (a >> 16) & 0x8000u;
    if (is_nan(a)) {                                          /* numpy: keep the top payload bits */
        uint32_t r = 0x7C00u + ((a & FRACM) >> 13);
        return (uint16_t)(sign | (r == 0x7C00u ? 0x7C01u : r));
    }
    if (is_inf(a)) return (uint16_t)(sign | 0x7C00u);
    int32_t e = (int32_t)((a >> 23) & 0xFF) - 127 + 15;
    uint32_t m = a & FRACM;
    if (!((a >> 23) & 0xFF)) return (uint16_t)sign;           /* f32 subnormals underflow to 0 */
    m |= 0x00800000u;
    if (e >= 31) return (uint16_t)(sign | 0x7C00u);
    uint32_t sh = e > 0 ? 13 : (uint32_t)(14 - e);           /* bits to drop */
    if (sh > 24) return (uint16_t)sign;
    uint32_t keep = m >> sh, rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
    if (rem > half || (rem == half && (keep & 1))) keep++;
    if (e > 0) {
        uint32_t r = ((uint32_t)e << 10) + (keep - 0x400u);  /* keep includes the hidden bit */
        if (r >= 0x7C00u) return (uint16_t)(sign | 0x7C00u);
        return (uint16_t)(sign | r);
    }
    return (uint16_t)(sign | keep);                           /* subnormal (may round up to min normal) */
}
uint32_t sf_from_f16(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FFu;
    if (e == 31) return sign | EXPM | (m << 13);             /* inf / NaN (payload kept, numpy) */
    if (e == 0) {
        if (!m) return sign;
        int32_t ee = 127 - 14;
        while (!(m & 0x400u)) { m <<= 1; ee--; }
        return sign | ((uint32_t)ee << 23) | ((m & 0x3FFu) << 13);
    }
    return sign | ((e + 112) << 23) | (m << 13);
}
