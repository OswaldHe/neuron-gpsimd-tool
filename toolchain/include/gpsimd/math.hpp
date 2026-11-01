// math.hpp - float32 elementary functions for the GPSIMD cores (soft-float, gp::f32).
//
// Algorithms follow Cephes single precision (range reduction + minimax polynomial, Horner form).
// Every step is one correctly rounded float32 operation, so python/gpsimd_math_ref.py (the same
// algorithms written with numpy float32 operations) reproduces the results bit-exactly. Accuracy vs the
// true function: a few ulp (examples/transcendental reports the measured maximum error).
#pragma once
#include <gpsimd/gpsimd.hpp>

namespace gp {
namespace math {

inline f32 c(uint32_t bits) { return f32::from_bits(bits); }
inline bool isnan(f32 x) { return x.is_nan(); }
inline bool isinf(f32 x) { return (x.bits & 0x7FFFFFFFu) == 0x7F800000u; }
inline f32 copysign(f32 x, f32 s) { return f32::from_bits((x.bits & 0x7FFFFFFFu) | (s.bits & 0x80000000u)); }

// x = m * 2^e with m in [0.5, 1) (x finite, nonzero; subnormals handled)
inline f32 frexp(f32 x, int32_t *e) {
    uint32_t b = x.bits & 0x7FFFFFFFu;
    int32_t adj = 0;
    if (b < 0x00800000u) {                              // subnormal: scale by 2^25 (exact)
        x = x * GP_F32(33554432.0);
        b = x.bits & 0x7FFFFFFFu;
        adj = -25;
    }
    *e = static_cast<int32_t>(b >> 23) - 126 + adj;
    return f32::from_bits((x.bits & 0x807FFFFFu) | 0x3F000000u);
}

// e^x
inline f32 exp(f32 x) {
    if (isnan(x)) return x;
    if (x > GP_F32(88.72283905206835)) return c(0x7F800000u);
    if (x < GP_F32(-103.972084)) return f32();
    const f32 n = rint(x * GP_F32(1.44269504088896341));
    f32 r = x - n * GP_F32(0.693359375);
    r = r - n * GP_F32(-2.12194440e-4);
    const f32 z = r * r;
    f32 y = GP_F32(1.9875691500E-4);
    y = y * r + GP_F32(1.3981999507E-3);
    y = y * r + GP_F32(8.3334519073E-3);
    y = y * r + GP_F32(4.1665795894E-2);
    y = y * r + GP_F32(1.6666665459E-1);
    y = y * r + GP_F32(5.0000001201E-1);
    y = y * z + r + GP_F32(1.0);
    return ldexp(y, n.to_i32());
}

// natural logarithm
inline f32 log(f32 x) {
    if (isnan(x)) return x;
    if (x.bits == 0x7F800000u) return x;
    if ((x.bits & 0x7FFFFFFFu) == 0) return c(0xFF800000u);
    if (x.bits & 0x80000000u) return c(0x7FC00000u);
    int32_t e;
    f32 m = frexp(x, &e);
    if (m < GP_F32(0.707106781186547524)) {
        e -= 1;
        m = m + m - GP_F32(1.0);
    } else {
        m = m - GP_F32(1.0);
    }
    const f32 z = m * m;
    f32 y = GP_F32(7.0376836292E-2);
    y = y * m + GP_F32(-1.1514610310E-1);
    y = y * m + GP_F32(1.1676998740E-1);
    y = y * m + GP_F32(-1.2420140846E-1);
    y = y * m + GP_F32(1.4249322787E-1);
    y = y * m + GP_F32(-1.6668057665E-1);
    y = y * m + GP_F32(2.0000714765E-1);
    y = y * m + GP_F32(-2.4999993993E-1);
    y = y * m + GP_F32(3.3333331174E-1);
    y = y * m * z;
    const f32 fe = f32(e);
    y = y + fe * GP_F32(-2.12194440e-4);
    y = y + GP_F32(-0.5) * z;
    f32 r = m + y;
    r = r + fe * GP_F32(0.693359375);
    return r;
}

// hyperbolic tangent
inline f32 tanh(f32 x) {
    if (isnan(x)) return x;
    const f32 ax = abs(x);
    if (ax > GP_F32(44.3614195558365)) return copysign(GP_F32(1.0), x);
    if (ax >= GP_F32(0.625)) {
        const f32 s = exp(ax + ax);
        const f32 r = GP_F32(1.0) - GP_F32(2.0) / (s + GP_F32(1.0));
        return copysign(r, x);
    }
    if (ax.bits == 0) return x;
    const f32 z = x * x;
    f32 y = GP_F32(-5.70498872745E-3);
    y = y * z + GP_F32(2.06390887954E-2);
    y = y * z + GP_F32(-5.37397155531E-2);
    y = y * z + GP_F32(1.33314422036E-1);
    y = y * z + GP_F32(-3.33332819422E-1);
    return y * z * x + x;
}

// logistic sigmoid 1 / (1 + e^-x)
inline f32 sigmoid(f32 x) { return GP_F32(1.0) / (GP_F32(1.0) + exp(-x)); }

// sin / cos (Cephes sinf/cosf: reduction by pi/4 in three float32 parts; |x| <= 8192 for full accuracy)
inline f32 sincos_poly(f32 z, int32_t j, bool want_cos, f32 sign) {
    const f32 zz = z * z;
    f32 s = GP_F32(-1.9515295891E-4);
    s = s * zz + GP_F32(8.3321608736E-3);
    s = s * zz + GP_F32(-1.6666654611E-1);
    s = s * zz * z + z;
    f32 co = GP_F32(2.443315711809948E-005);
    co = co * zz + GP_F32(-1.388731625493765E-003);
    co = co * zz + GP_F32(4.166664568298827E-002);
    co = co * zz * zz - GP_F32(0.5) * zz + GP_F32(1.0);
    // sin: j&3 in {1,2} -> cos poly; cos: j&3 in {1,2} -> sin poly
    const bool poly_cos = want_cos ? !((j == 1) || (j == 2)) : ((j == 1) || (j == 2));
    const f32 r = poly_cos ? co : s;
    return sign.bits ? -r : r;
}
inline f32 sin(f32 x) {
    if (isnan(x) || isinf(x)) return c(0x7FC00000u);
    f32 sign = f32::from_bits(x.bits & 0x80000000u);
    const f32 ax = abs(x);
    int32_t j = (ax * GP_F32(1.27323954473516)).to_i32();
    f32 y = f32(j);
    if (j & 1) { j += 1; y = y + GP_F32(1.0); }
    j &= 7;
    if (j > 3) { sign = f32::from_bits(sign.bits ^ 0x80000000u); j -= 4; }
    const f32 z = ((ax - y * GP_F32(0.78515625)) - y * GP_F32(2.4187564849853515625e-4)) - y * GP_F32(3.77489497744594108e-8);
    return sincos_poly(z, j, false, sign);
}
inline f32 cos(f32 x) {
    if (isnan(x) || isinf(x)) return c(0x7FC00000u);
    const f32 ax = abs(x);
    int32_t j = (ax * GP_F32(1.27323954473516)).to_i32();
    f32 y = f32(j);
    if (j & 1) { j += 1; y = y + GP_F32(1.0); }
    j &= 7;
    uint32_t sign = 0;
    if (j > 3) { j -= 4; sign ^= 1; }
    if (j > 1) sign ^= 1;
    const f32 z = ((ax - y * GP_F32(0.78515625)) - y * GP_F32(2.4187564849853515625e-4)) - y * GP_F32(3.77489497744594108e-8);
    return sincos_poly(z, j, true, f32::from_bits(sign ? 0x80000000u : 0));
}

// GELU, tanh approximation: 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
inline f32 gelu(f32 x) {
    const f32 inner = GP_F32(0.7978845608028654) * (x + GP_F32(0.044715) * x * x * x);
    return GP_F32(0.5) * x * (GP_F32(1.0) + tanh(inner));
}

inline f32 rsqrt(f32 x) { return GP_F32(1.0) / sqrt(x); }

}  // namespace math
}  // namespace gp
