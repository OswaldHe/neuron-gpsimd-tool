// fp32_math: soft-float binary32 operations (gp::f32); inputs and the result block staged via iDMA.
#include <gpsimd/gpsimd.hpp>

using gp::f32;

GPSIMD_OP(fp32_math) {
    gp::Call c(call);
    auto A = c.in<f32>(0), B = c.in<f32>(1), C = c.in<f32>(2);
    auto I = c.in<int32_t>(3);
    auto O = c.out<uint32_t>(0);
    const uint32_t n = A.elems(), NB = O.elems() / n;
    constexpr uint32_t ROWS = 4, CAP = 64;               // up to 64 elements per input row
    uint32_t la[ROWS * CAP], lb[ROWS * CAP], lc[ROWS * CAP];
    int32_t li[ROWS * CAP];
    uint32_t lo[ROWS * 18 * CAP];                        // 18 result blocks
    if (n > CAP || NB > 18) { c.fail(8); return; }
    const f32 one_half = GP_F32(1.5);
    for (gp::Range blk : c.blocks(A.partitions(), ROWS)) {
        const uint32_t rows = blk.size();
        c.fetch(A, blk.lo, rows, la);
        c.fetch(B, blk.lo, rows, lb);
        c.fetch(C, blk.lo, rows, lc);
        c.dma_wait(c.fetch(I, blk.lo, rows, li));
        for (uint32_t q = 0; q < rows; ++q) {
            uint32_t *o = lo + q * NB * n;
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t e = q * n + i;
                const f32 a = f32::from_bits(la[e]), b = f32::from_bits(lb[e]), cc = f32::from_bits(lc[e]);
                const int32_t iv = li[e];
                const f32 fa = (a.bits & 0x7F800000u) == 0x7F800000u ? one_half : a;   // finite fma operands
                const f32 fb = (b.bits & 0x7F800000u) == 0x7F800000u ? one_half : b;
                uint32_t k = 0;
                auto put = [&](uint32_t bits) { o[(k++) * n + i] = bits; };
                put((a + b).bits);
                put((a - b).bits);
                put((a * b).bits);
                put((a / b).bits);
                put(gp::sqrt(a).bits);
                put(gp::fma(fa, fb, cc).bits);
                put(gp::min(a, b).bits);
                put(gp::max(a, b).bits);
                put(gp::floor(a).bits);
                put(gp::ceil(a).bits);
                put(gp::trunc(a).bits);
                put(gp::rint(a).bits);
                put(gp::round(a).bits);
                put(static_cast<uint32_t>(a.to_i32()));
                put(f32(iv).bits);
                put(static_cast<uint32_t>((a < b) | (a <= b) << 1 | (a == b) << 2 | (a.is_nan() || b.is_nan()) << 3));
                put((-a).bits ^ gp::abs(b).bits);
                put(f32(static_cast<uint32_t>(iv)).bits);
            }
        }
        c.dma_wait(c.flush(O, blk.lo, rows, lo));
    }
}
