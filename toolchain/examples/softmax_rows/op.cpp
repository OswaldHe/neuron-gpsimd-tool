// softmax_rows: per-partition softmax in float32 (soft-float). Rows are staged through local RAM (iDMA).
#include <gpsimd/math.hpp>

using gp::f32;

GPSIMD_OP(softmax_rows) {
    gp::Call c(call);
    auto X = c.in<f32>(0);
    auto Y = c.out<f32>(0);
    const uint32_t n = X.elems();
    constexpr uint32_t CAP = 4096;
    uint32_t buf[CAP];                                   // float32 bits
    if (!X.contiguous() || !Y.contiguous() || n > CAP) { c.fail(8); return; }
    for (gp::Range blk : c.blocks(X.partitions(), CAP / n)) {
        c.dma_wait(c.fetch(X, blk.lo, blk.size(), buf));
        for (uint32_t q = 0; q < blk.size(); ++q) {
            uint32_t *row = buf + q * n;
            f32 m = f32::from_bits(row[0]);
            for (uint32_t i = 1; i < n; ++i) m = gp::max(m, f32::from_bits(row[i]));
            f32 s;
            for (uint32_t i = 0; i < n; ++i) {
                const f32 e = gp::math::exp(f32::from_bits(row[i]) - m);
                row[i] = e.bits;
                s = s + e;
            }
            for (uint32_t i = 0; i < n; ++i) row[i] = (f32::from_bits(row[i]) / s).bits;
        }
        c.dma_wait(c.flush(Y, blk.lo, blk.size(), buf));
    }
}
