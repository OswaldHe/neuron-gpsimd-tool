// half_formats: bf16 / fp16 storage types, exact widening to gp::f32, RNE narrowing, quantization.
// All tiles staged through local RAM with the iDMA.
#include <gpsimd/gpsimd.hpp>

using gp::bf16;
using gp::f16;
using gp::f32;

GPSIMD_OP(half_formats) {
    gp::Call c(call);
    auto X = c.in<bf16>(0);
    auto Y = c.in<f16>(1);
    auto Z = c.in<f32>(2);
    const uint32_t n = X.elems();
    constexpr uint32_t CAP = 1024;                       // elements per buffer
    uint16_t lx[CAP], ly[CAP], o0[CAP], o1[CAP], o3[CAP], o4[CAP];
    uint32_t lz[CAP], o2[CAP];
    int8_t o5[CAP];
    if (n > CAP) { c.fail(8); return; }
    const f32 scale = GP_F32(16.0), lo = GP_F32(-128.0), hi = GP_F32(127.0);
    for (gp::Range blk : c.blocks(X.partitions(), CAP / n)) {
        const uint32_t rows = blk.size();
        c.fetch(X, blk.lo, rows, lx);
        c.fetch(Y, blk.lo, rows, ly);
        c.dma_wait(c.fetch(Z, blk.lo, rows, lz));
        for (uint32_t k = 0; k < rows * n; ++k) {
            const f32 x = bf16::from_bits(lx[k]), y = f16::from_bits(ly[k]), z = f32::from_bits(lz[k]);
            o0[k] = bf16(x * y + z).bits;
            o1[k] = f16(x - y).bits;
            o2[k] = (x + y).bits;
            o3[k] = bf16(z).bits;
            o4[k] = f16(z).bits;
            const f32 q = gp::rint(x * scale);
            o5[k] = static_cast<int8_t>(q.is_nan() ? 0 : gp::max(lo, gp::min(q, hi)).to_i32());
        }
        c.flush(c.out<bf16>(0), blk.lo, rows, o0);
        c.flush(c.out<f16>(1), blk.lo, rows, o1);
        c.flush(c.out<f32>(2), blk.lo, rows, o2);
        c.flush(c.out<bf16>(3), blk.lo, rows, o3);
        c.flush(c.out<f16>(4), blk.lo, rows, o4);
        c.flush(c.out<int8_t>(5), blk.lo, rows, o5);
        c.dma_wait_all();
    }
}
