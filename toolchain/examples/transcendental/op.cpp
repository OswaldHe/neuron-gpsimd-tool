// transcendental: gp::math functions on every element; inputs/outputs staged through local RAM (iDMA).
#include <gpsimd/math.hpp>

namespace m = gp::math;
using gp::f32;

GPSIMD_OP(transcendental) {
    gp::Call c(call);
    auto X = c.in<f32>(0), Y = c.in<f32>(1);
    const uint32_t n = X.elems();
    constexpr uint32_t CAP = 1024;                       // elements per buffer; 10 buffers = 40 KB
    uint32_t lx[CAP], ly[CAP], lo[8][CAP];
    if (n > CAP) { c.fail(8); return; }
    for (gp::Range blk : c.blocks(X.partitions(), CAP / n)) {
        const uint32_t rows = blk.size();
        c.fetch(X, blk.lo, rows, lx);
        c.dma_wait(c.fetch(Y, blk.lo, rows, ly));
        for (uint32_t k = 0; k < rows * n; ++k) {
            const f32 x = f32::from_bits(lx[k]), y = f32::from_bits(ly[k]);
            lo[0][k] = m::exp(x).bits;
            lo[1][k] = m::log(y).bits;
            lo[2][k] = m::tanh(x).bits;
            lo[3][k] = m::sigmoid(x).bits;
            lo[4][k] = m::sin(x).bits;
            lo[5][k] = m::cos(x).bits;
            lo[6][k] = m::gelu(x).bits;
            lo[7][k] = m::rsqrt(y).bits;
        }
        for (uint32_t j = 0; j < 8; ++j) c.flush(c.out<f32>(j), blk.lo, rows, lo[j]);
        c.dma_wait_all();
    }
}
