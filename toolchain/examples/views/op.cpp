// views: tile views with strides / partition offsets / in-place update, staged with Tile::fetch/flush
// (fetch() stages a view's whole span per partition; offset(i) locates element i inside it).
#include <gpsimd/gpsimd.hpp>

// out [64, 64] int32 view; in [64, 8, 4] int16 strided view (32 elements per partition).
// out[p][i] = 10 * in[p][i % 32]
GPSIMD_OP(views_scale) {
    gp::Call c(call);
    auto in = c.in<int16_t>(0);
    auto out = c.out<int32_t>(0);
    const uint32_t span = in.span(), n = out.elems();
    constexpr uint32_t ROWS = 8;
    int16_t li[ROWS * 256];
    int32_t lo[ROWS * 256];
    if (span > 256 || n > 256) { c.fail(8); return; }
    for (gp::Range blk : c.blocks(out.partitions(), ROWS)) {
        c.dma_wait(c.fetch(in, blk.lo, blk.size(), li));
        for (uint32_t q = 0; q < blk.size(); ++q)
            for (uint32_t i = 0; i < n; ++i) lo[q * n + i] = 10 * li[q * span + in.offset(i % in.elems())];
        c.dma_wait(c.flush(out, blk.lo, blk.size(), lo));
    }
}

GPSIMD_OP(views_inc) {
    gp::Call c(call);
    auto x = c.in<int32_t>(0);
    auto y = c.out<int32_t>(0);                                 // same SBUF region as x
    constexpr uint32_t CAP = 2048;
    int32_t l[CAP];
    if (x.span() > CAP) { c.fail(8); return; }
    for (gp::Range blk : c.blocks(x.partitions(), CAP / x.span())) {
        c.dma_wait(c.fetch(x, blk.lo, blk.size(), l));
        for (uint32_t q = 0; q < blk.size(); ++q)
            for (uint32_t i = 0; i < x.elems(); ++i) l[q * x.span() + x.offset(i)] += 7;
        c.dma_wait(c.flush(y, blk.lo, blk.size(), l));
    }
}
