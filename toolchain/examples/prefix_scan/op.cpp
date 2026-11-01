// prefix_scan: element-split inclusive scan with a cross-core exclusive scan of the block totals. Each
// core moves its contiguous block of the row with single iDMA copies.
#include <gpsimd/gpsimd.hpp>

template <typename In>
static void scan(const gp::Call &c, const gp::Tile<In> &x, const gp::Tile<int32_t> &y) {
    using S = typename gp::Tile<In>::S;
    const gp::Range r = c.split(x.elems());              // this core's contiguous block of elements
    constexpr uint32_t CAP = 2048;
    S lx[CAP];
    int32_t ly[CAP];
    if (r.size() > CAP || !x.contiguous() || !y.contiguous()) { c.fail(8); return; }
    c.dma_wait(c.dma(lx, x.row(0) + r.lo, r.size() * sizeof(S)));
    uint32_t run = 0;
    for (uint32_t i = 0; i < r.size(); ++i) {
        run += static_cast<uint32_t>(static_cast<int32_t>(lx[i]));
        ly[i] = static_cast<int32_t>(run);
    }
    const uint32_t offset = c.exscan_add(run);           // sum of the blocks of cores 0..core-1
    for (uint32_t i = 0; i < r.size(); ++i) ly[i] = static_cast<int32_t>(static_cast<uint32_t>(ly[i]) + offset);
    c.dma_wait(c.dma(y.row(0) + r.lo, ly, r.size() * 4));
}

GPSIMD_OP(prefix_scan) {
    gp::Call c(call);
    scan(c, c.in<int32_t>(0), c.out<int32_t>(0));
    scan(c, c.in<int16_t>(1), c.out<int32_t>(1));         // [1, 64, 128] tile: flat index over both free dims
}
