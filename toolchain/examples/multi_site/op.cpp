// multi_site: two operators, three call sites; tiles staged through local RAM with the iDMA.
#include <gpsimd/gpsimd.hpp>

GPSIMD_OP(triple) {
    gp::Call c(call);
    auto x = c.in<int32_t>(0);
    auto y = c.out<int32_t>(0);
    constexpr uint32_t CAP = 4096;
    int32_t l[CAP];
    for (gp::Range blk : c.blocks(x.partitions(), CAP / x.elems())) {
        c.dma_wait(c.fetch(x, blk.lo, blk.size(), l));
        for (uint32_t k = 0; k < blk.size() * x.elems(); ++k) l[k] *= 3;
        c.dma_wait(c.flush(y, blk.lo, blk.size(), l));
    }
}

GPSIMD_OP(mix) {
    gp::Call c(call);
    auto a = c.in<int32_t>(0), b = c.in<int32_t>(1);
    auto y = c.out<int32_t>(0);
    constexpr uint32_t CAP = 4096;
    int32_t la[CAP], lb[CAP];
    for (gp::Range blk : c.blocks(a.partitions(), CAP / a.elems())) {
        c.fetch(a, blk.lo, blk.size(), la);
        c.dma_wait(c.fetch(b, blk.lo, blk.size(), lb));
        for (uint32_t k = 0; k < blk.size() * a.elems(); ++k) la[k] ^= lb[k];
        c.dma_wait(c.flush(y, blk.lo, blk.size(), la));
    }
    if (c.core() == 0) gp_printf("mix: site %u exec %u\n", c.raw()->site->site, c.exec());
}
