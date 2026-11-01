// copy_small: each core copies its 16 partitions with one iDMA fetch + flush (measures call overhead).
#include <gpsimd/gpsimd.hpp>
GPSIMD_OP(copy_small) {
    gp::Call c(call);
    auto x = c.in<int32_t>(0);
    auto o = c.out<int32_t>(0);
    int32_t buf[16 * 64];
    gp::Range r = c.partitions(o);
    c.fetch(x, r.lo, r.size(), buf);
    c.dma_wait(c.flush(o, r.lo, r.size(), buf));
}
