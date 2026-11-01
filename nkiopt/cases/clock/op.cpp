// spin: busy-wait 1,000,000 core cycles (rsr.ccount) on every core, then copy the tile.
#include <gpsimd/gpsimd.hpp>
static inline uint32_t ccount() { uint32_t v; __asm__ volatile("rsr.ccount %0" : "=a"(v)); return v; }
GPSIMD_OP(spin) {
    gp::Call c(call);
    auto x = c.in<int32_t>(0);
    auto o = c.out<int32_t>(0);
    const uint32_t t0 = ccount();
    while (ccount() - t0 < 1000000u) {}
    int32_t buf[16 * 64];
    gp::Range r = c.partitions(o);
    c.fetch(x, r.lo, r.size(), buf);
    c.dma_wait(c.flush(o, r.lo, r.size(), buf));
}
