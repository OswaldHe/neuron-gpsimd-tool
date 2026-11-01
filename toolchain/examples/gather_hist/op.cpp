// gather_hist: gather from a table tile spread over 32 partitions + lock-protected global histogram.
// The whole table (16 KB) and the index rows are staged in local RAM with the iDMA; gathers are local.
#include <gpsimd/gpsimd.hpp>

GPSIMD_OP(gather_hist) {
    gp::Call c(call);
    auto idx = c.in<int32_t>(0);
    auto table = c.in<uint32_t>(1);                 // float32 entries copied as raw bits
    auto out = c.out<uint32_t>(0);
    auto hist = c.out<int32_t>(1);
    const uint32_t per_part = table.elems(), entries = table.partitions() * per_part, nb = hist.elems();
    const uint32_t n = idx.elems();
    constexpr uint32_t TCAP = 4096, CAP = 1024;
    uint32_t lt[TCAP], lo[CAP];
    int32_t li[CAP];
    if (entries > TCAP || n > CAP || nb > 64) { c.fail(8); return; }
    c.fetch(table, 0, table.partitions(), lt);       // whole table: one 2D descriptor
    uint32_t local[64] = {0};
    for (gp::Range blk : c.blocks(idx.partitions(), CAP / n)) {
        c.dma_wait(c.fetch(idx, blk.lo, blk.size(), li));
        for (uint32_t k = 0; k < blk.size() * n; ++k) {
            const uint32_t e = static_cast<uint32_t>(li[k]);
            if (e >= entries) { c.fail(8); continue; }
            lo[k] = lt[e];                            // dense [32][128] staging: entry e at lt[e]
            ++local[(e >> 6) % nb];
        }
        c.dma_wait(c.flush(out, blk.lo, blk.size(), lo));
    }
    if (c.core() == 0)
        for (uint32_t b = 0; b < nb; ++b) hist.store(0, b, 0);
    c.barrier();                                      // output zeroed before anyone accumulates
    c.lock();
    for (uint32_t b = 0; b < nb; ++b) hist.store(0, b, hist.load(0, b) + static_cast<int32_t>(local[b]));
    c.unlock();
}
