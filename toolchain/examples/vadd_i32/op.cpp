// vadd_i32: out = a + b (int32 wrap-around). Each core stages blocks of its partitions through local
// data RAM with the iDMA (one 2D descriptor per tile and block), adds there, and writes the block back.
#include <gpsimd/gpsimd.hpp>

GPSIMD_OP(vadd_i32) {
    gp::Call c(call);
    auto a = c.in<int32_t>(0), b = c.in<int32_t>(1);
    auto o = c.out<int32_t>(0);
    const uint32_t n = o.elems();
    constexpr uint32_t CAP = 4096;                       // elements per staging buffer (16 KB, stack)
    int32_t la[CAP], lb[CAP];
    uint32_t sum = 0;
    if (!a.contiguous() || !b.contiguous() || !o.contiguous() || n > CAP) { c.fail(8); return; }
    for (gp::Range blk : c.blocks(o.partitions(), CAP / n)) {
        const uint32_t rows = blk.size(), m = rows * n;
        c.fetch(a, blk.lo, rows, la);
        c.dma_wait(c.fetch(b, blk.lo, rows, lb));         // in-order: a's block is there too
        for (uint32_t k = 0; k < m; ++k) {
            la[k] = static_cast<int32_t>(static_cast<uint32_t>(la[k]) + static_cast<uint32_t>(lb[k]));
            sum += static_cast<uint32_t>(la[k]);
        }
        c.dma_wait(c.flush(o, blk.lo, rows, la));
    }
    gp_receipt_user(0, sum);
    if (c.core() == 0) gp_printf("vadd_i32 exec %u: %u partitions x %u\n", c.exec(), o.partitions(), o.elems());
}
