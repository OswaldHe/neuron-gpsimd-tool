// dma_tiles: iDMA staging patterns. Cycle counts of core 0 go to receipt user words / the log.
#include <gpsimd/gpsimd.hpp>

GPSIMD_OP(dma_tiles) {
    gp::Call c(call);
    auto A = c.in<int32_t>(0);
    auto B = c.in<int16_t>(1);                    // strided view [128, 8, 4] of [128, 8, 16]
    auto C8 = c.in<int8_t>(2);
    auto T = c.in<int32_t>(3);
    auto IDX = c.in<int32_t>(4);
    auto params = c.in<int32_t>(5);
    auto O0 = c.out<int32_t>(0), O1 = c.out<int32_t>(1), O3 = c.out<int32_t>(3), O4 = c.out<int32_t>(4);
    auto O5 = c.out<int32_t>(5), O6 = c.out<int32_t>(6);
    auto O2 = c.out<int8_t>(2);
    const gp::Range r = c.partitions(A);          // 16 partitions per core
    const uint32_t np = r.size();
    uint32_t t0, cyc[6];

    // 0: one 2D descriptor in, compute in local RAM, one out
    static_assert(16 * 512 * 4 == 32768, "");
    int32_t buf[16 * 512];                        // 32 KB on the stack
    t0 = gp_cycles();
    c.dma_wait(c.fetch(A, r.lo, np, buf));
    for (uint32_t k = 0; k < np * 512; ++k) buf[k] = buf[k] * 3 + 1;
    c.dma_wait(c.flush(O0, r.lo, np, buf));
    cyc[0] = gp_cycles() - t0;

    // 1: strided view: stage the span (each partition: offset(i) indexes the staged row)
    {
        const uint32_t span = B.span();
        int16_t sb[16 * 128];
        int32_t ob[16 * 32];
        c.dma_wait(c.fetch(B, r.lo, np, sb));
        for (uint32_t q = 0; q < np; ++q)
            for (uint32_t i = 0; i < B.elems(); ++i) ob[q * 32 + i] = 7 * sb[q * span + B.offset(i)];
        c.dma_wait(c.flush(O1, r.lo, np, ob));
    }
    // 2: 37-byte rows
    {
        int8_t cb[16 * 37];
        c.dma_wait(c.fetch(C8, r.lo, np, cb));
        for (uint32_t k = 0; k < np * 37; ++k) cb[k] ^= 0x5A;
        c.dma_wait(c.flush(O2, r.lo, np, cb));
    }
    // 3: SBUF -> SBUF, 16 partitions x 2 KB, no local copy
    t0 = gp_cycles();
    c.dma_wait(c.dma_2d(O3.row(r.lo), A.row(r.lo), 512 * 4, np, GP_PARTITION_STRIDE, GP_PARTITION_STRIDE));
    cyc[1] = gp_cycles() - t0;
    // 4: transpose: column j of T (128 partitions x 4 bytes) -> local, then the row of O4
    t0 = gp_cycles();
    {
        int32_t col[2][128];
        uint32_t tk[2] = {0, 0};
        tk[0] = c.dma_2d(col[0], T.row(0) + r.lo, 4, 128, GP_PARTITION_STRIDE, 4);
        for (uint32_t j = r.lo; j < r.hi; ++j) {
            const uint32_t cur = (j - r.lo) & 1;
            if (j + 1 < r.hi) tk[cur ^ 1] = c.dma_2d(col[cur ^ 1], T.row(0) + j + 1, 4, 128, GP_PARTITION_STRIDE, 4);
            c.dma_wait(tk[cur]);
            c.dma_wait(c.dma(O4.row(j), col[cur], 128 * 4));
        }
    }
    cyc[2] = gp_cycles() - t0;
    // 5: double buffering: 4 chunks of 4 partitions, compute on one buffer while the next one arrives
    t0 = gp_cycles();
    {
        int32_t big[16384];                       // 64 KB on the stack (2 x 4 x 2 KB used + headroom test)
        int32_t *bufs[2] = {big, big + 4 * 512};
        uint32_t in_t[2] = {0, 0}, out_t[2] = {0, 0};
        in_t[0] = c.fetch(A, r.lo, 4, bufs[0]);
        for (uint32_t k = 0; k < np / 4; ++k) {
            const uint32_t cur = k & 1, nxt = cur ^ 1;
            if (k + 1 < np / 4) {
                if (out_t[nxt]) c.dma_wait(out_t[nxt]);                  // buffer free again?
                in_t[nxt] = c.fetch(A, r.lo + 4 * (k + 1), 4, bufs[nxt]);
            }
            c.dma_wait(in_t[cur]);
            for (uint32_t e = 0; e < 4 * 512; ++e) bufs[cur][e] -= 5;
            out_t[cur] = c.flush(O5, r.lo + 4 * k, 4, bufs[cur]);
        }
        c.dma_wait_all();
        big[16383] = 1;                           // touch the far end of the 64 KB buffer
        gp_receipt_user(3, static_cast<uint32_t>(big[16383]));
    }
    cyc[3] = gp_cycles() - t0;
    // 6: HBM gather: 16-byte aligned blocks around each index fetched by iDMA (one descriptor each)
    t0 = gp_cycles();
    {
        volatile int32_t *table = gp::hbm_ptr<int32_t>(params, 0);
        int32_t ib[16 * 64], ob[16 * 64];
        alignas(16) int32_t blk[4];
        c.dma_wait(c.fetch(IDX, r.lo, np, ib));
        for (uint32_t k = 0; k < np * 64; ++k) {
            const uint32_t i = static_cast<uint32_t>(ib[k]);
            c.dma_wait(c.dma(blk, table + (i & ~3u), 16));
            ob[k] = blk[i & 3];
        }
        c.dma_wait(c.flush(O6, r.lo, np, ob));
    }
    cyc[4] = gp_cycles() - t0;
    gp_receipt_user(0, cyc[0]);
    gp_receipt_user(1, cyc[2]);
    gp_receipt_user(2, cyc[4]);
    if (c.core() == 0 && c.exec() == 1)
        gp_printf("cycles: rows 64KB in+compute+out %u, sbuf->sbuf 32KB %u, transpose 16 cols %u, double-buffer %u, hbm gather 1024 %u\n",
                  cyc[0], cyc[1], cyc[2], cyc[3], cyc[4]);
}
