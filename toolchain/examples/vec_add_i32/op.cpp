// vec_add_i32: out = a + b (int32 wrap-around) with trn3 vector instructions (gpsimd/vec.h).
// Vector loads/stores need 64-byte-aligned addresses (misaligned ones raise an address exception), and NKI
// places tiles on 32-byte boundaries, so each core stages blocks of its partitions through 64-byte-aligned
// local-RAM buffers with the iDMA (as the scalar vadd_i32 example does) and adds 16 int32 per instruction:
//   v0 <- [a8]; v1 <- [a9]; v2 = v0 + v1; [a10] <- v2   (all three pointers += 64)
#include <gpsimd/gpsimd.hpp>
#include <gpsimd/vec.h>

static inline int32_t *align64(int32_t *p) {
    return reinterpret_cast<int32_t *>((reinterpret_cast<uintptr_t>(p) + 63) & ~static_cast<uintptr_t>(63));
}

// x[i] += y[i] for nblk blocks of 16 int32; x, y 64-byte aligned
static inline void vadd_blocks(int32_t *x, const int32_t *y, uint32_t nblk) {
    register int32_t *pa __asm__("a8") = x;
    register const int32_t *pb __asm__("a9") = y;
    register int32_t *po __asm__("a10") = x;
    for (uint32_t k = 0; k < nblk; ++k)
        __asm__ volatile(GPV_LD_IP(0, 8) GPV_LD_IP(1, 9) GPV_ADD32(2, 0, 1) GPV_ST_IP(2, 10)
                         : "+r"(pa), "+r"(pb), "+r"(po) : : "memory");
}

GPSIMD_OP(vec_add_i32) {
    gp::Call c(call);
    auto a = c.in<int32_t>(0), b = c.in<int32_t>(1);
    auto o = c.out<int32_t>(0);
    const uint32_t n = o.elems();
    constexpr uint32_t CAP = 4096;                       // elements per staging buffer (16 KB, stack)
    int32_t bufa[CAP + 16], bufb[CAP + 16];
    int32_t *la = align64(bufa), *lb = align64(bufb);
    if (!a.contiguous() || !b.contiguous() || !o.contiguous() || n > CAP || n % 16) { c.fail(8); return; }
    const uint32_t t0 = gp_cycles();
    uint32_t compute = 0;
    for (gp::Range blk : c.blocks(o.partitions(), CAP / n)) {
        const uint32_t rows = blk.size(), m = rows * n;
        c.fetch(a, blk.lo, rows, la);
        c.dma_wait(c.fetch(b, blk.lo, rows, lb));         // in-order: a's block is there too
        const uint32_t s = gp_cycles();
#ifdef VEC_SCALAR
        for (uint32_t k = 0; k < m; ++k)
            la[k] = static_cast<int32_t>(static_cast<uint32_t>(la[k]) + static_cast<uint32_t>(lb[k]));
#else
        vadd_blocks(la, lb, m / 16);
#endif
        compute += gp_cycles() - s;
        c.dma_wait(c.flush(o, blk.lo, rows, la));
    }
    gp_receipt_user(0, compute);
    gp_receipt_user(1, gp_cycles() - t0);
    if (c.core() == 0)
        gp_printf("vec_add_i32 exec %u: %u partitions x %u, core0 add %u cycles, total %u\n", c.exec(), o.partitions(), n,
                  compute, gp_cycles() - t0);
}
