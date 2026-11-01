#include <gpsimd/gpsimd.hpp>

static inline uint32_t rer(uint32_t a) { uint32_t v; __asm__ volatile("rer %0, %1" : "=r"(v) : "r"(a) : "memory"); return v; }

GPSIMD_OP(mem2) {
    gp::Call c(call);
    auto X = c.in<int32_t>(0);
    auto O = c.out<int32_t>(0);
    const bool rep = c.core() == 0;
    const uint32_t N = 1024;
    uint32_t a = reinterpret_cast<uint32_t>(X.row(c.core() * 16)), acc = 0;
    uint32_t d = reinterpret_cast<uint32_t>(O.row(c.core() * 16 + 2));
    uint32_t t0, t1;
    // 8 independent loads per iteration, no memw
    t0 = gp_cycles();
    for (uint32_t i = 0; i < N; i += 8) {
        uint32_t v0, v1, v2, v3, v4, v5, v6, v7, p = a + 4 * i;
        __asm__ volatile("l32i %0, %8, 0\nl32i %1, %8, 4\nl32i %2, %8, 8\nl32i %3, %8, 12\n"
                         "l32i %4, %8, 16\nl32i %5, %8, 20\nl32i %6, %8, 24\nl32i %7, %8, 28\n"
                         : "=&r"(v0), "=&r"(v1), "=&r"(v2), "=&r"(v3), "=&r"(v4), "=&r"(v5), "=&r"(v6), "=&r"(v7) : "r"(p));
        acc += v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7;
    }
    t1 = gp_cycles();
    if (rep) gp_printf("SBUF load, 8 back-to-back, no memw: %u cycles/load\n", (t1 - t0) / N);
    // stores without memw
    t0 = gp_cycles();
    for (uint32_t i = 0; i < N; i += 8) {
        uint32_t p = d + 4 * i;
        __asm__ volatile("s32i %1, %0, 0\ns32i %1, %0, 4\ns32i %1, %0, 8\ns32i %1, %0, 12\n"
                         "s32i %1, %0, 16\ns32i %1, %0, 20\ns32i %1, %0, 24\ns32i %1, %0, 28\n" : : "r"(p), "r"(i) : "memory");
    }
    t1 = gp_cycles();
    if (rep) gp_printf("SBUF store, 8 back-to-back, no memw: %u cycles/store\n", (t1 - t0) / N);
    t0 = gp_cycles();
    __asm__ volatile("memw" ::: "memory");
    t1 = gp_cycles();
    if (rep) gp_printf("memw after stores: %u cycles\n", t1 - t0);
    // copy (load+store) without memw
    t0 = gp_cycles();
    for (uint32_t i = 0; i < N; ++i) {
        uint32_t v;
        __asm__ volatile("l32i %0, %1, 0\ns32i %0, %2, 0" : "=&r"(v) : "r"(a + 4 * i), "r"(d + 4 * i) : "memory");
    }
    __asm__ volatile("memw" ::: "memory");
    t1 = gp_cycles();
    if (rep) gp_printf("SBUF copy word, no memw: %u cycles/word\n", (t1 - t0) / N);
    gp_receipt_user(0, acc);
    // output = input (bit-exact check)
    for (uint32_t p : c.partitions(O))
        for (uint32_t i = 0; i < O.elems(); ++i) O.row(p)[i] = X.row(p)[i];
    c.barrier();
    // iDMA registers (ERI 0x110000)
    if (rep) {
        gp_printf("reading iDMA registers...\n");
        static const uint16_t regs[] = {0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x40, 0x44, 0x48, 0x4C, 0x50};
        for (uint32_t k = 0; k < sizeof(regs) / sizeof(regs[0]); ++k)
            gp_printf("  idma[0x%02x] = 0x%08x\n", regs[k], rer(0x110000u + regs[k]));
    }
}
