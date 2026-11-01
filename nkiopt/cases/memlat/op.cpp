// memlat: cycles per load for stack arrays (independent / dependent), .bss, and DataRAM low region.
#include <gpsimd/gpsimd.hpp>
static inline uint32_t ccount() { uint32_t v; __asm__ volatile("rsr.ccount %0" : "=a"(v)); return v; }
static volatile uint32_t bss[1024];
#define TIME(name, n, body) do { uint32_t t0 = ccount(); body; uint32_t t1 = ccount(); \
    if (c.core() == 0 && c.exec() == 1) gp_printf("%-34s %4u cycles/op (x%u)\n", name, (t1 - t0) / (n), sink); } while (0)
GPSIMD_OP(memlat) {
    gp::Call c(call);
    auto x = c.in<int32_t>(0);
    auto o = c.out<int32_t>(0);
    uint32_t stk[1024];
    for (uint32_t i = 0; i < 1024; ++i) { stk[i] = (i * 7 + 1) & 1023; bss[i] = (i * 7 + 1) & 1023; }
    volatile uint32_t *vs = stk;
    uint32_t sink = 0;
    TIME("stack load, independent (volatile)", 1024, for (uint32_t i = 0; i < 1024; ++i) sink += vs[i]);
    TIME("stack load, 8 unrolled", 1024, for (uint32_t i = 0; i < 1024; i += 8) sink += vs[i] + vs[i+1] + vs[i+2] + vs[i+3] + vs[i+4] + vs[i+5] + vs[i+6] + vs[i+7]);
    TIME("stack load, dependent chain", 1024, { uint32_t p = 0; for (uint32_t i = 0; i < 1024; ++i) p = vs[p]; sink += p; });
    TIME("stack store", 1024, for (uint32_t i = 0; i < 1024; ++i) vs[i] = i);
    TIME(".bss load, independent", 1024, for (uint32_t i = 0; i < 1024; ++i) sink += bss[i]);
    TIME(".bss load, dependent chain", 1024, { uint32_t p = 0; for (uint32_t i = 0; i < 1024; ++i) p = bss[p]; sink += p; });
    TIME("loop + nop", 1024, for (uint32_t i = 0; i < 1024; ++i) __asm__ volatile("nop"));
    TIME("alu chain (add/xor)", 1024, { uint32_t a = sink; for (uint32_t i = 0; i < 1024; ++i) { a = (a ^ i) + 0x9e37; } sink += a; });
    volatile uint32_t *lowram = reinterpret_cast<volatile uint32_t *>(0x00080000u);
    TIME("DataRAM 0x80000 load (fw region)", 256, for (uint32_t i = 0; i < 256; ++i) sink += lowram[i]);
    if (c.core() == 0 && c.exec() == 1) gp_printf("stack at %p, bss at %p\n", (void *)stk, (void *)bss);
    int32_t buf[16 * 64];
    gp::Range r = c.partitions(o);
    c.fetch(x, r.lo, r.size(), buf);
    c.dma_wait(c.flush(o, r.lo, r.size(), buf));
    gp_receipt_user(0, sink);
}
