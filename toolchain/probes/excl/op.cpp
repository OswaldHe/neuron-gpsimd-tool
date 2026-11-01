#include <gpsimd/gpsimd.hpp>
static volatile uint32_t local_word = 5;
static uint32_t try_excl(volatile uint32_t *p, uint32_t *old) {
    uint32_t o, ok;
    __asm__ volatile("l32ex %0, %2\naddi %1, %0, 1\ns32ex %1, %2\ngetex %1\n" : "=&r"(o), "=&r"(ok) : "r"(p) : "memory");
    *old = o;
    return ok;
}
GPSIMD_OP(probe_excl) {
    gp::Call c(call);
    auto o = c.out<int32_t>(0);
    if (c.core() != 0) return;
    uint32_t old = 0, ok;
    gp_printf("start\n");
    (void)local_word;
    volatile uint32_t *s = reinterpret_cast<volatile uint32_t *>(o.row(0));
    s[0] = 41;
    gp_printf("sbuf before: %u\n", s[0]);
    ok = try_excl(s, &old);
    gp_printf("sbuf: ok %u old %u now %u\n", ok, old, s[0]);
    for (int k = 0; k < 4; ++k) { ok = try_excl(s, &old); gp_printf("sbuf retry %d: ok %u old %u now %u\n", k, ok, old, s[0]); }
    for (int i = 0; i < 16; ++i) o.row(0)[i] = 0;
}
