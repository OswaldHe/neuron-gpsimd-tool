/* ccount: does rsr.ccount work on the trn3 GPSIMD core? */
#include "neuron_kl.h"
#include "kprint.h"
void dbg_kernel(void) {
    uint32_t c0, c1, sp;
    __asm__ volatile("rsr.ccount %0" : "=r"(c0));
    for (volatile int i = 0; i < 1000; i++) { }
    __asm__ volatile("rsr.ccount %0\n mov %1, a1" : "=r"(c1), "=r"(sp));
    kprintf("core %u ccount %u -> %u (delta %u) sp=0x%08x fn=0x%08x\n", kprid(), c0, c1, c1 - c0, sp, (uint32_t)&dbg_kernel);
}
NEURON_KERNEL(dbg_kernel);
void *_start(void *arg) { (void)arg; return 0; }
