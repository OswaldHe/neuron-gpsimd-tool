/* spin: keep the KERNEL_CALL in flight (~20 s) so the host can inspect device state over BAR0. */
#include "neuron_kl.h"
#include "kprint.h"
void dbg_kernel(uint32_t a2, uint32_t a3, uint32_t a4, uint32_t a5, uint32_t a6, uint32_t a7) {
    uint32_t sp, ra;
    __asm__ volatile("mov %0, a1\n mov %1, a0" : "=r"(sp), "=r"(ra));
    kprintf("core %u spinning: sp=0x%08x ra=0x%08x a2=0x%08x a4=%u\n", kprid(), sp, ra, a2, a4);
    (void)a3; (void)a5; (void)a6; (void)a7;
    for (volatile uint32_t i = 0; i < 800000000u; i++) { }
    kprintf("core %u done\n", kprid());
}
NEURON_KERNEL(dbg_kernel);
void *_start(void *arg) { (void)arg; return 0; }
