/* kprint demo: every GPSIMD core logs its id and the firmware-supplied arguments. */
#include "neuron_kl.h"
#include "kprint.h"
static uint32_t ctx;
void dbg_print(uint32_t a2, uint32_t a3, uint32_t a4, uint32_t a5, uint32_t a6, uint32_t a7) {
    kprintf("hello from GPSIMD core %u: a2=0x%08x a3=%x a4=%u a5=0x%08x a6=0x%x a7=%x ctx=0x%08x\n",
            kprid(), a2, a3, a4, a5, a6, a7, ctx);
}
NEURON_KERNEL(dbg_print);
void *_start(void *arg) { ctx = (uint32_t)arg; return 0; }
