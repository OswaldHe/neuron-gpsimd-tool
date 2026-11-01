/* scan80: scan core-local 0x80000000..0x84000000 (64 MB, reads as zeros at known SBUF offsets) for
 * any non-zero word, the host-planted SBUF magic 0x5B0F5B0F, and the input tile (0.25f then 0.5f). */
#include "neuron_kl.h"
#include "kprint.h"
void dbg_kernel(void) {
    if (kprid() != 0) return;
    uint32_t nz = 0, shown = 0;
    for (uint32_t a = 0x80000000u; a < 0x84000000u; a += 4) {
        uint32_t v = *(volatile uint32_t *)a;
        if (!v) continue;
        nz++;
        int interesting = v == 0x5B0F5B0Fu || (v == 0x3e800000u && *(volatile uint32_t *)(a + 4) == 0x3f000000u);
        if (shown < 12 || (interesting && shown < 40)) {
            kprintf("%08x: %08x%s\n", a, v, interesting ? "  <==" : "");
            shown++;
        }
    }
    kprintf("non-zero words in 0x80000000-0x84000000: %u\n", nz);
}
NEURON_KERNEL(dbg_kernel);
void *_start(void *arg) { (void)arg; return 0; }
