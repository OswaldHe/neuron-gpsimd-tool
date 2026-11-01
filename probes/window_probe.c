/* window_probe: look for a core-local window onto the TPB address space. For candidate bases B
 * (64 MB aligned, B_LO..B_HI), read B + offsets where the host knows the content:
 *   +0x3001010  KERNEL_CALL operand magic 0xC0FFEE42 (Pool instruction ring, while in flight)
 *   +0x0020000  SBUF p0 off 0x20000, magic 0x5B0F5B0F written by the host over BAR0
 *   +0x3180878  core-0 data RAM word 0x80878 (firmware ctx, value 0x10)
 * Only core 0 probes; results via kprintf. */
#include "neuron_kl.h"
#include "kprint.h"
#ifndef B_LO
#define B_LO 0x80000000u
#endif
#ifndef B_HI
#define B_HI 0xC0000000u
#endif
void dbg_kernel(void) {
    if (kprid() != 0) return;
    for (uint32_t b = B_LO; b < B_HI && b >= B_LO; b += 0x4000000u) {
        uint32_t r = *(volatile uint32_t *)(b + 0x3001010u);
        uint32_t s = *(volatile uint32_t *)(b + 0x0020000u);
        uint32_t c = *(volatile uint32_t *)(b + 0x3180878u);
        kprintf("B=%08x ring=%08x sbuf=%08x ctx=%08x%s\n", b, r, s, c,
                (r == 0xC0FFEE42u || s == 0x5B0F5B0Fu || c == 0x10u) ? "   <== HIT" : "");
    }
}
NEURON_KERNEL(dbg_kernel);
void *_start(void *arg) { (void)arg; return 0; }
