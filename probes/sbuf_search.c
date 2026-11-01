/* sbuf_search: find a core-local window onto SBUF. The host plants 0x5B0F5B0F at SBUF p0+0x20000 and
 * 0x5B0F0001 at p1+0x20000 (TPB offsets 0x20000 / 0x60000). For each candidate base B (CANDS),
 * log "probe B" *before* touching it (post-mortem log survives a hang), then read a few offsets
 * covering several plausible layouts (TPB-linear, partition stride 0x40000 or 0x30000 or 0x10000). */
#include "neuron_kl.h"
#include "kprint.h"
#ifndef CANDS
#define CANDS 0x88000000u, 0x90000000u, 0xA0000000u, 0xC0000000u, 0xE0000000u, 0x40000000u, 0x60000000u
#endif
static const uint32_t cands[] = {CANDS};
static const uint32_t offs[] = {0x20000u, 0x60000u, 0x50000u, 0x30000u, 0x3FE04u};
void dbg_kernel(void) {
    if (kprid() != 0) return;
    for (unsigned i = 0; i < sizeof(cands) / sizeof(cands[0]); i++) {
        uint32_t b = cands[i];
        kprintf("probe %08x:", b);
        for (unsigned k = 0; k < sizeof(offs) / sizeof(offs[0]); k++) {
            uint32_t v = *(volatile uint32_t *)(b + offs[k]);
            kprintf(" +%x=%08x%s", offs[k], v, (v >> 16) == 0x5B0F ? "<==" : "");
        }
        kprintf("\n");
    }
    kprintf("all probes done\n");
}
NEURON_KERNEL(dbg_kernel);
void *_start(void *arg) { (void)arg; return 0; }
