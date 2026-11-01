/* Probe 7: argument registers a2..a7 as passed by the firmware dispatcher, the 16 words each
 * plausible data-RAM pointer points to (first two), and the dispatcher code around 0x1400. */
#include "neuron_kl.h"
#include "tchan.h"
#define NRES 64
static volatile uint32_t counter;
static volatile uint32_t res[NRES];
static volatile uint32_t done;
static int is_dram(uint32_t p) { return p >= 0x80000 && p < 0xC0000 && !(p & 3); }
void dbg_kernel(uint32_t a2, uint32_t a3, uint32_t a4, uint32_t a5, uint32_t a6, uint32_t a7) {
    uint32_t i;
    if (!done) {
        uint32_t args[6] = {a2, a3, a4, a5, a6, a7};
        uint32_t n = 0;
        for (int k = 0; k < 6; k++) res[n++] = args[k];
        int dumped = 0;
        for (int k = 0; k < 6 && dumped < 2; k++) {
            if (!is_dram(args[k])) continue;
            for (int j = 0; j < 12; j++) res[n++] = ((volatile uint32_t *)args[k])[j];
            dumped++;
        }
        n = 30;
        for (uint32_t a = 0x1420; a < 0x14a0 && n < NRES; a += 4) res[n++] = *(volatile uint32_t *)a;
        done = 1;
    }
    if (tchan_preamble(&counter, &i)) return;
    tchan_emit_words(res, NRES, i);
}
NEURON_KERNEL(dbg_kernel);
void *_start(void *arg) { (void)arg; return 0; }
