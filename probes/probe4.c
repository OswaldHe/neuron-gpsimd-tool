/* Probe 4: dump firmware instruction RAM around the context's function pointers. */
#include "neuron_kl.h"
#include "tchan.h"
#define NRES 44
static volatile uint32_t counter;
static volatile uint32_t res[NRES];
static volatile uint32_t done;
static void collect(void) {
    uint32_t n = 0;
    for (uint32_t a = 0x3620; a < 0x3680; a += 4) res[n++] = *(volatile uint32_t *)a;   /* 24 words */
    for (uint32_t a = 0x3090; a < 0x30e0; a += 4) res[n++] = *(volatile uint32_t *)a;   /* 20 words */
}
void dbg_kernel(void) {
    uint32_t i;
    if (!done) { collect(); done = 1; }
    if (tchan_preamble(&counter, &i)) return;
    tchan_emit_words(res, NRES, i);
}
NEURON_KERNEL(dbg_kernel);
void *_start(void *arg) { (void)arg; return 0; }
