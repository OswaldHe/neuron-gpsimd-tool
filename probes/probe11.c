/* Probe 11: read the core-local register block (trn1 layout: +0x28/+0x2c own SoC base,
 * +0x218/+0x228/+0x230 translation-window registers). */
#include "neuron_kl.h"
#include "tchan.h"
#define NRES 32
static volatile uint32_t counter;
static volatile uint32_t res[NRES];
static volatile uint32_t done;
void dbg_kernel(void) {
    uint32_t i;
    if (!done) {
        for (uint32_t k = 0; k < 16; k++) res[k] = ((volatile uint32_t *)0x00100000u)[k];
        for (uint32_t k = 0; k < 16; k++) res[16 + k] = ((volatile uint32_t *)0x00100200u)[k];
        done = 1;
    }
    if (tchan_preamble(&counter, &i)) return;
    tchan_emit_words(res, NRES, i);
}
NEURON_KERNEL(dbg_kernel);
void *_start(void *arg) { (void)arg; return 0; }
