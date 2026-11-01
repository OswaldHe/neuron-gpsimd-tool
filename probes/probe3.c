/* Probe 3: scan 0x80000..0xC0000 (256 KB) for the KERNEL_CALL magic; record hits + bounds reached. */
#include "neuron_kl.h"
#include "tchan.h"
#define MAGIC 0xC0FFEE42u
#define NRES 12
static volatile uint32_t counter;
static volatile uint32_t *volatile ctx;
static volatile uint32_t res[NRES];
static volatile uint32_t done;
static void collect(void) {
    uint32_t n = 2;
    res[0] = (uint32_t)ctx;
    for (uint32_t a = 0x80000; a < 0xC0000 && n < NRES; a += 4)
        if (*(volatile uint32_t *)a == MAGIC) res[n++] = a;
    res[1] = n - 2;   /* number of hits */
    while (n < NRES) res[n++] = 0;
}
void dbg_kernel(void) {
    uint32_t i;
    if (!done) { collect(); done = 1; }
    if (tchan_preamble(&counter, &i)) return;
    tchan_emit_words(res, NRES, i);
}
NEURON_KERNEL(dbg_kernel);
void *_start(void *arg) { ctx = (volatile uint32_t *)arg; return 0; }
