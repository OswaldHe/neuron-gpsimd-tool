/* Probe 2: find where the KERNEL_CALL instruction lives in data RAM (magic in its argument
 * bytes), and dump more of the _start context and the block at context[4]. */
#include "neuron_kl.h"
#include "tchan.h"
#define MAGIC 0xC0FFEE42u
#define NRES 40
static volatile uint32_t counter;
static volatile uint32_t *volatile ctx;
static volatile uint32_t res[NRES];
static volatile uint32_t done;
static void collect(void) {
    uint32_t n = 0;
    res[n++] = (uint32_t)ctx;
    for (uint32_t a = 0x80000; a < 0x90000 && n < 9; a += 4)      /* up to 8 hits */
        if (*(volatile uint32_t *)a == MAGIC) res[n++] = a;
    while (n < 9) res[n++] = 0;
    for (uint32_t k = 14; k < 30; k++) res[n++] = ctx[k];          /* context words 14..29 */
    volatile uint32_t *blk = (volatile uint32_t *)ctx[4];
    for (uint32_t k = 0; k < 15; k++) res[n++] = blk[k];           /* block at ctx[4] */
}
void dbg_kernel(void) {
    uint32_t i;
    if (!done) { collect(); done = 1; }
    if (tchan_preamble(&counter, &i)) return;
    tchan_emit_words(res, NRES, i);
}
NEURON_KERNEL(dbg_kernel);
void *_start(void *arg) { ctx = (volatile uint32_t *)arg; return 0; }
