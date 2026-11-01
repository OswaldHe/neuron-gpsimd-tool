/* Probe 5: call the firmware getter ctx[2]() and dump what it returns (+16 words there). */
#include "neuron_kl.h"
#include "tchan.h"
#define MAGIC 0xC0FFEE42u
#define NRES 20
typedef uint32_t (*fn0_t)(void);
static volatile uint32_t counter;
static volatile uint32_t *volatile ctx;
static volatile uint32_t res[NRES];
static volatile uint32_t done;
static void collect(void) {
    uint32_t r = ((fn0_t)ctx[2])();
    res[0] = r;
    res[1] = 0;
    if (r >= 0x80000 && r < 0xC0000) {
        for (uint32_t k = 0; k < 16; k++) res[2 + k] = ((volatile uint32_t *)r)[k];
        for (uint32_t k = 0; k < 16; k++) if (((volatile uint32_t *)r)[k] == MAGIC) res[1] = 0x100 + k;
    }
}
void dbg_kernel(void) {
    uint32_t i;
    if (!done) { collect(); done = 1; }
    if (tchan_preamble(&counter, &i)) return;
    tchan_emit_words(res, NRES, i);
}
NEURON_KERNEL(dbg_kernel);
void *_start(void *arg) { ctx = (volatile uint32_t *)arg; return 0; }
