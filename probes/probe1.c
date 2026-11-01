/* Probe 1: dump the _start argument and the first 16 words it points to. */
#include "neuron_kl.h"
#include "tchan.h"
static volatile uint32_t counter;
static volatile uint32_t *volatile start_arg;
void dbg_kernel(void) {
    uint32_t i;
    if (tchan_preamble(&counter, &i)) return;
    uint32_t w = i >> 3;
    uint32_t v = (w == 0) ? (uint32_t)start_arg : start_arg[w - 1];
    tchan_word_nibble(v, i & 7);
}
NEURON_KERNEL(dbg_kernel);
void *_start(void *arg) { start_arg = (volatile uint32_t *)arg; return 0; }
