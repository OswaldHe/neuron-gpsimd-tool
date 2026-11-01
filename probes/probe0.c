/* Probe 0: does .bss persist across executions, is _start called, and with what argument? */
#include "neuron_kl.h"
#include "tchan.h"
static volatile uint32_t counter;
static volatile uint32_t start_arg = 0x0badc0de;
static volatile uint32_t start_calls;
void dbg_kernel(void) {
    uint32_t i;
    if (tchan_preamble(&counter, &i)) return;
    uint32_t w = i >> 3;
    uint32_t v = (w == 0) ? start_arg : (w == 1) ? start_calls : 0x12345678u;
    tchan_word_nibble(v, i & 7);
}
NEURON_KERNEL(dbg_kernel);
void *_start(void *arg) { start_arg = (uint32_t)arg; start_calls++; return 0; }
