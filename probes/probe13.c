/* Probe 13: which GPSIMD cores run the kernel? Data word k (k=0..7): nibbles all 0xF on the core
 * whose PRID == k, 0 on the others (timing shows the max over cores). Word 8: 0xC0DE0001 marker. */
#include "neuron_kl.h"
#include "tchan.h"
static volatile uint32_t counter;
void dbg_kernel(void) {
    uint32_t i, prid;
    __asm__ volatile("rsr.prid %0" : "=r"(prid));
    if (tchan_preamble(&counter, &i)) return;
    uint32_t w = i >> 3;
    uint32_t v = (w < 8) ? ((prid & 0xff) == w ? 0xFFFFFFFFu : 0u) : 0xC0DE0001u;
    tchan_word_nibble(v, i & 7);
}
NEURON_KERNEL(dbg_kernel);
void *_start(void *arg) { (void)arg; return 0; }
