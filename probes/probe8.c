/* Probe 8: with a 0x86 payload slot after the KERNEL_CALL, dump the a2 buffer (24 words) and
 * scan data RAM for the payload magic 0xFEEDF00D. */
#include "neuron_kl.h"
#include "tchan.h"
#define NRES 40
static volatile uint32_t counter;
static volatile uint32_t res[NRES];
static volatile uint32_t done;
void dbg_kernel(uint32_t a2, uint32_t a3, uint32_t a4, uint32_t a5, uint32_t a6, uint32_t a7) {
    uint32_t i;
    if (!done) {
        uint32_t n = 0;
        res[n++] = a2; res[n++] = a3; res[n++] = a4; res[n++] = a5; res[n++] = a6; res[n++] = a7;
        for (int j = 0; j < 24; j++) res[n++] = (a2 >= 0x80000 && a2 < 0xC0000) ? ((volatile uint32_t *)a2)[j] : 0;
        uint32_t hits = 0, first = 0;
        for (uint32_t a = 0x80000; a < 0xC0000; a += 4)
            if (*(volatile uint32_t *)a == 0xFEEDF00Du) { if (!hits) first = a; hits++; }
        res[n++] = hits; res[n++] = first;
        done = 1;
    }
    if (tchan_preamble(&counter, &i)) return;
    tchan_emit_words(res, NRES, i);
}
NEURON_KERNEL(dbg_kernel);
void *_start(void *arg) { (void)arg; return 0; }
