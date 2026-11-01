/* Probe 10: search for the SBUF input tile (partition 0: 0.0, 0.25, 0.5 at +0x3FE00) by stepping
 * through [0x00100000, 0xFFFC0000) in 256 KB strides. Report hits count and first 4 bases. */
#include "neuron_kl.h"
#include "tchan.h"
#define TILE 0x3FE00u
#define NRES 6
static volatile uint32_t counter;
static volatile uint32_t res[NRES];
static volatile uint32_t done;
void dbg_kernel(void) {
    uint32_t i;
    if (!done) {
        uint32_t n = 0, nhit = 0;
        for (uint32_t b = 0x00100000u; b < 0xFFFC0000u; b += 0x40000u) {
            volatile uint32_t *p = (volatile uint32_t *)(b + TILE);
            if (p[1] == 0x3e800000u && p[2] == 0x3f000000u && p[0] == 0) {
                if (n < 4) res[2 + n++] = b;
                nhit++;
            }
        }
        res[0] = 0xA11DADA0u; res[1] = nhit;
        done = 1;
    }
    if (tchan_preamble(&counter, &i)) return;
    tchan_emit_words(res, NRES, i);
}
NEURON_KERNEL(dbg_kernel);
void *_start(void *arg) { (void)arg; return 0; }
