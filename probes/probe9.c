/* Probe 9: is SBUF visible at core address 0x80000000 + tpb_addr (the trn1 fixed window)?
 * The NKI input tile is at tpb_addr p*0x40000 + 0x3FE00: float32 (p*64 + i) * 0.25. */
#include "neuron_kl.h"
#include "tchan.h"
#define SBUF_LOCAL 0x80000000u
#define TILE 0x3FE00u
#define NRES 8
static volatile uint32_t counter;
static volatile uint32_t res[NRES];
static volatile uint32_t done;
void dbg_kernel(void) {
    uint32_t i;
    if (!done) {
        volatile uint32_t *p0 = (volatile uint32_t *)(SBUF_LOCAL + TILE);
        volatile uint32_t *p1 = (volatile uint32_t *)(SBUF_LOCAL + 0x40000u + TILE);
        res[0] = p0[0]; res[1] = p0[1]; res[2] = p0[2]; res[3] = p0[63];
        res[4] = p1[0]; res[5] = p1[1];
        uint32_t prid;
        __asm__ volatile("rsr.prid %0" : "=r"(prid));
        res[6] = prid; res[7] = 0x600DF00Du;
        done = 1;
    }
    if (tchan_preamble(&counter, &i)) return;
    tchan_emit_words(res, NRES, i);
}
NEURON_KERNEL(dbg_kernel);
void *_start(void *arg) { (void)arg; return 0; }
