/* Probe 6: kernel's return address (a0) and stack pointer (a1) at entry; scan data RAM for the
 * KERNEL_CALL header word (bytes 4..7 = 05 08 13 03) and the 0x85 opcode word. */
#include "neuron_kl.h"
#include "tchan.h"
#define NRES 16
static volatile uint32_t counter;
static volatile uint32_t res[NRES];
static volatile uint32_t done;
static volatile uint32_t ra, sp;
static void collect(void) {
    uint32_t n = 3;
    res[0] = ra; res[1] = sp;
    for (uint32_t a = 0x80000; a < 0xC0000 && n < NRES; a += 4) {
        uint32_t v = *(volatile uint32_t *)a;
        if (v == 0x03130805u || v == 0x00001085u) res[n++] = a | (v == 0x00001085u ? 0x80000000u : 0);
    }
    res[2] = n - 3;
    while (n < NRES) res[n++] = 0;
}
void dbg_kernel(void) {
    uint32_t i;
    if (!done) {
        uint32_t a0v, a1v;
        __asm__ volatile("mov %0, a0\n mov %1, a1" : "=r"(a0v), "=r"(a1v));
        ra = a0v; sp = a1v;
        collect(); done = 1;
    }
    if (tchan_preamble(&counter, &i)) return;
    tchan_emit_words(res, NRES, i);
}
NEURON_KERNEL(dbg_kernel);
void *_start(void *arg) { (void)arg; return 0; }
