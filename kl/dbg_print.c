/* dbg_print: a trn3 GPSIMD kernel-library kernel that "prints" a debug message.
 * The message is formatted on the device (no libc) and transmitted over the timing side
 * channel (tchan.h): only core 0 transmits, the other 7 cores return immediately. */
#include "neuron_kl.h"
#include "tchan.h"
#define MSG_MAX 96
static volatile uint32_t counter;
static volatile uint32_t ctx;
static char msg[MSG_MAX];
static volatile uint32_t ready;

static uint32_t put_str(uint32_t n, const char *s) { while (*s && n < MSG_MAX - 1) msg[n++] = *s++; return n; }
static uint32_t put_hex(uint32_t n, uint32_t v) {
    n = put_str(n, "0x");
    for (int k = 7; k >= 0 && n < MSG_MAX - 1; k--) msg[n++] = "0123456789abcdef"[(v >> (4 * k)) & 15];
    return n;
}

void dbg_print(uint32_t a2, uint32_t a3, uint32_t a4, uint32_t a5, uint32_t a6, uint32_t a7) {
    uint32_t prid, i;
    __asm__ volatile("rsr.prid %0" : "=r"(prid));
    if (!ready) {
        uint32_t n = 0;
        n = put_str(n, "dbg_print cpu"); msg[n++] = (char)('0' + (prid & 7));
        n = put_str(n, " a2="); n = put_hex(n, a2);
        n = put_str(n, " a4="); n = put_hex(n, a4);
        n = put_str(n, " a5="); n = put_hex(n, a5);
        n = put_str(n, " a6="); n = put_hex(n, a6);
        n = put_str(n, " ctx="); n = put_hex(n, ctx);
        (void)a3; (void)a7;
        msg[n] = 0;
        ready = 1;
    }
    if (tchan_preamble(&counter, &i)) return;
    if ((prid & 0xff) != 0) return;                 /* only core 0 transmits */
    uint32_t b = i >> 1;
    uint8_t c = b < MSG_MAX ? (uint8_t)msg[b] : 0;
    tchan_emit((i & 1) ? (c & 15) : (c >> 4));
}
NEURON_KERNEL(dbg_print);
void *_start(void *arg) { ctx = (uint32_t)arg; return 0; }
