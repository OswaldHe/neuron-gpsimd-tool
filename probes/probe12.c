/* Probe 12: do ncore2gp Vision-P TIE (IVP) instructions execute correctly on the trn3 core?
 * sum/max of 0..15 through IVP_LVN_2XF32_I + IVP_RADDN_2XF32 / IVP_RMAXN_2XF32. */
#include "neuron_kl.h"
#include "tchan.h"
#include <xtensa/tie/xt_ivpn.h>
#define NRES 4
static volatile uint32_t counter;
static volatile uint32_t res[NRES];
static volatile uint32_t done;
static float buf[16] __attribute__((aligned(64)));
void dbg_kernel(void) {
    uint32_t i;
    if (!done) {
        for (int k = 0; k < 16; k++) buf[k] = (float)k;
        xb_vecN_2xf32 v = IVP_LVN_2XF32_I((const xb_vecN_2xf32 *)buf, 0);
        float s = IVP_RADDN_2XF32(v), m = IVP_RMAXN_2XF32(v);
        union { float f; uint32_t u; } a = {s}, b = {m};
        res[0] = a.u; res[1] = b.u; res[2] = 0x1E5u; res[3] = 0;
        done = 1;
    }
    if (tchan_preamble(&counter, &i)) return;
    tchan_emit_words(res, NRES, i);
}
NEURON_KERNEL(dbg_kernel);
void *_start(void *arg) { (void)arg; return 0; }
