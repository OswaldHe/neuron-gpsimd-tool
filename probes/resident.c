/* resident: is the kernel library's data present in GPSIMD data RAM right after nrt_load? */
#include "neuron_kl.h"
#include "kprint.h"
volatile uint32_t params[8] = {0x50415241u, 0x4d53424bu, 1, 2, 3, 4, 5, 6};   /* "ARAP" "KBSM" */
void dbg_kernel(void) { kprintf("core %u params %x %x %u %u\n", kprid(), params[0], params[1], params[2], params[3]); }
NEURON_KERNEL(dbg_kernel);
void *_start(void *arg) { (void)arg; return 0; }
