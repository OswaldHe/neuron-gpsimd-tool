/* Experiment 0: a kernel that does nothing and returns 0. */
#include "neuron_kl.h"
volatile int kl_marker = 0x4b4c3030;  /* keeps a RW data segment in the image */
int dbg_kernel(void *a, void *b, void *c, void *d) { (void)a; (void)b; (void)c; (void)d; return 0; }
NEURON_KERNEL(dbg_kernel);
int _start(void) { return kl_marker; }
