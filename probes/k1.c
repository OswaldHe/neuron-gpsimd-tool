/* Experiment 1: execute an illegal instruction to prove the kernel is dispatched. */
#include "neuron_kl.h"
volatile int kl_marker = 0x4b4c3031;
int dbg_kernel(void *a, void *b, void *c, void *d) { (void)a; (void)b; (void)c; (void)d; __asm__ volatile("ill"); return 0; }
NEURON_KERNEL(dbg_kernel);
int _start(void) { return kl_marker; }
