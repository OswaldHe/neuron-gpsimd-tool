/* isa_probe: run a group of operation tests on GPSIMD core 0 and kprintf every result. */
#include "neuron_kl.h"
#include "kprint.h"
#include "isa_tests.h"
struct test { const char *name; test_fn fn; };
static const struct test tests[] = { TESTS };
void dbg_kernel(void) {
    if (kprid() != 0) return;
    uint32_t out[8];
    for (unsigned t = 0; t < sizeof(tests) / sizeof(tests[0]); t++) {
        kprintf("%s:", tests[t].name);          /* logged before running: a hang points here */
        tests[t].fn(out);
        for (int i = 0; i < 8; i++) kprintf(" %08x", out[i]);
        kprintf("\n");
    }
    kprintf("done\n");
}
NEURON_KERNEL(dbg_kernel);
void *_start(void *arg) { (void)arg; return 0; }
