/* host reference: same tests compiled for x86 */
#include <stdio.h>
#include <math.h>
#include "isa_tests.h"
struct test { const char *name; test_fn fn; };
static const struct test tests[] = { TESTS };
int main(void) {
    uint32_t out[8];
    for (unsigned t = 0; t < sizeof(tests) / sizeof(tests[0]); t++) {
        tests[t].fn(out); printf("%s:", tests[t].name);
        for (int i = 0; i < 8; i++) printf(" %08x", out[i]);
        printf("\n");
    }
    return 0;
}
