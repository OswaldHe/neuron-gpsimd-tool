/* sbuf_add: read an int32 tile from SBUF, add a constant on all 8 GPSIMD cores in parallel (core k
 * handles partitions 16k..16k+15), write the result back to SBUF in place.
 * Tile: SBUF offset 0x3FE00, 128 partitions x 64 int32 (the NKI test kernel's SBUF tile). */
#include "neuron_kl.h"
#include "kprint.h"
#include "gpsimd.h"
#define TILE_OFF 0x3FE00u
#define NELEM 64
#define ADDEND 1000

void sbuf_add(void) {
    uint32_t core = gpsimd_core_id();
    uint32_t p0 = core * SBUF_PARTITIONS_PER_CORE, p1 = p0 + SBUF_PARTITIONS_PER_CORE;
    kprintf("core %u: partitions %u..%u at SBUF window 0x%08x\n", core, p0, p1 - 1,
            (uint32_t)sbuf_ptr(p0, TILE_OFF));
    int32_t first_before = sbuf_i32(p0, TILE_OFF)[0];
    int32_t last_before = sbuf_i32(p1 - 1, TILE_OFF)[NELEM - 1];
    for (uint32_t p = p0; p < p1; p++) {
        volatile int32_t *x = sbuf_i32(p, TILE_OFF);
        for (uint32_t i = 0; i < NELEM; i++) x[i] = x[i] + ADDEND;
    }
    kprintf("core %u: x[%u][0] %d -> %d, x[%u][%u] %d -> %d\n", core, p0, first_before,
            sbuf_i32(p0, TILE_OFF)[0], p1 - 1, NELEM - 1, last_before, sbuf_i32(p1 - 1, TILE_OFF)[NELEM - 1]);
}
NEURON_KERNEL(sbuf_add);
void *_start(void *arg) { (void)arg; return 0; }
