// hbm_gather: random reads from an HBM table and direct writes to an HBM output tensor.
#include <gpsimd/gpsimd.hpp>

GPSIMD_OP(hbm_gather) {
    gp::Call c(call);
    auto idx = c.in<int32_t>(0);
    auto params = c.in<int32_t>(1);
    auto out = c.out<uint32_t>(0);
    if (!gp::hbm_mapped(params)) { c.fail(16); return; }
    volatile uint32_t *table = gp::hbm_ptr<uint32_t>(params, 0);   // float32 bits
    volatile int32_t *direct = gp::hbm_ptr<int32_t>(params, 1);     // [128, 64] int32, row-major
    const uint32_t n = idx.elems();
    for (uint32_t p : c.partitions(idx)) {
        for (uint32_t i = 0; i < n; ++i) {
            const int32_t k = idx.load(p, i);
            out.store(p, i, table[k]);
            direct[p * n + i] = 2 * k + 1;
        }
    }
    gp_memw();
}
