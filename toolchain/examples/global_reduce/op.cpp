// global_reduce: per-core partial reductions over staged rows + cross-core all-reduce (SBUF scratch).
#include <gpsimd/gpsimd.hpp>

using gp::f32;

GPSIMD_OP(global_reduce) {
    gp::Call c(call);
    auto X = c.in<int32_t>(0);
    auto XF = c.in<f32>(1);
    auto OI = c.out<int32_t>(0);
    auto OF = c.out<f32>(1);
    const uint32_t n = X.elems();
    constexpr uint32_t CAP = 2048;
    int32_t lx[CAP];
    uint32_t lf[CAP];
    if (n > CAP) { c.fail(8); return; }
    uint32_t sum = 0, neg = 0;
    int32_t mx = INT32_MIN, mn = INT32_MAX, arg = INT32_MAX;
    f32 fsum, fmax = f32::from_bits(0xFF800000u);
    for (gp::Range blk : c.blocks(X.partitions(), CAP / n)) {
        c.fetch(X, blk.lo, blk.size(), lx);
        c.dma_wait(c.fetch(XF, blk.lo, blk.size(), lf));
        for (uint32_t k = 0; k < blk.size() * n; ++k) {
            const int32_t v = lx[k];
            sum += static_cast<uint32_t>(v);
            neg += v < 0;
            if (v > mx) { mx = v; arg = static_cast<int32_t>(blk.lo * n + k); }
            mn = v < mn ? v : mn;
            const f32 f = f32::from_bits(lf[k]);
            fsum = fsum + f;
            fmax = gp::max(fmax, f);
        }
    }
    const uint32_t gsum = c.allreduce_add(sum), gneg = c.allreduce_add(neg);
    const int32_t gmax = c.allreduce_max(mx), gmin = c.allreduce_min(mn);
    const int32_t garg = c.allreduce_min(mx == gmax ? arg : INT32_MAX);    // first index holding the max
    const f32 gfsum = c.allreduce_add(fsum), gfmax = c.allreduce_max(fmax);
    if (c.core() == 0) {
        const int32_t r[8] = {static_cast<int32_t>(gsum), gmax, gmin, garg, static_cast<int32_t>(gneg), 0, 0, 0};
        for (uint32_t k = 0; k < 8; ++k) OI.store(0, k, r[k]);
        OF.store(0, 0, gfsum);
        OF.store(0, 1, gfmax);
    }
}
