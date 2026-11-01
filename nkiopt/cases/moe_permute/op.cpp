// moe_route_idx: MoE dispatch order from routing decisions. expert_index [1, T*K] int32 (slot s = t*K + k),
// stable counting sort of the slots by expert; for sorted position p emits
//     tok[p] = s / K                 (token whose hidden row goes to row p)
//     aff[p] = (s / K) * E + e_s     (index of its affinity in the flattened [T, E] matrix)
// as [T*K, 1] uint32 tiles (one value per partition). Small problem: one core, local memory; the results
// leave with two 2D iDMA transfers (T*K rows of 4 bytes, one per partition).
#include <gpsimd/gpsimd.hpp>

GPSIMD_OP(moe_route_idx) {
    gp::Call c(call);
    if (c.core() != 0) return;
    auto ids = c.in<int32_t>(0);
    auto tok = c.out<uint32_t>(0), aff = c.out<uint32_t>(1);
    const uint32_t N = ids.elems();
    constexpr uint32_t K = MOE_K, E = MOE_E, NMAX = 4096;
    if (N > NMAX || N % K || tok.partitions() != N || aff.partitions() != N || tok.elems() != 1) { c.fail(8); return; }
    int32_t e[NMAX];
    uint32_t cnt[E + 1], tk[NMAX], af[NMAX];
    c.dma_wait(c.dma(e, ids.row(0), N * 4));
    for (uint32_t b = 0; b <= E; ++b) cnt[b] = 0;
    for (uint32_t s = 0; s < N; ++s) {
        uint32_t x = static_cast<uint32_t>(e[s]);
        if (x >= E) { c.fail(8); x = E; }               // out-of-range expert ids sort last
        ++cnt[x];
    }
    uint32_t run = 0;
    for (uint32_t b = 0; b <= E; ++b) { const uint32_t v = cnt[b]; cnt[b] = run; run += v; }
    for (uint32_t s = 0; s < N; ++s) {
        uint32_t x = static_cast<uint32_t>(e[s]);
        if (x > E) x = E;
        const uint32_t p = cnt[x]++, t = s / K;
        tk[p] = t;
        af[p] = t * E + x;
    }
    c.dma_2d(tok.row(0), tk, 4, N, 4, GP_PARTITION_STRIDE);
    c.dma_wait(c.dma_2d(aff.row(0), af, 4, N, 4, GP_PARTITION_STRIDE));
}
