// dispatch_meta: all-to-all-v dispatch metadata from MoE routing. expert_index [T, K] int32 (one token per
// partition) -> [4, R] float32: send_counts[r] = #tokens routed to >= 1 expert on rank r (expert / NLOCAL),
// times NPT; send_displs = exclusive prefix sum; rows 2, 3 zero. One core; per-token rank bitset.
#include <gpsimd/gpsimd.hpp>

// exact float32 bits of an integer v < 2^24 (no soft-float call)
static inline uint32_t u24_to_f32(uint32_t v) {
    if (!v) return 0;
    const uint32_t e = 31u - static_cast<uint32_t>(__builtin_clz(v));
    return ((e + 127u) << 23) | ((v << (23u - e)) & 0x7FFFFFu);
}

GPSIMD_OP(dispatch_meta) {
    gp::Call c(call);
    if (c.core() != 0) return;
    auto ei = c.in<int32_t>(0);
    auto out = c.out<uint32_t>(0);                         // float32 bits
    constexpr uint32_t R = META_R, NLOCAL = META_NLOCAL, NPT = META_NPT, WORDS = (R + 31) / 32;
    const uint32_t T = ei.partitions(), K = ei.elems();
    if (T > 128 || K > 32 || out.partitions() != 4 || out.elems() != R) { c.fail(8); return; }
    int32_t e[128 * 32];
    c.dma_wait(c.fetch(ei, 0, T, e));
    uint32_t cnt[R];
    for (uint32_t r = 0; r < R; ++r) cnt[r] = 0;
    for (uint32_t t = 0; t < T; ++t) {
        uint32_t bits[WORDS];
        for (uint32_t w = 0; w < WORDS; ++w) bits[w] = 0;
        for (uint32_t k = 0; k < K; ++k) {
            uint32_t r = static_cast<uint32_t>(e[t * K + k]) / NLOCAL;
            if (r >= R) { c.fail(8); continue; }
            bits[r >> 5] |= 1u << (r & 31);
        }
        for (uint32_t w = 0; w < WORDS; ++w)
            for (uint32_t b = bits[w]; b; b &= b - 1) ++cnt[(w << 5) + static_cast<uint32_t>(__builtin_ctz(b))];
    }
    uint32_t rows[4][R];
    uint32_t run = 0;
    for (uint32_t r = 0; r < R; ++r) {
        const uint32_t v = cnt[r] * NPT;                   // T * NPT < 2^24: exact in float32
        rows[0][r] = u24_to_f32(v);
        rows[1][r] = u24_to_f32(run);
        rows[2][r] = 0;
        rows[3][r] = 0;
        run += v;
    }
    c.dma_wait(c.flush(out, 0, 4, &rows[0][0]));
}
