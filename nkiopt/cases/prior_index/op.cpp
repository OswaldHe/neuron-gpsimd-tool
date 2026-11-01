// prior_index: resolve DeepSeek-V3.2 indexer-selected prior positions to paged-KV cache rows.
//
// ins:  gi [T, NP] int32 (selected positions, one token per partition), bt [T, MAXB] int32 (block table).
// out:  nat [128, T, NPT] int32: rows in natural key-major order (key tile*128 + p of token t at [p, t, tile]).
// row(k) = clamp(bt[t][clamp(pos >> log2(BS), 0, MAXB-1)] * BS + (pos & (BS-1)), 0, NBLK*BS-1), pos = gi[t][k]
// (-1 for the ragged tail) -- nkilib's formula and clamps. Core c handles tokens c, c+8, ...: the token's two
// rows come in with the iDMA (prefetched one token ahead), every key is one table lookup in local RAM, and the
// [128][NPT] result goes out with one 2D iDMA (128 partitions). This replaces nkilib's broadcast of the block
// table to all 128 partitions (2 MB per pair of tokens at MAXB = 2048) and its nc_n_gather resolution.
#include <gpsimd/gpsimd.hpp>

namespace {
constexpr int32_t BS = PI_BS, SHIFT = __builtin_ctz(PI_BS), NBLK = PI_NBLK;
constexpr uint32_t KMAX = 4096, BMAX = 8192;
inline int32_t clamp(int32_t v, int32_t lo, int32_t hi) { return v < lo ? lo : (v > hi ? hi : v); }
}

GPSIMD_OP(prior_index) {
    gp::Call c(call);
    auto gi_t = c.in<int32_t>(0), bt_t = c.in<int32_t>(1);
    auto nat_t = c.out<int32_t>(0);
    const uint32_t T = gi_t.partitions(), NP = gi_t.elems(), MAXB = bt_t.elems();
    const uint32_t NPT = (NP + 127) / 128, NK = NPT * 128;
    if (NK > KMAX || MAXB > BMAX || nat_t.partitions() != 128 || nat_t.elems() != T * NPT || (NPT & 3)) { c.fail(8); return; }
    alignas(16) int32_t g[2][KMAX], b[2][BMAX], nat[KMAX];
    const int32_t maxrow = NBLK * BS - 1, maxb = static_cast<int32_t>(MAXB) - 1;
    uint32_t t = c.core(), k = 0;
    if (t >= T) return;
    c.fetch(gi_t, t, 1, g[0]);
    c.fetch(bt_t, t, 1, b[0]);
    for (; t < T; t += c.ncores(), k ^= 1) {
        c.dma_wait_all();                                        // this token's rows (and the last output)
        if (t + c.ncores() < T) {                                // prefetch the next token
            c.fetch(gi_t, t + c.ncores(), 1, g[k ^ 1]);
            c.fetch(bt_t, t + c.ncores(), 1, b[k ^ 1]);
        }
        int32_t *gk = g[k];
        const int32_t *bk = b[k];
        for (uint32_t i = NP; i < NK; ++i) gk[i] = -1;          // ragged tail sentinel
        int32_t *o = nat;
        for (uint32_t p = 0; p < 128; ++p) {
            const int32_t *src = gk + p;
            for (uint32_t tile = 0; tile < NPT; tile += 4, o += 4) {      // o = nat + p*NPT + tile
                const int32_t p0 = src[(tile + 0) * 128], p1 = src[(tile + 1) * 128];
                const int32_t p2 = src[(tile + 2) * 128], p3 = src[(tile + 3) * 128];
                const int32_t b0 = bk[clamp(p0 >> SHIFT, 0, maxb)], b1 = bk[clamp(p1 >> SHIFT, 0, maxb)];
                const int32_t b2 = bk[clamp(p2 >> SHIFT, 0, maxb)], b3 = bk[clamp(p3 >> SHIFT, 0, maxb)];
                o[0] = clamp(b0 * BS + (p0 & (BS - 1)), 0, maxrow);
                o[1] = clamp(b1 * BS + (p1 & (BS - 1)), 0, maxrow);
                o[2] = clamp(b2 * BS + (p2 & (BS - 1)), 0, maxrow);
                o[3] = clamp(b3 * BS + (p3 & (BS - 1)), 0, maxrow);
            }
        }
        c.dma_2d(nat_t.row(0) + t * NPT, nat, NPT * 4, 128, NPT * 4, GP_PARTITION_STRIDE);
        c.dma_wait_all();                                        // nat is reused by the next token
    }
}
