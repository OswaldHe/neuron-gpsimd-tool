// moe_dispatch: all routing math of an expert-parallel MoE dispatch step in one call.
//
// ins:  ei  [T, K] int32   top-K expert ids per token (router output)
//       aff [T, E] bf16    masked expert affinities (router output)
// outs: tok  [T*K, 1] uint32   token index of every packed row (rows sorted by expert, stable)
//       tail [T*K, 4] bf16     packed row columns H..H+3: affinity[t, e], token index (uint32 as 2 x bf16), 0
//       meta [4, R] float32    all-to-all-v metadata as nkilib build_all2all_dispatch_metadata: send counts
//                              (tokens with >= 1 expert on rank r, times NPT), send displacements, zeros
// = nkilib permute_routed_tokens steps 2-3 + 4.1 + 5 (argsort, transposes, affinity gathers) and
//   build_all2all_dispatch_metadata.
// Every core sorts all N = T*K <= 128 slots (a counting sort, no barrier needed) and writes only its share of
// the output rows (posted SBUF stores, one partition per row) and of the metadata columns (one 2D iDMA: the
// 4 metadata partitions are shared by all cores, and stores of 8 cores into one partition serialize).
// One expert per rank (R = E): a token's experts are distinct, so the per-rank token counts are the sort's
// per-expert counts; otherwise a per-token rank bitset deduplicates.
#include <gpsimd/gpsimd.hpp>

namespace {
constexpr uint32_t T = MOE_T, K = MOE_K, N = MOE_T * MOE_K, E = MOE_E, R = MOE_R, NLOCAL = MOE_E / MOE_R;
constexpr uint32_t NPT = MOE_NPT, WORDS = (R + 31) / 32;
static_assert(N <= 128 && N % 4 == 0 && (K & (K - 1)) == 0, "T*K <= 128, K a power of two");

#ifdef MOE_TIMING
inline uint32_t ccount() { uint32_t v; __asm__ volatile("rsr.ccount %0" : "=a"(v)); return v; }
#define MARK() (TM[nt++] = ccount())
#else
#define MARK() ((void)0)
#endif

inline uint32_t u24_to_f32(uint32_t v) {      // exact float32 bits of an integer v < 2^24
    if (!v) return 0;
    const uint32_t e = 31u - static_cast<uint32_t>(__builtin_clz(v));
    return ((e + 127u) << 23) | ((v << (23u - e)) & 0x7FFFFFu);
}
inline uint32_t clampe(int32_t x) { return static_cast<uint32_t>(x) < E ? static_cast<uint32_t>(x) : E; }
}  // namespace

GPSIMD_OP(moe_dispatch) {
    gp::Call c(call);
#ifdef MOE_TIMING
    uint32_t TM[16], nt = 0;
#endif
    MARK();
    auto ei_t = c.in<int32_t>(0);
    auto aff_t = c.in<uint16_t>(1);
    auto tok_t = c.out<uint32_t>(0);
    auto tail_t = c.out<uint16_t>(1);
    auto meta_t = c.out<uint32_t>(2);
    if (ei_t.partitions() != T || ei_t.elems() != K || aff_t.partitions() != T || aff_t.elems() != E ||
        tok_t.partitions() != N || tail_t.elems() != 4 || meta_t.partitions() != 4 || meta_t.elems() != R) {
        c.fail(8);
        return;
    }
    const gp::Range rows = c.split(N);            // my output rows
    const gp::Range cols = c.split(R);            // my metadata columns
    alignas(16) int32_t ei[N];
    alignas(16) uint16_t aff[T * E];
    const uint32_t t_ei = c.fetch(ei_t, 0, T, ei);
    const uint32_t t_aff = c.fetch(aff_t, 0, T, aff);     // arrives while the experts are counted
    c.dma_wait(t_ei);
    MARK();

    // stable counting sort of the slots s = t*K + k by expert (bin E collects invalid ids)
    uint32_t cnt[E + 1];
    for (uint32_t b = 0; b <= E; ++b) cnt[b] = 0;
    uint32_t bad = 0;
    for (uint32_t s = 0; s < N; s += 4) {
        const uint32_t a = clampe(ei[s]), b = clampe(ei[s + 1]), d = clampe(ei[s + 2]), f = clampe(ei[s + 3]);
        bad |= (a == E) | (b == E) | (d == E) | (f == E);
        ++cnt[a];
        ++cnt[b];
        ++cnt[d];
        ++cnt[f];
    }
    if (bad) c.fail(8);
    uint32_t run = 0, first[E + 1];                                 // first[e]: first row of expert e
    for (uint32_t b = 0; b <= E; ++b) { const uint32_t v = cnt[b]; cnt[b] = first[b] = run; run += v; }
    c.dma_wait(t_aff);
    MARK();
    // my rows go out as posted SBUF stores (one partition each); only experts [elo, ehi] have rows in
    // [rows.lo, rows.hi) (cnt[e] = first row of expert e), so the other slots are skipped
    uint32_t elo = 0, ehi = E;
    while (elo < E && first[elo + 1] <= rows.lo) ++elo;
    for (ehi = elo; ehi < E && first[ehi + 1] < rows.hi; ++ehi) {}
    alignas(16) uint32_t tok[128], tail[128][2];
    for (uint32_t s = 0; s < N; ++s) {
        const uint32_t x = clampe(ei[s]);
        if (x - elo > ehi - elo) continue;
        const uint32_t q = cnt[x]++ - rows.lo;
        if (q >= rows.size()) continue;
        const uint32_t t = s / K;
        tok[q] = t;
        tail[q][0] = (x < E ? aff[t * E + x] : 0u) | (t << 16);     // affinity bits, token index low half
        tail[q][1] = t >> 16;                                       // token index high half, 0
    }
    if (rows.size()) {
        c.dma_2d(tok_t.row(rows.lo), tok, 4, rows.size(), 4, GP_PARTITION_STRIDE);
        c.dma_2d(tail_t.row(rows.lo), tail, 8, rows.size(), 8, GP_PARTITION_STRIDE);
    }
    MARK();

    // all-to-all-v metadata, my columns: built locally, one 2D iDMA into the 4 partitions
    alignas(16) uint32_t m[4][R];
    const uint32_t nc = cols.size();
    if (nc && NLOCAL == 1) {
        for (uint32_t r = cols.lo; r < cols.hi; ++r) {
            const uint32_t q = r - cols.lo;
            m[0][q] = u24_to_f32((first[r + 1] - first[r]) * NPT);
            m[1][q] = u24_to_f32(first[r] * NPT);
            m[2][q] = 0;
            m[3][q] = 0;
        }
    } else if (nc) {
        uint32_t rc[R];
        for (uint32_t r = 0; r < R; ++r) rc[r] = 0;
        for (uint32_t t = 0; t < T; ++t) {
            uint32_t bits[WORDS];
            for (uint32_t w = 0; w < WORDS; ++w) bits[w] = 0;
            for (uint32_t k = 0; k < K; ++k) {
                const uint32_t r = static_cast<uint32_t>(ei[t * K + k]) / NLOCAL;
                if (r < R) bits[r >> 5] |= 1u << (r & 31);
            }
            for (uint32_t w = 0; w < WORDS; ++w)
                for (uint32_t b = bits[w]; b; b &= b - 1) ++rc[(w << 5) + static_cast<uint32_t>(__builtin_ctz(b))];
        }
        run = 0;
        for (uint32_t r = 0; r < cols.lo; ++r) run += rc[r];
        run *= NPT;
        for (uint32_t r = cols.lo; r < cols.hi; ++r) {
            const uint32_t v = rc[r] * NPT, q = r - cols.lo;
            m[0][q] = u24_to_f32(v);
            m[1][q] = u24_to_f32(run);
            m[2][q] = 0;
            m[3][q] = 0;
            run += v;
        }
    }
    if (nc) c.dma_2d(meta_t.row(0) + cols.lo, m, nc * 4, 4, R * 4, GP_PARTITION_STRIDE);
    MARK();
    __asm__ volatile("memw" ::: "memory");                          // posted stores visible before return
    c.dma_wait_all();
    MARK();
#ifdef MOE_TIMING
    if (c.exec() == 2) {
        gp_printf("core %u:", c.core());
        for (uint32_t k = 1; k < nt; ++k) gp_printf(" %u", TM[k] - TM[k - 1]);
        gp_printf("\n");
    }
#endif
}
