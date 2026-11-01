// argsort_f32: stable ascending argsort of a [1, N] float32 row (numpy argsort(kind="stable") order),
// output [1, N] uint32 indices. Parallel LSD radix sort over the 8 GPSIMD cores:
//
//   keys      float32 bits -> order-preserving uint32 (-0.0 folded into +0.0, as numpy compares them)
//   digits    8-bit digits; digits on which all N keys agree are skipped (expert ids: 1-2 passes)
//   per pass  every core histograms its chunk of the current permutation, publishes the histogram to
//             SBUF, reads all 8, computes where each of its elements goes (stable: cores in order),
//             and writes the indices with posted stores into the next permutation buffer in SBUF
//             (the output tile on the last pass, the tmp tile otherwise); a barrier ends the pass.
//   layout    permutation buffers are [128, W] tiles (W = N/128), position j at [j / W, j % W]: random
//             positions spread over 128 partitions (stores of 8 cores into one partition serialize,
//             ~80 instead of ~19 cycles each), and a core's chunk is 16 whole partitions (one 2D iDMA).
//   local     every core keeps a full copy of the raw keys (iDMA), so only indices move between cores.
// Small N (<= 512): core 0 alone runs the same LSD radix sort in local RAM (fixed multi-core costs dominate).
// -DARGSORT_TIMING: core 0 logs the cycles of each phase (second execution).
#include <gpsimd/gpsimd.hpp>

namespace {
constexpr uint32_t NMAX = 16384, SMALL_N = 512;      // N <= SMALL_N: single-core path

inline uint32_t sort_key(uint32_t b) {
    const uint32_t z = (b << 1) == 0 ? 0u : b;              // -0.0 == +0.0
    return z ^ (static_cast<uint32_t>(static_cast<int32_t>(z) >> 31) | 0x80000000u);
}

// iDMA copy in pieces of at most 16 KB (returns the last ticket; transfers complete in order)
uint32_t copy(const gp::Call &c, volatile void *dst, const volatile void *src, uint32_t bytes) {
    uint32_t t = 0;
    auto *d = reinterpret_cast<volatile uint8_t *>(dst);
    auto *s = reinterpret_cast<const volatile uint8_t *>(src);
    for (uint32_t o = 0; o < bytes; o += 16384) {
        const uint32_t m = bytes - o < 16384 ? bytes - o : 16384;
        t = c.dma(d + o, s + o, m);
    }
    return t;
}

#ifdef ARGSORT_TIMING
inline uint32_t ccount() { uint32_t v; __asm__ volatile("rsr.ccount %0" : "=a"(v)); return v; }
#define MARK() (T[nt++] = ccount())
#else
#define MARK() ((void)0)
#endif
}  // namespace

GPSIMD_OP(argsort_f32) {
    gp::Call c(call);
#ifdef ARGSORT_TIMING
    uint32_t T[40], nt = 0;
#endif
    MARK();
    auto data = c.in<uint32_t>(0);     // float32 bits
    auto tmp = c.in<uint32_t>(1);      // [128, W] scratch permutation
    auto hist = c.in<uint32_t>(2);     // [8, 128] scratch histograms (row = core)
    auto out = c.out<uint32_t>(0);     // [128, W]
    const uint32_t N = data.elems(), NC = c.ncores(), me = c.core();
    const uint32_t W = N / 128;
    if (NC != 8 || N % 128 || N > NMAX || out.partitions() != 128 || out.elems() != W || tmp.partitions() != 128 ||
        tmp.elems() != W || hist.partitions() < NC || hist.elems() < 128 || (W & (W - 1))) {
        c.fail(8);
        return;
    }
    const uint32_t n = N / NC, lo = me * n, lw = 31u - static_cast<uint32_t>(__builtin_clz(W)), wm = W - 1;

    if (N <= SMALL_N) {   // small: one core sorts in local RAM (no barriers / histogram exchange)
        if (me != 0) return;
        uint32_t key[SMALL_N], ia[SMALL_N], ib[SMALL_N], cnt[256];
        c.dma_wait(c.dma(key, data.row(0), N * 4));
        uint32_t diff = 0;
        for (uint32_t i = 0; i < N; ++i) { key[i] = sort_key(key[i]); diff |= key[i] ^ key[0]; ia[i] = i; }
        uint32_t *src = ia, *dst = ib;
        for (uint32_t d = 0; d < 4; ++d) {
            const uint32_t s8 = 8 * d;
            if (!((diff >> s8) & 0xFFu)) continue;
            for (uint32_t b = 0; b < 256; ++b) cnt[b] = 0;
            for (uint32_t i = 0; i < N; ++i) ++cnt[(key[src[i]] >> s8) & 0xFFu];
            uint32_t run = 0;
            for (uint32_t b = 0; b < 256; ++b) { const uint32_t v = cnt[b]; cnt[b] = run; run += v; }
            for (uint32_t i = 0; i < N; ++i) { const uint32_t j = src[i]; dst[cnt[(key[j] >> s8) & 0xFFu]++] = j; }
            uint32_t *t = src; src = dst; dst = t;
        }
        c.dma_wait(c.flush(out, 0, 128, src));                 // position j -> [j / W, j % W]
        return;
    }

    uint32_t raw[NMAX];
    uint32_t cur[NMAX / 8], dpos[NMAX / 8];
    uint32_t H[8][128];                // [core][w]: 16-bit counters of bins 2w (low half) and 2w+1 (high half)
    uint32_t base[256];
    c.dma_wait(copy(c, raw, data.row(0), N * 4));
    MARK();

    // digits on which the keys differ: OR of key ^ key(0) over my chunk, combined through SBUF
    const uint32_t k0 = sort_key(raw[0]);
    uint32_t diff = 0;
    for (uint32_t i = lo; i < lo + n; ++i) diff |= sort_key(raw[i]) ^ k0;
    volatile uint32_t *sh = c.shared();
    sh[me] = diff;
    c.barrier();
    uint32_t all[8];
    c.dma_wait(c.dma(all, sh, NC * 4));
    diff = 0;
    for (uint32_t k = 0; k < NC; ++k) diff |= all[k];
    uint32_t digits[4], P = 0;
    for (uint32_t d = 0; d < 4; ++d)
        if ((diff >> (8 * d)) & 0xFFu) digits[P++] = d;
    for (uint32_t i = 0; i < n; ++i) cur[i] = lo + i;
    MARK();
    if (P == 0) {                                          // all keys equal: identity
        c.dma_wait(c.flush(out, lo >> lw, n >> lw, cur));
        return;
    }

    uint32_t m[8];                                         // m[k] = all ones for cores before me
    for (uint32_t k = 0; k < 8; ++k) m[k] = k < me ? 0xFFFFFFFFu : 0u;
    for (uint32_t q = 0; q < P; ++q) {
        const uint32_t sh8 = 8 * digits[q];
        const gp::Tile<uint32_t> &dt = ((P - 1 - q) & 1) ? tmp : out;   // the last pass writes the output
        volatile uint32_t *const d0 = dt.row(0);
        uint32_t *h = H[me];
        for (uint32_t w = 0; w < 128; ++w) h[w] = 0;
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t b = (sort_key(raw[cur[i]]) >> sh8) & 0xFFu;
            dpos[i] = b;
            h[b >> 1] += 1u << ((b & 1u) << 4);
        }
        c.dma_wait(c.dma(hist.row(me), h, 512));
        MARK();
        c.barrier();
        MARK();
        c.dma_wait(c.dma_2d(H, hist.row(0), 512, NC, GP_PARTITION_STRIDE, 512));
        MARK();
        // base[b] = all cores' counts of bins < b + earlier cores' counts of bin b (stable order)
        uint32_t run = 0;
        for (uint32_t w = 0; w < 128; ++w) {
            const uint32_t a0 = H[0][w], a1 = H[1][w], a2 = H[2][w], a3 = H[3][w];
            const uint32_t a4 = H[4][w], a5 = H[5][w], a6 = H[6][w], a7 = H[7][w];
            const uint32_t tot = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
            const uint32_t bef = (a0 & m[0]) + (a1 & m[1]) + (a2 & m[2]) + (a3 & m[3]) + (a4 & m[4]) + (a5 & m[5]) +
                                 (a6 & m[6]) + (a7 & m[7]);
            base[2 * w] = run + (bef & 0xFFFFu);
            run += tot & 0xFFFFu;
            base[2 * w + 1] = run + (bef >> 16);
            run += tot >> 16;
        }
        MARK();
        for (uint32_t i = 0; i < n; ++i) dpos[i] = base[dpos[i]]++;   // bin -> destination (stable)
        MARK();
        for (uint32_t i = 0; i < n; ++i) {                          // posted SBUF stores
            const uint32_t j = dpos[i];
            d0[(j >> lw) * (GP_PARTITION_STRIDE / 4) + (j & wm)] = cur[i];
        }
        MARK();
        c.barrier();                                       // memw: all cores' stores visible
        MARK();
        if (q + 1 < P) c.dma_wait(c.fetch(dt, lo >> lw, n >> lw, cur));
        MARK();
    }
#ifdef ARGSORT_TIMING
    if (me == 0 && c.exec() == 2) {
        gp_printf("N=%u P=%u:", N, P);
        for (uint32_t k = 1; k < nt; ++k) gp_printf(" %u", T[k] - T[k - 1]);
        gp_printf("\n");
    }
#endif
}
