// ngram_hash: DeepSeek-V4.1 engram n-gram row ids (NgramHashState) from the rows' n-gram token windows.
//
// ins:  toks [128, NT*4] int32   toks[p, i*4 + s] = compressed token id at shift s (0 = the row's own token) of
//                                row t = i*128 + p, the pad id where the window is blocked (as the port's
//                                `_hash_rows` forms it)
//       hcol [4, 48] int32       rows: the 48 primes (layer, n-gram size, head), offsets - row_start,
//                                floor(2^64 / prime) as (hi, lo) words
//       mult [1, 16] int32       multiplier of (layer l, shift s) as (hi, lo) words at 8l + 2s, 8l + 2s + 1
// outs: hv   [128, NT*48] int32  hv[p, i*48 + c] = ((XOR over shifts <= n of tok_s * mult_ls) mod prime_c)
//                                + offset_c  for column c = (l*3 + n-1)*8 + head
//
// The 64-bit products come from MULL / MULUH (exact, no product table); each 63-bit rolled value is reduced
// modulo its prime by Horner's rule over 8-bit chunks with a Barrett step (MULUH by floor((2^32-1)/p),
// two corrections): exact for p < 2^24. Each core hashes its 16 partitions of every tile.
#include <gpsimd/gpsimd.hpp>

namespace {
constexpr uint32_t NT = NG_NT, NS = 4, NC = 48, LAYERS = 2, NGRAMS = 3, HEADS = 8;

inline uint32_t mulhi(uint32_t a, uint32_t b) {
    uint32_t r;
    __asm__("muluh %0, %1, %2" : "=a"(r) : "a"(a), "a"(b));
    return r;
}
struct Prime {
    uint32_t p, mu;                       // mu = floor((2^32 - 1) / p)
};
inline uint32_t bmod(uint32_t x, const Prime &q) {     // x mod p, any 32-bit x
    uint32_t r = x - mulhi(x, q.mu) * q.p;            // in [0, 3p)
    if (r >= q.p) r -= q.p;
    if (r >= q.p) r -= q.p;
    return r;
}
// v = hi * 2^32 + lo (v < 2^63) mod p with mu = floor(2^64 / p) = muh * 2^32 + mul (muh < 2^9): the quotient
// estimate floor(v * mu / 2^64) is at most 3 below the true one, and only its low 32 bits are needed.
inline uint32_t mod63b(uint32_t hi, uint32_t lo, uint32_t p, uint32_t muh, uint32_t mul) {
    const uint64_t mid = static_cast<uint64_t>(lo) * muh + static_cast<uint64_t>(hi) * mul + mulhi(lo, mul);
    const uint32_t q = hi * muh + static_cast<uint32_t>(mid >> 32);
    uint32_t r = lo - q * p;                           // v - q p < 4p < 2^32
    if (r >= p) r -= p;
    if (r >= p) r -= p;
    if (r >= p) r -= p;
    return r;
}
inline uint32_t mod63(uint32_t hi, uint32_t lo, const Prime &q) {
    uint32_t r = bmod(hi, q);                          // < p < 2^24, so r << 8 fits
    r = bmod((r << 8) | (lo >> 24), q);
    r = bmod((r << 8) | ((lo >> 16) & 0xFFu), q);
    r = bmod((r << 8) | ((lo >> 8) & 0xFFu), q);
    return bmod((r << 8) | (lo & 0xFFu), q);
}
}  // namespace

GPSIMD_OP(ngram_hash) {
    gp::Call c(call);
    auto toks_t = c.in<int32_t>(0);
    auto hcol_t = c.in<int32_t>(1);
    auto mult_t = c.in<uint32_t>(2);
    auto hv_t = c.out<int32_t>(0);
    if (toks_t.partitions() != 128 || toks_t.elems() != NT * NS || hcol_t.elems() != NC || hcol_t.partitions() != 4 || mult_t.elems() != 16 ||
        hv_t.partitions() != 128 || hv_t.elems() != NT * NC) {
        c.fail(8);
        return;
    }
    // my partitions: one tile -> the T real rows split over the cores, else 16 partitions of every tile
    const gp::Range rows = c.split(NT == 1 ? NG_T : 128u);
    const uint32_t nr = rows.size();
    if (!nr) return;
    constexpr uint32_t MAXR = NT == 1 ? (NG_T + 7) / 8 : 16;
    alignas(16) int32_t toks[MAXR * NT * NS];
    alignas(16) int32_t hcol[4 * NC];            // primes, offsets, mu hi, mu lo
    alignas(16) uint32_t mult[16];
    const uint32_t t0 = c.fetch(toks_t, rows.lo, nr, toks);
    const uint32_t t1 = c.fetch(hcol_t, 0, 4, hcol);
    const uint32_t t2 = c.fetch(mult_t, 0, 1, mult);
    c.dma_wait(t0);
    c.dma_wait(t2);
    // 1. the rolled 63-bit values of every (row, layer, n-gram): R[k][g], g = l*3 + n-1
    constexpr uint32_t NG = LAYERS * NGRAMS;
    const uint32_t nk = nr * NT;                       // (row, tile) pairs, k = q*NT + i
    alignas(16) uint32_t RH[MAXR * NT * NG], RL[MAXR * NT * NG];
    for (uint32_t k = 0; k < nk; ++k) {
        const int32_t *tk = toks + k * NS;
        for (uint32_t l = 0; l < LAYERS; ++l) {
            uint32_t rh = 0, rl = 0;
            for (uint32_t s = 0; s < NS; ++s) {
                const uint32_t cid = static_cast<uint32_t>(tk[s]);
                const uint32_t mh = mult[8 * l + 2 * s], ml = mult[8 * l + 2 * s + 1];
                rh ^= mulhi(cid, ml) + cid * mh;         // (cid * m) mod 2^64, cid < 2^17
                rl ^= cid * ml;
                if (s) {
                    RH[k * NG + l * NGRAMS + s - 1] = rh;
                    RL[k * NG + l * NGRAMS + s - 1] = rl;
                }
            }
        }
    }
    // 2. column by column (the prime's constants stay in registers): mod + offset
    c.dma_wait(t1);
    alignas(16) int32_t out[MAXR * NT * NC];
    for (uint32_t col = 0; col < NC; ++col) {
        const uint32_t p = static_cast<uint32_t>(hcol[col]), muh = static_cast<uint32_t>(hcol[2 * NC + col]);
        const uint32_t mul = static_cast<uint32_t>(hcol[3 * NC + col]);
        const int32_t off = hcol[NC + col];
        const uint32_t g = col / HEADS;
        for (uint32_t k = 0; k < nk; ++k)
            out[k * NC + col] = static_cast<int32_t>(mod63b(RH[k * NG + g], RL[k * NG + g], p, muh, mul)) + off;
    }
    c.dma_2d(hv_t.row(rows.lo), out, NT * NC * 4, nr, NT * NC * 4, GP_PARTITION_STRIDE);
    c.dma_wait_all();
}
