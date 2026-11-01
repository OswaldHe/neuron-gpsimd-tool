// int64_bits: 64-bit arithmetic / bit utilities / mem* through the runtime's base-ISA helpers.
#include <string.h>

#include <gpsimd/gpsimd.hpp>

static uint64_t mulhi(uint64_t a, uint64_t b) {                 // high 64 bits of a 64x64 product
    const uint64_t al = a & 0xFFFFFFFFu, ah = a >> 32, bl = b & 0xFFFFFFFFu, bh = b >> 32;
    const uint64_t ll = al * bl, lh = al * bh, hl = ah * bl, hh = ah * bh;
    const uint64_t mid = (ll >> 32) + (lh & 0xFFFFFFFFu) + (hl & 0xFFFFFFFFu);
    return hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
}

static uint64_t mem_test(uint64_t a, uint64_t b) {
    uint8_t buf[16];
    memcpy(buf, &a, 8);
    memcpy(buf + 8, &b, 8);
    memmove(buf + 3, buf, 8);
    memset(buf + 12, buf[1], 4);
    int cmp = 0;
    for (int k = 0; k < 8 && !cmp; ++k) cmp = (buf[k] > buf[8 + k]) - (buf[k] < buf[8 + k]);   // lexicographic
    const int libc = memcmp(buf, buf + 8, 8);
    if ((libc > 0) != (cmp > 0) || (libc < 0) != (cmp < 0)) return 0xBAD;
    uint64_t x, y;
    memcpy(&x, buf, 8);
    memcpy(&y, buf + 8, 8);
    return (x ^ y) + static_cast<uint64_t>(cmp & 0xF);
}

GPSIMD_OP(int64_bits) {
    gp::Call c(call);
    auto A = c.in<uint32_t>(0), B = c.in<uint32_t>(1);
    auto O = c.out<uint32_t>(0);
    const uint32_t n = A.elems() / 2, NB = O.elems() / (2 * n);
    constexpr uint32_t ROWS = 4;
    uint32_t la[ROWS * 32], lb[ROWS * 32], lo[ROWS * 19 * 32];
    if (2 * n > 32 || NB > 19) { c.fail(8); return; }
    for (gp::Range blk : c.blocks(O.partitions(), ROWS)) {
        c.fetch(A, blk.lo, blk.size(), la);
        c.dma_wait(c.fetch(B, blk.lo, blk.size(), lb));
        for (uint32_t q = 0; q < blk.size(); ++q) {
            const uint32_t *ra = la + q * 2 * n, *rb = lb + q * 2 * n;
            uint32_t *ro = lo + q * O.elems();
            for (uint32_t i = 0; i < n; ++i) {
                const uint64_t a = ra[2 * i] | static_cast<uint64_t>(ra[2 * i + 1]) << 32;
                const uint64_t b = rb[2 * i] | static_cast<uint64_t>(rb[2 * i + 1]) << 32;
                const int64_t sa = static_cast<int64_t>(a), sb = static_cast<int64_t>(b ? b : 1);
                const bool ovf = sa == INT64_MIN && sb == -1;
                const uint32_t lo32 = static_cast<uint32_t>(a);
                const uint64_t r[] = {
                    a + b, a - b, a * b, a / (b ? b : 1), a % (b ? b : 1),
                    static_cast<uint64_t>(ovf ? sa : sa / sb), static_cast<uint64_t>(ovf ? 0 : sa % sb),
                    a << (b & 63), a >> (b & 63), static_cast<uint64_t>(sa >> (b & 63)),
                    static_cast<uint64_t>((sa < static_cast<int64_t>(b)) | (a < b) << 1 | (a == b) << 2),
                    static_cast<uint64_t>(__builtin_popcountll(a) | __builtin_popcount(lo32) << 8),
                    static_cast<uint64_t>((a ? __builtin_clzll(a) : 64) | (lo32 ? __builtin_clz(lo32) : 32) << 8),
                    static_cast<uint64_t>((a ? __builtin_ctzll(a) : 64) | (lo32 ? __builtin_ctz(lo32) : 32) << 8),
                    static_cast<uint64_t>(__builtin_ffsll(static_cast<long long>(a))),
                    __builtin_bswap64(a),
                    static_cast<uint64_t>(__builtin_parityll(a)),
                    mulhi(a, b),
                    mem_test(a, b),
                };
                const uint32_t nops = sizeof(r) / sizeof(r[0]);
                for (uint32_t k = 0; k < nops; ++k) {
                    ro[(k * n + i) * 2] = static_cast<uint32_t>(r[k]);
                    ro[(k * n + i) * 2 + 1] = static_cast<uint32_t>(r[k] >> 32);
                }
            }
        }
        c.dma_wait(c.flush(O, blk.lo, blk.size(), lo));
    }
}
