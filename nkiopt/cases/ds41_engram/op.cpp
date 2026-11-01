// engram_hits: the n-gram table lookups that hit this rank's shard, compacted.
//
// ins:  ids  [128, NT*NCOL] int32   ids[p, i*NCOL + j] = shard-local row id of token t = i*128 + p, hash column j
//                                  (global id - row_start; this rank's rows are 0 <= id < ROWW)
// outs: hits [2, CAP] int32        row 0: table row of every hit, row 1: its row in emb viewed as
//                                  [Tp*NCOL, hd] (t*NCOL + j); hits in (core, partition, tile, column) order,
//                                  padded to a multiple of 128 with (0, OOB) (OOB = Tp*NCOL: skipped by the scatter)
//       cnt  [1, 1] int32          number of 128-hit chunks (the trip count of the gather loop)
//
// Every core scans its 16 partitions (all tiles, all columns): one iDMA fetch, a compare per id, and the
// hit's two indices appended to local lists. An exclusive scan over the cores places each core's lists;
// each core writes them with one iDMA per list; core 0 writes the padding and the count.
#include <gpsimd/gpsimd.hpp>

namespace {
constexpr uint32_t NT = EG_NT, NCOL = EG_NCOL, ROWW = EG_ROWW, CAP = EG_CAP, W = EG_NT * EG_NCOL;
constexpr uint32_t OOB = EG_NT * 128u * EG_NCOL, PCAP = 2048;     // per-core list capacity
static_assert(CAP % 128 == 0, "CAP is a multiple of 128");
}  // namespace

GPSIMD_OP(engram_hits) {
    gp::Call c(call);
    auto ids_t = c.in<int32_t>(0);
    auto hits_t = c.out<int32_t>(0);
    auto cnt_t = c.out<int32_t>(1);
    if (ids_t.partitions() != 128 || ids_t.elems() != W || hits_t.partitions() != 2 || hits_t.elems() != CAP) {
        c.fail(8);
        return;
    }
    const gp::Range rows = c.split(128);                       // 16 partitions per core
    alignas(16) int32_t ids[16 * W];
    c.dma_wait(c.fetch(ids_t, rows.lo, rows.size(), ids));
    alignas(16) uint32_t src[PCAP], dst[PCAP];
    uint32_t n = 0;
    for (uint32_t q = 0; q < rows.size(); ++q) {
        const int32_t *r = ids + q * W;
        const uint32_t p = rows.lo + q;
        for (uint32_t i = 0; i < NT; ++i) {
            const uint32_t base = (i * 128u + p) * NCOL;
            for (uint32_t j = 0; j < NCOL; ++j) {
                const uint32_t v = static_cast<uint32_t>(r[i * NCOL + j]);
                if (v < ROWW) {                                   // negative ids wrap above ROWW
                    if (n < PCAP) {
                        src[n] = v;
                        dst[n] = base + j;
                    }
                    ++n;
                }
            }
        }
    }
    uint32_t total;
    const uint32_t off = c.exscan_add(n, &total);
    const uint32_t padded = (total + 127u) & ~127u;
    if (n > PCAP || padded > CAP) {
        c.fail(16);                                               // more hits than the lists hold
        return;
    }
    if (n) {
        c.dma(hits_t.row(0) + off, src, n * 4);
        c.dma(hits_t.row(1) + off, dst, n * 4);
    }
    if (c.core() == 0) {
        alignas(16) uint32_t zpad[128], opad[128];
        const uint32_t np = padded - total;
        for (uint32_t k = 0; k < np; ++k) {
            zpad[k] = 0;
            opad[k] = OOB;
        }
        if (np) {
            c.dma(hits_t.row(0) + total, zpad, np * 4);
            c.dma(hits_t.row(1) + total, opad, np * 4);
        }
        cnt_t.row(0)[0] = static_cast<int32_t>(padded / 128u);
    }
    __asm__ volatile("memw" ::: "memory");
    c.dma_wait_all();
}

// engram_hits_t: the same hits in the layout NKI's indirect DMAs take, for a STATIC number of DMAs:
// hit k at partition k % 128, column k / 128 of `src` (table row) and `dst` (emb row); every other entry
// is padding (src 0, dst OOB: skipped by the scatter). A rank's shard (ROWW rows) is smaller than every
// hash column's prime range, so it overlaps at most two columns: a token has at most 2 hits and
// NCH = 2 * Tp / 128 columns always hold them (checked: status bit 16 otherwise).
#ifdef EG_NCH
GPSIMD_OP(engram_hits_t) {
    constexpr uint32_t NCH = EG_NCH;
    gp::Call c(call);
    auto ids_t = c.in<int32_t>(0);
    auto src_t = c.out<uint32_t>(0);
    auto dst_t = c.out<uint32_t>(1);
    if (ids_t.partitions() != 128 || ids_t.elems() != W || src_t.partitions() != 128 || src_t.elems() != NCH ||
        dst_t.elems() != NCH) {
        c.fail(8);
        return;
    }
    const gp::Range rows = c.split(128);
    alignas(16) int32_t ids[16 * W];
    const uint32_t tk = c.fetch(ids_t, rows.lo, rows.size(), ids);
    // padding everywhere in my partitions first (one 2-D iDMA per output)
    alignas(16) uint32_t zpad[16 * NCH], opad[16 * NCH];
    for (uint32_t k = 0; k < 16 * NCH; ++k) {
        zpad[k] = 0;
        opad[k] = OOB;
    }
    c.dma_2d(src_t.row(rows.lo), zpad, NCH * 4, rows.size(), NCH * 4, GP_PARTITION_STRIDE);
    c.dma_2d(dst_t.row(rows.lo), opad, NCH * 4, rows.size(), NCH * 4, GP_PARTITION_STRIDE);
    c.dma_wait(tk);
    alignas(16) uint32_t src[PCAP], dst[PCAP];
    uint32_t n = 0;
    for (uint32_t q = 0; q < rows.size(); ++q) {
        const int32_t *r = ids + q * W;
        const uint32_t p = rows.lo + q;
        for (uint32_t i = 0; i < NT; ++i) {
            const uint32_t base = (i * 128u + p) * NCOL;
            for (uint32_t j = 0; j < NCOL; ++j) {
                const uint32_t v = static_cast<uint32_t>(r[i * NCOL + j]);
                if (v < ROWW) {
                    if (n < PCAP) {
                        src[n] = v;
                        dst[n] = base + j;
                    }
                    ++n;
                }
            }
        }
    }
    c.dma_wait_all();                                  // my padding is in place ...
    uint32_t total;
    const uint32_t off = c.exscan_add(n, &total);      // ... on every core (the scan is a barrier)
    if (n > PCAP || total > NCH * 128u) {
        c.fail(16);
        return;
    }
    for (uint32_t m = 0; m < n; ++m) {                 // posted stores into any partition
        const uint32_t k = off + m;
        src_t.row(k & 127u)[k >> 7] = src[m];
        dst_t.row(k & 127u)[k >> 7] = dst[m];
    }
    __asm__ volatile("memw" ::: "memory");
}
#endif
