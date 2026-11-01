// ann_search: graph beam search with inline PQ codes + exact int8 re-rank (MIPS), one query per core.
//
// ins:  lut [B, M*256] int32 (M = 16) (query b's PQ table: lut[m*256 + j] = q_m . centroid_mj, exact integers),
//       q [B, 128] int8, entry [1, 16] int32 (entry id, its 16 PQ code bytes in words 1-4),
//       params [1, 16] int32 (gp_hbm: element 0 = core address of idx, mapped by gpsimd-run --hbm idx).
//       idx (HBM): N node records of 640 B (32 neighbour ids int32 + 32 x 16 neighbour code bytes), then the
//       N int8 vectors (128 B each).
// outs: ids [B, 16] int32: top-10 ids (exact score desc, id asc), [10] = hops (expansions), [11] = visited;
//       scores [B, 16] int32: their exact int8 dot products.
// Search (deterministic, mirrored by case.py search_ref): beam of <= L (dist, id) sorted ascending
// (dist = -PQ score, ties by id), expand the best unexpanded node: read its record (one iDMA), and for every
// neighbour not seen before compute the PQ distance (8 table lookups) and insert it if it beats the beam's
// worst; stop when every beam entry is expanded (or after ANN_HOPS expansions). Seen set: open-addressing
// hash table in local RAM (zeroed with the iDMA). Re-rank: the beam's vectors are read with the iDMA in
// batches of 8 and scored exactly.
#include <gpsimd/gpsimd.hpp>

namespace {
constexpr uint32_t N = 1u << 19, R = 32, M = 16, KS = 256, L = ANN_L, HOPS = ANN_HOPS, TOPK = 10;
constexpr uint32_t REC = R * 4 + R * M, HBITS = 14, HSIZE = 1u << HBITS;

inline uint32_t hslot(uint32_t v) { return (v * 2654435761u) >> (32 - HBITS); }

// insert v into the seen set; false if already present
inline bool see(uint32_t *h, uint32_t v) {
    uint32_t s = hslot(v);
    for (;;) {
        const uint32_t x = h[s];
        if (x == 0) { h[s] = v + 1; return true; }
        if (x == v + 1) return false;
        s = (s + 1) & (HSIZE - 1);
    }
}

inline bool less(int32_t da, uint32_t ia, int32_t db, uint32_t ib) { return da < db || (da == db && ia < ib); }
}  // namespace

GPSIMD_OP(ann_search) {
    gp::Call c(call);
    auto lut_t = c.in<int32_t>(0);
    auto q_t = c.in<int8_t>(1);
    auto ent_t = c.in<int32_t>(2);
    auto par_t = c.in<int32_t>(3);
    auto oid = c.out<int32_t>(0), osc = c.out<int32_t>(1);
    const uint32_t b = c.core(), B = lut_t.partitions();
    if (b >= B) return;
    if (lut_t.elems() != M * KS || q_t.elems() != 128 || oid.elems() != 16) { c.fail(8); return; }
    if (!gp::hbm_mapped(par_t)) { c.fail(16); return; }
    const volatile uint8_t *idx = gp::hbm_ptr<uint8_t>(par_t, 0);

    alignas(16) int32_t lut[M * KS];
    alignas(16) int8_t q[128];
    alignas(16) int32_t ent[16];
    alignas(16) uint32_t seen[HSIZE];
    alignas(16) uint32_t zero[1024];
    alignas(16) uint8_t rec[REC];
    alignas(16) int8_t vec[8][128];
    int32_t bd[L + 1];
    uint32_t bid[L + 1];
    uint8_t bex[L + 1];

    c.fetch(lut_t, b, 1, lut);
    c.fetch(q_t, b, 1, q);
    c.fetch(ent_t, 0, 1, ent);
    for (uint32_t i = 0; i < 1024; ++i) zero[i] = 0;
    c.dma_wait(c.dma_2d(seen, zero, 4096, HSIZE * 4 / 4096, 0, 4096));   // zero the hash table

    auto pq = [&](const uint8_t *code) {
        int32_t s = 0;
        for (uint32_t m = 0; m < M; ++m) s += lut[m * KS + code[m]];
        return -s;
    };
    const uint32_t e = static_cast<uint32_t>(ent[0]);
    uint32_t n = 1, cursor = 0, hops = 0, visited = 1;
    bd[0] = pq(reinterpret_cast<const uint8_t *>(&ent[1]));
    bid[0] = e;
    bex[0] = 0;
    see(seen, e);
    while (hops < HOPS) {
        while (cursor < n && bex[cursor]) ++cursor;                 // best unexpanded entry
        if (cursor >= n) break;
        bex[cursor] = 1;
        const uint32_t u = bid[cursor];
        ++hops;
        c.dma_wait(c.dma(rec, idx + static_cast<uintptr_t>(u) * REC, REC));
        const uint32_t *nb = reinterpret_cast<const uint32_t *>(rec);
        const uint8_t *codes = rec + R * 4;
        for (uint32_t r = 0; r < R; ++r) {
            const uint32_t v = nb[r];
            if (!see(seen, v)) continue;
            ++visited;
            const int32_t d = pq(codes + r * M);
            if (n == L && !less(d, v, bd[L - 1], bid[L - 1])) continue;
            uint32_t p = n < L ? n : L - 1;                         // shift worse entries down
            while (p > 0 && less(d, v, bd[p - 1], bid[p - 1])) {
                bd[p] = bd[p - 1]; bid[p] = bid[p - 1]; bex[p] = bex[p - 1];
                --p;
            }
            bd[p] = d; bid[p] = v; bex[p] = 0;
            if (n < L) ++n;
            if (p < cursor) cursor = p;
        }
    }

    // exact re-rank of the beam: top-10 by (score desc, id asc)
    int32_t ts[TOPK];
    uint32_t ti[TOPK], nt = 0;
    const volatile uint8_t *vecs = idx + static_cast<uintptr_t>(N) * REC;
    for (uint32_t k0 = 0; k0 < n; k0 += 8) {
        const uint32_t kn = n - k0 < 8 ? n - k0 : 8;
        uint32_t t = 0;
        for (uint32_t k = 0; k < kn; ++k) t = c.dma(vec[k], vecs + static_cast<uintptr_t>(bid[k0 + k]) * 128, 128);
        c.dma_wait(t);
        for (uint32_t k = 0; k < kn; ++k) {
            const uint32_t *xw = reinterpret_cast<const uint32_t *>(vec[k]);
            const uint32_t *qw = reinterpret_cast<const uint32_t *>(q);
            int32_t s = 0;
            for (uint32_t w = 0; w < 32; ++w) {
                const uint32_t a = xw[w], bq = qw[w];
                s += (static_cast<int32_t>(a << 24) >> 24) * (static_cast<int32_t>(bq << 24) >> 24);
                s += (static_cast<int32_t>(a << 16) >> 24) * (static_cast<int32_t>(bq << 16) >> 24);
                s += (static_cast<int32_t>(a << 8) >> 24) * (static_cast<int32_t>(bq << 8) >> 24);
                s += (static_cast<int32_t>(a) >> 24) * (static_cast<int32_t>(bq) >> 24);
            }
            const uint32_t id = bid[k0 + k];
            // insert into the top-10 (score desc, id asc)
            if (nt == TOPK && !(s > ts[TOPK - 1] || (s == ts[TOPK - 1] && id < ti[TOPK - 1]))) continue;
            uint32_t p = nt < TOPK ? nt : TOPK - 1;
            while (p > 0 && (s > ts[p - 1] || (s == ts[p - 1] && id < ti[p - 1]))) {
                ts[p] = ts[p - 1]; ti[p] = ti[p - 1];
                --p;
            }
            ts[p] = s; ti[p] = id;
            if (nt < TOPK) ++nt;
        }
    }
    int32_t ro[16], so[16];
    for (uint32_t k = 0; k < 16; ++k) { ro[k] = k < nt ? static_cast<int32_t>(ti[k]) : -1; so[k] = k < nt ? ts[k] : 0; }
    ro[10] = static_cast<int32_t>(hops);
    ro[11] = static_cast<int32_t>(visited);
    c.flush(oid, b, 1, ro);
    c.dma_wait(c.flush(osc, b, 1, so));
}
