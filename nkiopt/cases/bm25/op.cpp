// bm25_accum: term-at-a-time BM25 score accumulation over an impact-quantized inverted index in HBM.
//
// ins:  split [Q, 16] int32: for query term q, postings of docs in [c*D/8, (c+1)*D/8) are post[split[q][c] ..
//       split[q][c+1]) (true postings, padding excluded); params [1, 16] int32: gp_hbm (element 0 = core address
//       of the postings array `post`, mapped by gpsimd-run --hbm post).
// out:  scores [128, F] uint16, doc = p * F + f (F = D / 128).
// Core c owns docs [c*D/8, (c+1)*D/8) = output partitions [16c, 16c + 16): it zeroes a uint16 accumulator
// array in local RAM (iDMA from a zero row), streams its slice of every query term's postings from HBM
// (iDMA, 4 KB chunks, double buffered), accumulates acc[doc - base] += impact, and writes the 64 KB
// accumulator to its 16 output partitions with one 2D iDMA. Docs are unique within a term's list, so the
// 4-way unrolled update has no read-after-write hazard.
#include <gpsimd/gpsimd.hpp>

namespace {
constexpr uint32_t DOCS = 1u << 18, PER_CORE = DOCS / 8, CHUNK = 1024;   // postings per iDMA chunk
}

GPSIMD_OP(bm25_accum) {
    gp::Call c(call);
    auto split = c.in<int32_t>(0);
    auto params = c.in<int32_t>(1);
    auto out = c.out<uint16_t>(0);
    const uint32_t me = c.core(), Q = split.partitions();
    if (c.ncores() != 8 || out.partitions() != 128 || out.elems() * 128 != DOCS || split.elems() < 9 || Q > 64) {
        c.fail(8);
        return;
    }
    if (!gp::hbm_mapped(params)) { c.fail(16); return; }
    const volatile uint32_t *post = gp::hbm_ptr<uint32_t>(params, 0);

    alignas(16) uint16_t acc[PER_CORE];
    alignas(16) uint32_t buf[2][CHUNK];
    alignas(16) uint32_t zero[1024];
    int32_t sp[64 * 16];
    const uint32_t t_sp = c.fetch(split, 0, Q, sp);
    for (uint32_t i = 0; i < 1024; ++i) zero[i] = 0;
    c.dma_2d(acc, zero, 4096, PER_CORE * 2 / 4096, 0, 4096);      // zero the accumulators (src pitch 0)
    c.dma_wait(t_sp);

    // all (term, range) slices of this core, streamed through two chunk buffers
    const uint32_t base = me * PER_CORE;
    uint32_t q = 0, pos = 0, end = 0;
    auto next_range = [&]() {
        while (q < Q) {
            pos = static_cast<uint32_t>(sp[q * 16 + me]);
            end = static_cast<uint32_t>(sp[q * 16 + me + 1]);
            ++q;
            if (pos < end) return true;
        }
        return false;
    };
    struct Chunk { uint32_t n, ticket; };
    Chunk ch[2] = {{0, 0}, {0, 0}};
    auto issue = [&](uint32_t k) {
        if (pos >= end && !next_range()) { ch[k].n = 0; return; }
        const uint32_t n = end - pos < CHUNK ? end - pos : CHUNK;
        ch[k].n = n;
        ch[k].ticket = c.dma(buf[k], post + pos, n * 4);
        pos += n;
    };
    issue(0);
    issue(1);
    uint32_t k = 0, total = 0;
    while (ch[k].n) {
        c.dma_wait(ch[k].ticket);
        const uint32_t n = ch[k].n;
        const uint32_t *__restrict b = buf[k];
        uint16_t *__restrict a = acc;
        uint32_t i = 0;
        for (; i + 4 <= n; i += 4) {
            const uint32_t w0 = b[i], w1 = b[i + 1], w2 = b[i + 2], w3 = b[i + 3];
            const uint32_t d0 = (w0 >> 8) - base, d1 = (w1 >> 8) - base, d2 = (w2 >> 8) - base, d3 = (w3 >> 8) - base;
            const uint32_t a0 = a[d0], a1 = a[d1], a2 = a[d2], a3 = a[d3];
            a[d0] = static_cast<uint16_t>(a0 + (w0 & 0xFFu));
            a[d1] = static_cast<uint16_t>(a1 + (w1 & 0xFFu));
            a[d2] = static_cast<uint16_t>(a2 + (w2 & 0xFFu));
            a[d3] = static_cast<uint16_t>(a3 + (w3 & 0xFFu));
        }
        for (; i < n; ++i) {
            const uint32_t w = b[i];
            a[(w >> 8) - base] = static_cast<uint16_t>(a[(w >> 8) - base] + (w & 0xFFu));
        }
        total += n;
        issue(k);                                          // refill this buffer (the other one is in flight)
        k ^= 1;
    }
    c.dma_wait(c.flush(out, 16 * me, 16, acc));
    gp_receipt_user(0, total);
}
