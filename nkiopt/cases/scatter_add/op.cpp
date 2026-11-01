// scatter_add_f32: out = input; out[index[i]] += src[i] (D = 1, fp32, duplicate indices allowed).
// ins: input [128, F] fp32 (row r at [r / F, r % F]), index [128, K/128] int32 and src [128, K/128] fp32 (row-major
// K-vectors). out: [128, F] fp32. Core c owns rows [c*N/8, (c+1)*N/8) = partitions [16c, 16c+16): it streams
// all K (index, src) pairs through local RAM once (iDMA, 8 partitions per chunk), keeps the pairs that hit
// its rows (in index order), then applies them to its rows staged in local RAM, 4 partitions at a time
// (soft-float adds in index order) and writes the rows back with the iDMA.
#include <gpsimd/gpsimd.hpp>

GPSIMD_OP(scatter_add_f32) {
    gp::Call c(call);
    auto in_t = c.in<uint32_t>(0), sv_t = c.in<uint32_t>(2);
    auto ix_t = c.in<int32_t>(1);
    auto out_t = c.out<uint32_t>(0);
    const uint32_t F = in_t.elems(), KC = ix_t.elems(), me = c.core();
    const uint32_t N = 128 * F, per = N / 8, lo = me * per;
    constexpr uint32_t LMAX = 2560, HALF_P = 4, CH_P = 8;     // local RAM: ~68 KB of stack
    if (c.ncores() != 8 || out_t.elems() != F || F * HALF_P > 8192 || KC * CH_P > 1024) { c.fail(8); return; }
    alignas(16) int32_t ib[2][1024];
    alignas(16) uint32_t sb[2][1024];
    alignas(16) uint32_t rows[8192];
    uint32_t lrow[LMAX], lval[LMAX], nl_ = 0;
    // pass 1: collect this core's (row, value) pairs in index order
    const uint32_t nch = 128 / CH_P;
    c.fetch(ix_t, 0, CH_P, ib[0]);
    c.fetch(sv_t, 0, CH_P, sb[0]);
    for (uint32_t ch = 0; ch < nch; ++ch) {
        const uint32_t k = ch & 1;
        c.dma_wait_all();
        if (ch + 1 < nch) { c.fetch(ix_t, (ch + 1) * CH_P, CH_P, ib[k ^ 1]); c.fetch(sv_t, (ch + 1) * CH_P, CH_P, sb[k ^ 1]); }
        const uint32_t n = CH_P * KC;
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t r = static_cast<uint32_t>(ib[k][i]) - lo;
            if (r < per) {
                if (nl_ == LMAX) { c.fail(8); continue; }
                lrow[nl_] = r;
                lval[nl_++] = sb[k][i];
            }
        }
    }
    c.dma_wait_all();
    // pass 2: apply to my rows, a quarter range (4 partitions) at a time
    for (uint32_t h = 0; h < 16 / HALF_P; ++h) {
        const uint32_t p0 = 16 * me + HALF_P * h, base = h * HALF_P * F;
        uint32_t tk = 0;
        for (uint32_t j = 0; j < HALF_P; ++j) tk = c.dma(rows + j * F, in_t.row(p0 + j), F * 4);   // 1D per row
        c.dma_wait(tk);
        for (uint32_t i = 0; i < nl_; ++i) {
            const uint32_t r = lrow[i] - base;
            if (r < HALF_P * F) rows[r] = sf_add(rows[r], lval[i]);
        }
        for (uint32_t j = 0; j < HALF_P; ++j) tk = c.dma(out_t.row(p0 + j), rows + j * F, F * 4);
        c.dma_wait(tk);
    }
    gp_receipt_user(0, nl_);
}
