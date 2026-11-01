// transpose: out[j][i] = in[i][j]. For each output partition j, one 2D iDMA descriptor gathers input
// column j (one element from each input partition: row size = element size, source pitch = partition
// stride) into local RAM; a second copies it to the output partition. Double-buffered.
#include <gpsimd/gpsimd.hpp>

template <typename T>
static void transpose(const gp::Call &c, const gp::Tile<T> &in, const gp::Tile<T> &out) {
    using S = typename gp::Tile<T>::S;
    const uint32_t n = out.elems(), es = sizeof(S);      // n = input partitions
    S col[2][128];
    uint32_t got[2] = {0, 0}, put[2] = {0, 0};
    const gp::Range r = c.partitions(out);
    if (n > 128 || !in.contiguous() || !out.contiguous()) { c.fail(8); return; }
    if (r.lo < r.hi) got[0] = c.dma_2d(col[0], in.row(0) + r.lo, es, n, GP_PARTITION_STRIDE, es);
    for (uint32_t j = r.lo; j < r.hi; ++j) {
        const uint32_t cur = (j - r.lo) & 1, nxt = cur ^ 1;
        if (j + 1 < r.hi) {
            if (put[nxt]) c.dma_wait(put[nxt]);           // buffer nxt written out?
            got[nxt] = c.dma_2d(col[nxt], in.row(0) + j + 1, es, n, GP_PARTITION_STRIDE, es);
        }
        c.dma_wait(got[cur]);
        put[cur] = c.dma(out.row(j), col[cur], n * es);
    }
    c.dma_wait_all();
}

GPSIMD_OP(transpose) {
    gp::Call c(call);
    transpose(c, c.in<int32_t>(0), c.out<int32_t>(0));
    transpose(c, c.in<gp::bf16>(1), c.out<gp::bf16>(1));
}
