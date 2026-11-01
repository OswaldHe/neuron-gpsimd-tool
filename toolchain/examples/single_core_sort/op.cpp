// single_core_sort: heap sort on one core, in its local data RAM (row copied in/out with the iDMA).
#include <gpsimd/gpsimd.hpp>

static void sift_down(int32_t *a, uint32_t root, uint32_t n) {
    for (;;) {
        uint32_t child = 2 * root + 1;
        if (child >= n) return;
        if (child + 1 < n && a[child + 1] > a[child]) ++child;
        if (a[root] >= a[child]) return;
        const int32_t t = a[root];
        a[root] = a[child];
        a[child] = t;
        root = child;
    }
}

GPSIMD_OP(single_core_sort) {
    gp::Call c(call);
    auto X = c.in<int32_t>(0);
    auto Y = c.out<int32_t>(0);
    auto C = c.out<int32_t>(1);
    const uint32_t n = X.elems();
    if (n > 1024) { c.fail(8); return; }
    int32_t a[1024];                                  // 4 KB on this core's stack (local data RAM)
    c.dma_wait(c.dma(a, X.row(0), n * 4));
    if (c.core() == 0) {
        for (uint32_t i = n / 2; i-- > 0;) sift_down(a, i, n);
        for (uint32_t end = n; end-- > 1;) {
            const int32_t t = a[0];
            a[0] = a[end];
            a[end] = t;
            sift_down(a, 0, end);
        }
        c.dma_wait(c.dma(Y.row(0), a, n * 4));
    } else if (c.core() == 7) {                       // distinct count: O(n^2) on the local copy
        uint32_t distinct = 0;
        for (uint32_t i = 0; i < n; ++i) {
            bool seen = false;
            for (uint32_t j = 0; j < i && !seen; ++j) seen = a[j] == a[i];
            distinct += !seen;
        }
        C.store(0, 0, static_cast<int32_t>(distinct));
    }
}
