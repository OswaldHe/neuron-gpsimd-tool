#include <gpsimd/gpsimd.hpp>
#include <gpsimd/math.hpp>

#define TIME(label, n, body)                                                       \
    do {                                                                           \
        const uint32_t t0_ = gp_cycles();                                          \
        body;                                                                      \
        const uint32_t t1_ = gp_cycles();                                          \
        if (report) gp_printf("%-34s %6u cycles / op\n", label, (t1_ - t0_) / (n)); \
    } while (0)

GPSIMD_OP(bench) {
    gp::Call c(call);
    auto X = c.in<int32_t>(0);
    auto params = c.in<int32_t>(1);
    auto O = c.out<int32_t>(0);
    const bool report = c.core() == 0;
    const uint32_t N = 1024;
    volatile int32_t *sb = X.row(c.core() * 16);
    volatile int32_t *hb = gp::hbm_ptr<int32_t>(params, 0) + c.core() * 4096;
    static volatile int32_t local[256];
    uint32_t acc = 0;
    if (report) gp_printf("bench (core 0 timings; all 8 cores run the same loops concurrently)\n");
    TIME("local data RAM load (l32i)", N, for (uint32_t i = 0; i < N; ++i) acc += local[i & 255]);
    TIME("local data RAM store (s32i)", N, for (uint32_t i = 0; i < N; ++i) local[i & 255] = i);
    TIME("SBUF load via window (l32i)", N, for (uint32_t i = 0; i < N; ++i) acc += sb[i]);
    TIME("SBUF load, 4 independent", N, for (uint32_t i = 0; i < N; i += 4) acc += sb[i] + sb[i + 1] + sb[i + 2] + sb[i + 3]);
    TIME("SBUF store via window (s32i)", N, for (uint32_t i = 0; i < N; ++i) O.row(c.core() * 16)[i] = sb[i]);
    TIME("SBUF store only", N, for (uint32_t i = 0; i < N; ++i) O.row(c.core() * 16 + 1)[i] = i);
    TIME("SBUF load 8-bit (l8ui)", N, { volatile uint8_t *b8 = reinterpret_cast<volatile uint8_t *>(sb); for (uint32_t i = 0; i < N; ++i) acc += b8[i]; });
    TIME("HBM load via window", N, for (uint32_t i = 0; i < N; ++i) acc += hb[i]);
    TIME("HBM load, stride 4 KB", N, for (uint32_t i = 0; i < N; ++i) acc += hb[(i * 1024) & 0x3FFFF]);
    TIME("HBM store via window", N, for (uint32_t i = 0; i < N; ++i) hb[i + 2048] = i);
    {   // iDMA bulk copies (cycles per KB)
        int32_t lb[4096];
        TIME("iDMA SBUF->local 16 KB (per KB)", 16, c.dma_wait(c.dma(lb, sb, 16384)));
        TIME("iDMA local->SBUF 16 KB (per KB)", 16, c.dma_wait(c.dma(O.row(c.core() * 16 + 3), lb, 16384)));
        TIME("iDMA 16 partitions x 1 KB, 2D (per KB)", 16, c.dma_wait(c.dma_2d(lb, X.row(c.core() * 16), 1024, 16, GP_PARTITION_STRIDE, 1024)));
        TIME("iDMA SBUF->SBUF 16 KB (per KB)", 16, c.dma_wait(c.dma(O.row(c.core() * 16 + 4), sb, 16384)));
        TIME("iDMA HBM->local 16 KB (per KB)", 16, c.dma_wait(c.dma(lb, hb, 16384)));
        TIME("iDMA 4 KB SBUF->local, one transfer", 1, c.dma_wait(c.dma(lb, sb, 4096)));
        TIME("iDMA 16 B SBUF->local, one transfer", 1, c.dma_wait(c.dma(lb, sb, 16)));
        acc += lb[5];
    }
    gp::f32 f = GP_F32(1.000001), g = GP_F32(0.999);
    TIME("soft-float add", N, for (uint32_t i = 0; i < N; ++i) f = f + g);
    TIME("soft-float mul", N, for (uint32_t i = 0; i < N; ++i) f = f * g);
    TIME("soft-float div", N, for (uint32_t i = 0; i < N; ++i) f = f / g);
    TIME("soft-float sqrt", N, for (uint32_t i = 0; i < N; ++i) f = gp::sqrt(f));
    TIME("soft-float fma", N, for (uint32_t i = 0; i < N; ++i) f = gp::fma(f, g, g));
    TIME("math::exp", 256, for (uint32_t i = 0; i < 256; ++i) f = gp::math::exp(g) + f);
    TIME("math::tanh", 256, for (uint32_t i = 0; i < 256; ++i) f = gp::math::tanh(g) + f);
    volatile uint32_t d = 7;
    TIME("int32 divide (quou)", N, for (uint32_t i = 0; i < N; ++i) acc += (acc + i) / d);
    TIME("int32 multiply (mull)", N, for (uint32_t i = 0; i < N; ++i) acc = acc * (i | 1));
    TIME("barrier (8 cores)", 64, for (uint32_t i = 0; i < 64; ++i) c.barrier());
    TIME("lock+unlock (8 cores contending)", 64, for (uint32_t i = 0; i < 64; ++i) { c.lock(); c.unlock(); });
    TIME("allreduce_add (8 cores)", 64, for (uint32_t i = 0; i < 64; ++i) acc += c.allreduce_add(i));
    gp_receipt_user(0, acc + f.bits);
    // keep the output = input for the bit-exact check
    for (uint32_t p : c.partitions(O))
        for (uint32_t i = 0; i < O.elems(); ++i) O.row(p)[i] = X.row(p)[i];
}
