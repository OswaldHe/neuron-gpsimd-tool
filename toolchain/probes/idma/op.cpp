#include <gpsimd/gpsimd.hpp>

static inline uint32_t rer(uint32_t a) { uint32_t v; __asm__ volatile("rer %0, %1" : "=r"(v) : "r"(a) : "memory"); return v; }
static inline void wer(uint32_t a, uint32_t v) { __asm__ volatile("wer %0, %1" : : "r"(v), "r"(a) : "memory"); }
enum { SETTINGS = 0x110000, TIMEOUT = 0x110004, DESC_START = 0x110008, NUM_DESC = 0x11000C, DESC_INC = 0x110010,
       CONTROL = 0x110014, STATUS = 0x110040, CURR_DESC = 0x110044 };

struct alignas(64) Desc64 {
    uint32_t control, src, dst, size, src_pitch, dst_pitch, nrows, rsv1;
    int32_t src_tile_pitch, dst_tile_pitch;
    uint32_t ntiles, pred_mask, rsv2, rsv3, ext_src, ext_dst;
};
static Desc64 g_desc[2];

// 2D copy: nrows rows of row_bytes, returns status (state | errors), cycles in *cyc
static uint32_t idma_2d(volatile void *dst, const volatile void *src, uint32_t row_bytes, uint32_t nrows,
                        uint32_t src_pitch, uint32_t dst_pitch, uint32_t *cyc) {
    Desc64 &d = g_desc[0];
    d = Desc64{};
    d.control = 1u | (1u << 4);                       // 64-byte descriptor, 2D subtype
    d.src = reinterpret_cast<uint32_t>(const_cast<void *>(src));
    d.dst = reinterpret_cast<uint32_t>(const_cast<void *>(dst));
    d.size = row_bytes;
    d.nrows = nrows;
    d.src_pitch = src_pitch;
    d.dst_pitch = dst_pitch;
    gp_memw();
    const uint32_t t0 = gp_cycles();
    wer(SETTINGS, rer(SETTINGS) | (1u << 7));         // fetch start
    wer(DESC_START, reinterpret_cast<uint32_t>(&d));
    wer(DESC_INC, 1);
    wer(CONTROL, 1);
    uint32_t st, spins = 0;
    do { st = rer(STATUS); } while (((st & 7) == 1 || (st & 7) == 2 || rer(NUM_DESC) != 0) && ++spins < 10000000);
    *cyc = gp_cycles() - t0;
    return st;
}

GPSIMD_OP(idma_probe) {
    gp::Call c(call);
    auto X = c.in<int32_t>(0);
    auto O = c.out<int32_t>(0);
    const bool rep = c.core() == 0 && c.exec() == 1;
    const uint32_t save_settings = rer(SETTINGS), save_curr = rer(CURR_DESC), save_status = rer(STATUS), save_start = rer(DESC_START);
    if (c.exec() == 1) gp_printf("core %u before: settings 0x%08x status 0x%08x start 0x%08x curr 0x%08x num %u\n", c.core(), save_settings, save_status, save_start, save_curr, rer(NUM_DESC));
    alignas(16) static int32_t buf[1024];
    uint32_t cyc, st;
#if MODE == 0
    for (uint32_t p : c.partitions(X)) for (uint32_t i = 0; i < 1024; ++i) O.row(p)[i] = X.row(p)[i];
    (void)cyc; (void)st; (void)save_start;
    return;
#elif MODE == 1
    for (uint32_t p : c.partitions(X)) for (uint32_t i = 0; i < 1024; ++i) O.row(p)[i] = X.row(p)[i];
    (void)cyc; (void)st; (void)save_start;
    return;
#endif
    for (uint32_t p : c.partitions(X)) {
        st = idma_2d(buf, X.row(p), 4096, 1, 0, 0, &cyc);
        if (rep && p == c.partitions(X).lo) gp_printf("SBUF->local 4096 B: status 0x%08x, %u cycles\n", st, cyc);
        st = idma_2d(O.row(p), buf, 4096, 1, 0, 0, &cyc);
        if (rep && p == c.partitions(X).lo) gp_printf("local->SBUF 4096 B: status 0x%08x, %u cycles\n", st, cyc);
        if ((st & 7) != 3 || (st >> 12)) c.fail(32);
    }
    if (rep) {   // 2D: 16 partitions x 256 bytes at once (partition stride 0x40000)
        st = idma_2d(buf, X.row(0), 256, 16, 0x40000, 256, &cyc);
        gp_printf("SBUF->local 2D 16 x 256 B: status 0x%08x, %u cycles, buf[64] %d == %d\n", st, cyc, buf[64], X.row(1)[0]);
    }
    if (rep) gp_printf("curr after our descriptor: 0x%08x (our descriptor at 0x%08x)\n", rer(CURR_DESC), reinterpret_cast<uint32_t>(&g_desc[0]));
#if MODE == 2
    return;
#endif
#if MODE == 4
    // replay: make the hardware process a harmless descriptor placed at the firmware's last descriptor
    // address, so CURR_DESC ends where the firmware left it; then restore that descriptor's bytes.
    if (save_curr) {
        volatile uint32_t *fw = reinterpret_cast<volatile uint32_t *>(save_curr);
        uint32_t keep[16];
        for (int k = 0; k < 16; ++k) keep[k] = fw[k];
        const uint32_t nop[16] = {1u, reinterpret_cast<uint32_t>(buf), reinterpret_cast<uint32_t>(buf + 4), 16u, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0};
        for (int k = 0; k < 16; ++k) fw[k] = nop[k];
        gp_memw();
        wer(SETTINGS, save_settings | (1u << 7));
        wer(DESC_START, save_curr);
        wer(DESC_INC, 1);
        uint32_t st2, spins = 0;
        do { st2 = rer(STATUS); } while (((st2 & 7) == 1 || (st2 & 7) == 2 || rer(NUM_DESC) != 0) && ++spins < 10000000);
        for (int k = 0; k < 16; ++k) fw[k] = keep[k];
        gp_memw();
    }
#endif
    // restore the firmware's fetch pointer: CURR_DESC is the last processed descriptor -> continue after it
    wer(SETTINGS, save_settings | (1u << 7));
    wer(DESC_START, save_curr ? save_curr + 64 : save_start);
    wer(SETTINGS, save_settings);
    if (c.exec() == 2 && c.core() < 2) gp_printf("core %u exit: curr 0x%08x (fw last 0x%08x) status 0x%08x\n", c.core(), rer(CURR_DESC), save_curr, rer(STATUS));
    if (rep) gp_printf("after: settings 0x%08x status 0x%08x curr 0x%08x num %u\n", rer(SETTINGS), rer(STATUS), rer(CURR_DESC), rer(NUM_DESC));
}
