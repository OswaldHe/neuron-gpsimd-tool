// isa_mem: explicit load/store instructions on SBUF (window) and local data RAM.
#include <gpsimd/gpsimd.hpp>

#define LD(ins, off) ({ uint32_t r_; __asm__ volatile(ins " %0, %1, " #off : "=r"(r_) : "r"(a) : "memory"); r_; })

// 15 load results for the word at `word`; `pair` = first word of its even/odd pair
static void loads(volatile uint32_t *word, volatile uint32_t *pair, uint32_t *r) {
    const uint32_t a = reinterpret_cast<uint32_t>(word);
    r[0] = LD("l8ui", 0);
    r[1] = LD("l8ui", 1);
    r[2] = LD("l8ui", 2);
    r[3] = LD("l8ui", 3);
    r[4] = LD("l16ui", 0);
    r[5] = LD("l16ui", 2);
    r[6] = LD("l16si", 0);
    r[7] = LD("l16si", 2);
    r[8] = LD("l32i", 0);
    r[9] = LD("l32ai", 0);
    uint32_t lit;
    __asm__ volatile("movi %0, 0x12345678" : "=r"(lit));        // assembled as l32r from the literal pool
    r[10] = lit;
    const uint32_t pb = reinterpret_cast<uint32_t>(pair);
    uint32_t v0, v1;
    __asm__ volatile("l32i %0, %2, 0\nl32i %1, %2, 4" : "=&r"(v0), "=&r"(v1) : "r"(pb) : "memory");
    r[11] = word == pair ? v1 : v0;                              // l32i with offset: the other word
    r[12] = r[0] | r[1] << 8 | r[2] << 16 | r[3] << 24;
    r[13] = ~r[8];
    r[14] = r[8] + r[11];
}

// three words at dst: byte-reversed w (4 x s8i), half-swapped w (2 x s16i), w ^ 0xA5A5A5A5 (s32ri)
static void stores(volatile uint32_t *dst, uint32_t w) {
    uint32_t d = reinterpret_cast<uint32_t>(dst), t = w;
    __asm__ volatile("s8i %1, %0, 3\nsrli %1, %1, 8\ns8i %1, %0, 2\nsrli %1, %1, 8\n"
                     "s8i %1, %0, 1\nsrli %1, %1, 8\ns8i %1, %0, 0"
                     : "+r"(d), "+r"(t) : : "memory");
    t = w;
    __asm__ volatile("s16i %1, %0, 6\nsrli %1, %1, 16\ns16i %1, %0, 4" : "+r"(d), "+r"(t) : : "memory");
    __asm__ volatile("s32ri %1, %0, 8" : : "r"(d), "r"(w ^ 0xA5A5A5A5u) : "memory");
}

GPSIMD_OP(isa_mem) {
    gp::Call c(call);
    auto X = c.in<uint32_t>(0);
    auto O = c.out<uint32_t>(0);
    auto S = c.out<uint32_t>(1);
    const uint32_t n = X.elems(), NT = 15;
    if (n > 16) { c.fail(8); return; }
    for (uint32_t p : c.partitions(X)) {
        volatile uint32_t *row = X.row(p);
        uint32_t local[16];
        for (uint32_t i = 0; i < n; ++i) local[i] = row[i];
        volatile uint32_t *lrow = local;
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t r[15];
            for (uint32_t kind = 0; kind < 2; ++kind) {
                volatile uint32_t *base = kind ? lrow : row;
                loads(base + i, base + (i & ~1u), r);
                for (uint32_t k = 0; k < NT; ++k) O.store(p, (kind * NT + k) * n + i, r[k]);
            }
        }
        // stores into SBUF: staged in S's first 3n columns, then permuted into blocks 0..2
        for (uint32_t i = 0; i < n; ++i) stores(S.row(p) + 3 * i, row[i]);
        uint32_t staged[48];
        for (uint32_t k = 0; k < 3 * n; ++k) staged[k] = S.row(p)[k];
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t k = 0; k < 3; ++k) S.store(p, k * n + i, staged[3 * i + k]);
        // stores into local RAM: blocks 3..5
        for (uint32_t i = 0; i < n; ++i) {
            volatile uint32_t tmp[3];
            stores(tmp, row[i]);
            for (uint32_t k = 0; k < 3; ++k) S.store(p, (3 + k) * n + i, tmp[k]);
        }
    }
}
