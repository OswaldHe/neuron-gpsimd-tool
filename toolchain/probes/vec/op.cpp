// vprobe: sandbox that executes raw instruction bytes (trn3 FLIX/vector bundles) on GPSIMD core 0 and
// reports the effect: exception cause, address-register changes, scratch-memory changes.
//
// Input tile  x  [128, 64] int32, one test per partition row:
//   w0      code length in bytes (0 = no test; at most 56)
//   w1..14  code bytes (memory order, little-endian words)
//   w16+k   initial value of a_k (k = 3..15): 0x8000xxxx -> scratch base + xxxx, else the absolute value
//   w32..47 optional 64-byte pattern written at scratch offset w48 before the test (w48 = -1: none)
// Output tile [128, 64] int32, one result per test row:
//   w0 status (1 = ran, 2 = exception), w1 EXCCAUSE, w2 exception pc - stub, w3 EXCVADDR,
//   w4 scratch base, w5 number of changed scratch bytes, w6 first changed offset, w7 last changed offset,
//   w16+k final a_k (k = 3..15), w32..63 128 bytes of scratch starting at the first changed offset
//   (or at offset w48 of the input if nothing changed and w48 >= 0)
// Row 127 of the output: harness diagnostics (exception table found, stub address, ...).
#include <gpsimd/gpsimd.hpp>

#define SCRATCH_WORDS 4096                  // 16 KB scratch, on the stack
#define GUARD_WORDS 256                     // 1 KB guard zones on both sides of the scratch
#define EXC_TABLE 0x81b40u                  // firmware XTOS exception handler table (9 entries)

static volatile uint32_t g_cause, g_pc, g_vaddr, g_count, g_recover;
static volatile uint32_t g_fw_base = 0;            // firmware vectors live at IRAM 0 (VECBASE = 0)
// the stub lives in instruction RAM (our .text); written with 32-bit stores, then isync
__attribute__((section(".text.vprobe_stub"), aligned(64))) static uint32_t stub_area[256] = {0};

// vector-context mode (input w49 bit 0): v0..v15 are loaded with VPAT before the test and stored after it,
// using the trn3 encodings found with this sandbox:
//   1f <v><a> 40 2d 22 e8 12 41   load  v<v> <- [a<a>], a<a> += 64
//   1f <v><a> 40 2d 44 e8 12 41   store v<v> -> [a<a>], a<a> += 64
static inline uint32_t vpat(uint32_t k, uint32_t i) { uint32_t x = (k * 16 + i + 1) * 0x9E3779B9u; return x ^ (x >> 15); }

extern "C" void vprobe_exc_handler(uint32_t *ef) {   // windowed C handler, ef = XEA3 ExcFrame
    g_cause = ef[6];
    g_pc = ef[4];
    g_vaddr = ef[7];
    g_count++;
    ef[4] = g_recover;                                // resume at the stub's epilogue
    ef[1] = 0;                                        // LCOUNT: cancel a zero-overhead loop the test set up
}

namespace {
struct Emit {
    uint8_t b[1024];
    uint32_t n = 0;
    void byte(uint32_t v) { b[n++] = (uint8_t)v; }
    void i24(uint32_t v) { byte(v); byte(v >> 8); byte(v >> 16); }
    void entry(uint32_t frame) { uint32_t imm = frame / 8; byte(0x36); byte(0x01 | ((imm & 0xf) << 4)); byte(imm >> 4); }
    void l32r(uint32_t t, uint32_t lit) {           // lit: byte offset of the literal in the stub
        int32_t imm = ((int32_t)lit - (int32_t)((n + 3) & ~3u)) >> 2;
        byte((t << 4) | 1); byte(imm & 0xff); byte((imm >> 8) & 0xff);
    }
    void l32i(uint32_t t, uint32_t s, uint32_t off) { byte((t << 4) | 2); byte((2 << 4) | s); byte(off / 4); }
    void s32i(uint32_t t, uint32_t s, uint32_t off) { byte((t << 4) | 2); byte((6 << 4) | s); byte(off / 4); }
    void fill(uint32_t to) {                          // nops up to offset `to` (3-byte nop, 2-byte nop.n)
        while (n < to) {
            uint32_t left = to - n;
            if (left == 2 || left == 4) { byte(0x3d); byte(0xf0); }
            else { byte(0xf0); byte(0x20); byte(0x00); }
        }
    }
};
}

static inline void isync() { __asm__ volatile("isync" ::: "memory"); }

// checksum of instruction RAM the tests must not touch: firmware vectors/dispatch [0, 0x2000) and the
// harness code around the handler (the stub area excluded)
static uint32_t guard_sum(uint32_t stub) {
    volatile uint32_t *fw = reinterpret_cast<volatile uint32_t *>((uintptr_t)g_fw_base);
    uint32_t h = 0x12345678u;
    for (uint32_t i = 0; i < 0x2000 / 4; ++i) h = (h ^ fw[i]) * 16777619u;
    uint32_t lo = ((uint32_t)(uintptr_t)&vprobe_exc_handler & ~0xfffu) - 0x2000, hi = lo + 0x6000;
    for (uint32_t a = lo; a < hi; a += 4) {
        if (a >= stub && a < stub + sizeof(stub_area)) continue;
        h = (h ^ *reinterpret_cast<volatile uint32_t *>((uintptr_t)a)) * 16777619u;
    }
    return h;
}

GPSIMD_OP(vprobe) {
    gp::Call c(call);
    auto in = c.in<int32_t>(0);
    auto o = c.out<int32_t>(0);
    {
        uint32_t cp; __asm__ volatile("rsr.cpenable %0" : "=r"(cp));
        gp_printf("core %u prid-ok cpenable=%x\n", c.core(), cp);
    }
    if (c.core() != 0) return;

    uint32_t scratch_all[SCRATCH_WORDS + 2 * GUARD_WORDS];
    uint32_t *scratch = scratch_all + GUARD_WORDS;
    for (uint32_t i = 0; i < GUARD_WORDS; ++i) { scratch_all[i] = 0x6B000000u | i; scratch[SCRATCH_WORDS + i] = 0x6C000000u | i; }
    uint32_t ctx[40];
    volatile uint32_t *tab = reinterpret_cast<volatile uint32_t *>(EXC_TABLE);
    uint32_t saved[9];
    bool table_ok = true;
    for (int k = 0; k < 9; ++k) saved[k] = tab[k];
    for (int k = 1; k < 9; ++k) table_ok = table_ok && saved[k] == saved[1];
    volatile int32_t *diag = o.row(127);
    diag[0] = table_ok;
    diag[1] = (int32_t)saved[0];
    diag[2] = (int32_t)saved[1];
    diag[3] = (int32_t)(uintptr_t)stub_area;
    diag[4] = (int32_t)(uintptr_t)&vprobe_exc_handler;
    diag[5] = (int32_t)(uintptr_t)scratch;
    if (!table_ok) return;
    for (int k = 1; k < 9; ++k) tab[k] = (uint32_t)(uintptr_t)&vprobe_exc_handler;

    const uint32_t base = (uint32_t)(uintptr_t)stub_area;
    const uint32_t guard0 = guard_sum(base);
    diag[6] = (int32_t)guard0;
    for (uint32_t t = 0; t < 127; ++t) {
        volatile int32_t *ti = in.row(t);
        volatile int32_t *to = o.row(t);
        uint32_t len = (uint32_t)ti[0];
        for (int k = 0; k < 64; ++k) to[k] = 0;
        if (len == 0 || len > 56) continue;
        // scratch pattern: word i = 0xA5000000 | i, then the optional 64-byte pattern
        for (uint32_t i = 0; i < SCRATCH_WORDS; ++i) scratch[i] = 0xA5000000u | i;
        int32_t poff = ti[48];
        if (poff >= 0 && poff + 64 <= SCRATCH_WORDS * 4)
            for (int k = 0; k < 16; ++k) scratch[poff / 4 + k] = (uint32_t)ti[32 + k];
        const uint32_t sb = (uint32_t)(uintptr_t)scratch;
        for (int k = 0; k < 40; ++k) ctx[k] = 0;
        const bool vmode = (uint32_t)ti[49] & 1;
        if (vmode)
            for (uint32_t k = 0; k < 16; ++k)
                for (uint32_t i = 0; i < 16; ++i) scratch[16 * k + i] = vpat(k, i);
        for (int k = 3; k < 16; ++k) {
            uint32_t v = (uint32_t)ti[16 + k];
            ctx[k] = ((v & 0xffff0000u) == 0x80000000u) ? sb + (v & 0xffffu) : v;   // 0x8000xxxx: scratch-relative
        }
        ctx[1] = sb + 0x2800;                         // a0 during the test
        ctx[2] = sb + 0x3000;                         // a2 during the test
        ctx[32] = sb;                                 // vector patterns (scratch 0x000..0x3ff)
        ctx[33] = sb + 0x400;                         // vector dump (scratch 0x400..0x7ff)
        // build the stub
        Emit e;
        e.i24(0); e.byte(0);                         // literal: &ctx at offset 0
        e.entry(8192);                                // offset 4; a large frame absorbs stores through a1
        e.l32r(2, 0);
        e.s32i(0, 2, 0);                              // keep the return address in ctx[0]
        if (vmode) {
            e.l32i(15, 2, 4 * 32);
            for (uint32_t k = 0; k < 16; ++k) { e.byte(0x1f); e.byte((k << 4) | 15); e.byte(0x40); e.byte(0x2d); e.byte(0x22); e.byte(0xe8); e.byte(0x12); e.byte(0x41); }
        }
        e.l32i(0, 2, 4);                              // a0 -> scratch while the test runs
        for (uint32_t k = 3; k < 16; ++k) e.l32i(k, 2, 4 * k);
        e.l32i(2, 2, 8);                              // a2 -> scratch
        e.fill(256);
        uint8_t code[56];
        for (int k = 0; k < 14; ++k) { uint32_t w = (uint32_t)ti[1 + k]; for (int j = 0; j < 4; ++j) code[4 * k + j] = (uint8_t)(w >> (8 * j)); }
        for (uint32_t k = 0; k < len; ++k) e.byte(code[k]);
        e.fill(320);
        uint32_t recover = e.n;                       // 320
        e.l32r(2, 0);
        e.s32i(0, 2, 64);                             // a0 after the test
        for (uint32_t k = 3; k < 16; ++k) e.s32i(k, 2, 64 + 4 * k);
        if (vmode) {
            e.l32i(2, 2, 4 * 33);
            for (uint32_t k = 0; k < 16; ++k) { e.byte(0x1f); e.byte((k << 4) | 2); e.byte(0x40); e.byte(0x2d); e.byte(0x44); e.byte(0xe8); e.byte(0x12); e.byte(0x41); }
            e.l32r(2, 0);
        }
        e.l32i(0, 2, 0);                              // restore the return address
        e.i24(0x00a032);                              // movi a3, 0
        e.i24(0x130230);                              // wsr.lcount a3 (cancel any loop the test set up)
        e.byte(0x1d); e.byte(0xf0);                   // retw.n
        e.fill((e.n + 3) & ~3u);
        uint32_t lit = (uint32_t)(uintptr_t)ctx;
        e.b[0] = lit; e.b[1] = lit >> 8; e.b[2] = lit >> 16; e.b[3] = lit >> 24;
        volatile uint32_t *dst = stub_area;
        for (uint32_t k = 0; k < e.n / 4; ++k)
            dst[k] = e.b[4 * k] | (e.b[4 * k + 1] << 8) | (e.b[4 * k + 2] << 16) | ((uint32_t)e.b[4 * k + 3] << 24);
        isync();
        g_recover = base + recover;
        uint32_t before = g_count;
        to[0] = 3;                                    // started (a hang leaves 3 here; the host reads SBUF)
        reinterpret_cast<void (*)()>(base + 4)();
        bool exc = g_count != before;
        to[0] = exc ? 2 : 1;
        to[1] = exc ? (int32_t)g_cause : 0;
        to[2] = exc ? (int32_t)(g_pc - base) : 0;
        to[3] = exc ? (int32_t)g_vaddr : 0;
        to[4] = (int32_t)sb;
        uint32_t nch = 0, first = 0xffffffffu, last = 0;
        for (uint32_t i = vmode ? 512 : 0; i < SCRATCH_WORDS; ++i) {
            uint32_t want = 0xA5000000u | i;
            if (poff >= 0 && i >= (uint32_t)poff / 4 && i < (uint32_t)poff / 4 + 16) want = (uint32_t)ti[32 + i - poff / 4];
            uint32_t d = scratch[i] ^ want;
            for (int j = 0; j < 4; ++j)
                if ((d >> (8 * j)) & 0xff) { ++nch; uint32_t off = 4 * i + j; if (off < first) first = off; last = off; }
        }
        to[5] = (int32_t)nch;
        to[6] = (int32_t)first;
        to[7] = (int32_t)last;
        for (int k = 3; k < 16; ++k) to[16 + k] = (int32_t)ctx[16 + k];
        uint32_t show = nch ? (first & ~3u) : (poff >= 0 ? (uint32_t)poff : 0xffffffffu);
        if (show != 0xffffffffu)
            for (int k = 0; k < 32 && show / 4 + k < SCRATCH_WORDS; ++k) to[32 + k] = (int32_t)scratch[show / 4 + k];
        to[8] = (int32_t)ctx[16];                     // a0 after the test
        if (vmode) {                                  // which vector registers changed, and the first two new values
            uint32_t mask = 0, nshow = 0;
            for (uint32_t k = 0; k < 16; ++k) {
                bool ch = false;
                for (uint32_t i = 0; i < 16; ++i) ch = ch || scratch[256 + 16 * k + i] != vpat(k, i);
                if (!ch) continue;
                mask |= 1u << k;
                if (nshow < 2) {
                    to[10 + nshow] = (int32_t)k;
                    for (uint32_t i = 0; i < 16; ++i) to[32 + 16 * nshow + i] = (int32_t)scratch[256 + 16 * k + i];
                    ++nshow;
                }
            }
            to[9] = (int32_t)mask;
        }
        // integrity: firmware vectors, harness code and the exception table must be unchanged
        bool bad = guard_sum(base) != guard0 || tab[0] != saved[0];
        for (uint32_t i = 0; i < GUARD_WORDS; ++i)
            bad = bad || scratch_all[i] != (0x6B000000u | i) || scratch[SCRATCH_WORDS + i] != (0x6C000000u | i);
        for (int k = 1; k < 9; ++k) bad = bad || tab[k] != (uint32_t)(uintptr_t)&vprobe_exc_handler;
        if (bad) { to[0] = 5; diag[7] = (int32_t)t + 1; break; }
    }
    for (int k = 1; k < 9; ++k) tab[k] = saved[k];
}
