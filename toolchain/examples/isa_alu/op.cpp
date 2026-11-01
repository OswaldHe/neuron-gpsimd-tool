// isa_alu: every base-ISA integer instruction of the trn3 GPSIMD core, executed through inline asm
// on random + edge-case operands. Output column block k holds instruction k's results (order = OPS in
// kernel.py, which models each instruction's semantics in numpy).
#include <gpsimd/gpsimd.hpp>

#define R3(ins) [](uint32_t a, uint32_t b, uint32_t) { uint32_t r; __asm__(ins " %0, %1, %2" : "=r"(r) : "r"(a), "r"(b)); return r; }
#define DIV(ins) [](uint32_t a, uint32_t b, uint32_t) { uint32_t r, d = b ? b : 1; __asm__(ins " %0, %1, %2" : "=r"(r) : "r"(a), "r"(d)); return r; }
#define R2(ins) [](uint32_t a, uint32_t, uint32_t) { uint32_t r; __asm__(ins " %0, %1" : "=r"(r) : "r"(a)); return r; }
#define RI(ins, imm) [](uint32_t a, uint32_t, uint32_t) { uint32_t r; __asm__(ins " %0, %1, " #imm : "=r"(r) : "r"(a)); return r; }
#define RII(ins, i1, i2) [](uint32_t a, uint32_t, uint32_t) { uint32_t r; __asm__(ins " %0, %1, " #i1 ", " #i2 : "=r"(r) : "r"(a)); return r; }
#define SAR(setup, ins) [](uint32_t a, uint32_t b, uint32_t c) { uint32_t r; __asm__(setup " %2\n" ins : "=&r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
#define MOV(ins) [](uint32_t a, uint32_t b, uint32_t c) { uint32_t r = a; __asm__(ins " %0, %1, %2" : "+r"(r) : "r"(b), "r"(c)); return r; }
#define DEP(sh, w) [](uint32_t a, uint32_t, uint32_t c) { uint32_t r = c; __asm__("depbits %0, %1, " #sh ", " #w : "+r"(r) : "r"(a)); return r; }
// branch: result 1 if taken
#define BR2(ins) [](uint32_t a, uint32_t b, uint32_t) { uint32_t r; __asm__("movi %0, 1\n" ins " %1, %2, 1f\nmovi %0, 0\n1:" : "=&r"(r) : "r"(a), "r"(b)); return r; }
#define BR1(ins) [](uint32_t a, uint32_t, uint32_t) { uint32_t r; __asm__("movi %0, 1\n" ins " %1, 1f\nmovi %0, 0\n1:" : "=&r"(r) : "r"(a)); return r; }
#define BRI(ins, imm) [](uint32_t a, uint32_t, uint32_t) { uint32_t r; __asm__("movi %0, 1\n" ins " %1, " #imm ", 1f\nmovi %0, 0\n1:" : "=&r"(r) : "r"(a)); return r; }

#define OPS(X)                                                                                                    \
    X(R3("add")) X(R3("sub")) X(R3("addx2")) X(R3("addx4")) X(R3("addx8")) X(R3("subx2")) X(R3("subx4"))          \
    X(R3("subx8")) X(R3("and")) X(R3("or")) X(R3("xor")) X(R2("neg")) X(R2("abs"))                                \
    X(R3("mull")) X(R3("muluh")) X(R3("mulsh")) X(R3("mul16u")) X(R3("mul16s"))                                    \
    X(DIV("quou")) X(DIV("quos")) X(DIV("remu")) X(DIV("rems"))                                                       \
    X(R3("min")) X(R3("max")) X(R3("minu")) X(R3("maxu")) X(R3("salt")) X(R3("saltu"))                            \
    X(R2("nsa")) X(R2("nsau")) X(RI("sext", 7)) X(RI("sext", 15)) X(RI("sext", 22)) X(RI("clamps", 7))           \
    X(RI("clamps", 15)) X(RII("extui", 5, 12)) X(RII("extui", 0, 1)) X(RII("extui", 28, 4)) X(RII("extui", 16, 16)) \
    X(RI("slli", 1)) X(RI("slli", 17)) X(RI("slli", 31)) X(RI("srli", 1)) X(RI("srli", 15)) X(RI("srai", 3))      \
    X(RI("srai", 31)) X(RI("addi", -128)) X(RI("addi", 127)) X(RI("addmi", -32768)) X(RI("addmi", 32512))        \
    X(SAR("ssl", "sll %0, %1")) X(SAR("ssr", "srl %0, %1")) X(SAR("ssr", "sra %0, %1"))                          \
    X(SAR("ssr", "src %0, %1, %3")) X(SAR("ssa8l", "src %0, %1, %3")) X(SAR("ssa8b", "src %0, %1, %3"))          \
    X(SAR("ssai 13\n#", "src %0, %1, %3")) X(SAR("ssai 0\n#", "src %0, %1, %3"))                                  \
    X(MOV("moveqz")) X(MOV("movnez")) X(MOV("movltz")) X(MOV("movgez"))                                         \
    X(DEP(4, 8)) X(DEP(0, 1)) X(DEP(20, 12)) X(DEP(0, 16))                                                        \
    X(BR2("beq")) X(BR2("bne")) X(BR2("blt")) X(BR2("bge")) X(BR2("bltu")) X(BR2("bgeu")) X(BR2("bany"))          \
    X(BR2("bnone")) X(BR2("ball")) X(BR2("bnall")) X(BR2("bbc")) X(BR2("bbs")) X(BR1("beqz")) X(BR1("bnez"))      \
    X(BR1("bltz")) X(BR1("bgez")) X(BRI("beqi", 4)) X(BRI("bnei", -1)) X(BRI("blti", 8)) X(BRI("bgei", 256))      \
    X(BRI("bltui", 16)) X(BRI("bgeui", 32768)) X(BRI("bbci", 31)) X(BRI("bbsi", 0))

GPSIMD_OP(isa_alu) {
    gp::Call c(call);
    auto A = c.in<uint32_t>(0), B = c.in<uint32_t>(1), C = c.in<uint32_t>(2);
    auto O = c.out<uint32_t>(0);
    const uint32_t n = A.elems(), no = O.elems();
    constexpr uint32_t ROWS = 2;
    uint32_t la[ROWS * 16], lb[ROWS * 16], lc[ROWS * 16], lo[ROWS * 90 * 16];
    if (n > 16 || no > 90 * 16) { c.fail(8); return; }
    for (gp::Range blk : c.blocks(O.partitions(), ROWS)) {
        c.fetch(A, blk.lo, blk.size(), la);
        c.fetch(B, blk.lo, blk.size(), lb);
        c.dma_wait(c.fetch(C, blk.lo, blk.size(), lc));
        for (uint32_t q = 0; q < blk.size(); ++q) {
            uint32_t *ro = lo + q * no;
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t a = la[q * n + i], b = lb[q * n + i], cc = lc[q * n + i];
                uint32_t k = 0;
#define X(fn) ro[(k++) * n + i] = (fn)(a, b, cc);
                OPS(X)
#undef X
            }
        }
        c.dma_wait(c.flush(O, blk.lo, blk.size(), lo));
    }
}
