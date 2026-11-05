/* t3dis: trn3 GPSIMD disassembler skeleton.
 * Base-ISA instructions (op0 0..13) are decoded with the Xtensa ISA library (formats x24/x16a/x16b are
 * shared with ncore2gp); FLIX bundles (op0 14/15) are split with the trn3 length rule and printed as hex.
 * usage: t3dis <bin> <start> <end> [base_addr] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xtensa-isa.h>
static xtensa_isa isa;
static xtensa_insnbuf ib, sb;
int t3_len(const unsigned char *p) {          /* learned on the device (vecre/lenprobe.py) */
    int op0 = p[0] & 0xf;
    if (op0 < 8) return 3;
    if (op0 < 14) return 2;
    if (op0 == 14) return 16;
    switch (p[3] & 3) { case 1: return 8; case 3: return 10; default: return 16; }   /* op0 = 0xF: byte 3 [1:0] */
}
static void base(const unsigned char *p, int len, unsigned addr, char *out, size_t n) {
    unsigned char tmp[32] = {0}; memcpy(tmp, p, len);
    xtensa_insnbuf_from_chars(isa, ib, tmp, len);
    xtensa_format f = xtensa_format_decode(isa, ib);
    if (f == XTENSA_UNDEFINED || xtensa_format_length(isa, f) != len) { snprintf(out, n, "?"); return; }
    memset(sb, 0, sizeof(xtensa_insnbuf_word) * xtensa_insnbuf_size(isa));
    xtensa_format_get_slot(isa, f, 0, ib, sb);
    xtensa_opcode o = xtensa_opcode_decode(isa, f, 0, sb);
    if (o == XTENSA_UNDEFINED) { snprintf(out, n, "?"); return; }
    size_t k = snprintf(out, n, "%-10s", xtensa_opcode_name(isa, o));
    for (int i = 0; i < xtensa_opcode_num_operands(isa, o); i++) {
        uint32 v;
        if (xtensa_operand_get_field(isa, o, i, f, 0, sb, &v) || xtensa_operand_decode(isa, o, i, &v)) continue;
        const char *sep = i ? ", " : " ";
        if (xtensa_operand_is_register(isa, o, i) == 1)
            k += snprintf(out + k, n - k, "%s%s%u", sep, xtensa_regfile_shortname(isa, xtensa_operand_regfile(isa, o, i)), v);
        else if (xtensa_operand_is_PCrelative(isa, o, i) == 1) {
            uint32 t = v; xtensa_operand_undo_reloc(isa, o, i, &t, addr);
            k += snprintf(out + k, n - k, "%s0x%x", sep, t);
        } else k += snprintf(out + k, n - k, "%s%d", sep, (int)v);
    }
}
int main(int argc, char **argv) {
    char *dlls[] = {"/opt/aws/neuron/gpsimd/tools/ncore2gp/config/libisa-core-hw.so", "/opt/aws/neuron/gpsimd/tools/ncore2gp/config/libisa-core.so", NULL};
    xtensa_isa_status st; char *msg;
    isa = xtensa_isa_init(dlls, &st, &msg);
    ib = xtensa_insnbuf_alloc(isa); sb = xtensa_insnbuf_alloc(isa);
    FILE *fp = fopen(argv[1], "rb"); fseek(fp, 0, SEEK_END); long sz = ftell(fp); fseek(fp, 0, SEEK_SET);
    unsigned char *c = malloc(sz + 32); memset(c, 0, sz + 32); fread(c, 1, sz, fp);
    long a = strtol(argv[2], 0, 0), e = strtol(argv[3], 0, 0); unsigned b = argc > 4 ? strtoul(argv[4], 0, 0) : 0;
    char out[512];
    while (a < e && a < sz) {
        int L = t3_len(c + a);
        printf("%6lx: ", a + b);
        for (int i = 0; i < 16; i++) printf(i < L ? "%02x" : "  ", c[a + i]);
        if (L <= 3) { base(c + a, L, (unsigned)(a + b), out, sizeof out); printf("  %s\n", out); }
        else printf("  FLIX%d\n", L);
        a += L;
    }
    return 0;
}
