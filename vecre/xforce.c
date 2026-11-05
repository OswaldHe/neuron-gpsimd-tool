/* xforce: decode bytes as a FORCED ncore2gp format (format-id bits ignored), slot by slot.
 * usage: xforce <fmt|all8|all16> <hex> [...]   prints, per format, the slot opcodes + operands.
 * Library use: score(bytes,len,fmt) = number of slots that decode to a known opcode. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xtensa-isa.h>
static xtensa_isa isa;
static xtensa_insnbuf ib, sb;
static int dec(const unsigned char *p, int fmt, char *out, size_t n) {
    int len = xtensa_format_length(isa, fmt);
    unsigned char tmp[32] = {0}; memcpy(tmp, p, len);
    xtensa_insnbuf_from_chars(isa, ib, tmp, len);
    int ns = xtensa_format_num_slots(isa, fmt), ok = 0; size_t o = 0;
    o += snprintf(out + o, n - o, "%s{", xtensa_format_name(isa, fmt));
    for (int s = 0; s < ns; s++) {
        memset(sb, 0, sizeof(xtensa_insnbuf_word) * xtensa_insnbuf_size(isa));
        xtensa_format_get_slot(isa, fmt, s, ib, sb);
        xtensa_opcode opc = xtensa_opcode_decode(isa, fmt, s, sb);
        if (s) o += snprintf(out + o, n - o, "; ");
        if (opc == XTENSA_UNDEFINED) { o += snprintf(out + o, n - o, "?%x", (unsigned)sb[0]); continue; }
        ok++;
        o += snprintf(out + o, n - o, "%s", xtensa_opcode_name(isa, opc));
        int no = xtensa_opcode_num_operands(isa, opc);
        for (int k = 0; k < no; k++) {
            uint32 v;
            if (xtensa_operand_get_field(isa, opc, k, fmt, s, sb, &v)) continue;
            if (xtensa_operand_decode(isa, opc, k, &v)) continue;
            if (xtensa_operand_is_register(isa, opc, k) == 1)
                o += snprintf(out + o, n - o, "%s%s%u", k ? "," : " ", xtensa_regfile_shortname(isa, xtensa_operand_regfile(isa, opc, k)), v);
            else o += snprintf(out + o, n - o, "%s%d", k ? "," : " ", (int)v);
        }
    }
    snprintf(out + o, n - o, "}");
    return ok * 100 + ns;
}
int main(int argc, char **argv) {
    char *dlls[] = {"/opt/aws/neuron/gpsimd/tools/ncore2gp/config/libisa-core-hw.so", "/opt/aws/neuron/gpsimd/tools/ncore2gp/config/libisa-core.so", NULL};
    xtensa_isa_status st; char *msg;
    isa = xtensa_isa_init(dlls, &st, &msg);
    ib = xtensa_insnbuf_alloc(isa); sb = xtensa_insnbuf_alloc(isa);
    char out[2048];
    for (int a = 2; a < argc; a++) {
        unsigned char b[32] = {0}; int n = strlen(argv[a]) / 2;
        for (int i = 0; i < n && i < 32; i++) sscanf(argv[a] + 2 * i, "%2hhx", &b[i]);
        for (int f = 0; f < xtensa_isa_num_formats(isa); f++) {
            const char *fn = xtensa_format_name(isa, f); int len = xtensa_format_length(isa, f);
            if (!strcmp(argv[1], "all8") ? len != 8 : !strcmp(argv[1], "all16") ? len != 16 : strcmp(argv[1], fn)) continue;
            int r = dec(b, f, out, sizeof out);
            printf("%s  %d/%d  %s\n", argv[a], r / 100, r % 100, out);
        }
    }
    return 0;
}
