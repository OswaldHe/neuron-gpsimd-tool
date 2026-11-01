/* xslot: decode Xtensa instructions/FLIX bundles slot by slot with the ncore2gp ISA library.
 *
 * usage:
 *   xslot hex <hexbytes>                 decode one instruction/bundle given as hex (memory order)
 *   xslot file <bin> <base> <start> <end> walk code: length via the ISA length decoder, but
 *                                          for op0=0xF bundles try 8 and 16 bytes and keep the
 *                                          interpretation whose slots all decode (trn3 differs
 *                                          from ncore2gp for some formats)
 * Unknown slot opcodes (trn3-specific TIE) are printed as "?<slot bits>".
 * build: see tools/build_xslot.sh
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xtensa-isa.h>

static xtensa_isa isa;
static xtensa_insnbuf ibuf, sbuf;

/* returns number of unknown slots (or -1 if the format itself doesn't decode); prints into out */
static int decode(const unsigned char *p, int len, unsigned addr, char *out, size_t outsz) {
    unsigned char tmp[32] = {0};
    memcpy(tmp, p, len);
    xtensa_insnbuf_from_chars(isa, ibuf, tmp, len);
    xtensa_format fmt = xtensa_format_decode(isa, ibuf);
    if (fmt == XTENSA_UNDEFINED) { snprintf(out, outsz, "<fmt?>"); return -1; }
    if (xtensa_format_length(isa, fmt) != len) { snprintf(out, outsz, "<len mismatch>"); return -1; }
    int ns = xtensa_format_num_slots(isa, fmt), unknown = 0;
    size_t o = 0;
    if (ns > 1) o += snprintf(out + o, outsz - o, "{%s: ", xtensa_format_name(isa, fmt));
    for (int s = 0; s < ns; s++) {
        xtensa_format_get_slot(isa, fmt, s, ibuf, sbuf);
        xtensa_opcode opc = xtensa_opcode_decode(isa, fmt, s, sbuf);
        if (s) o += snprintf(out + o, outsz - o, "; ");
        if (opc == XTENSA_UNDEFINED) { o += snprintf(out + o, outsz - o, "?%08x", (unsigned)sbuf[0]); unknown++; continue; }
        const char *nm = xtensa_opcode_name(isa, opc);
        if (!strcmp(nm, "nop") && ns > 1) { o += snprintf(out + o, outsz - o, "nop"); continue; }
        o += snprintf(out + o, outsz - o, "%s", nm);
        int nopnd = xtensa_opcode_num_operands(isa, opc);
        for (int k = 0; k < nopnd; k++) {
            uint32 v;
            if (xtensa_operand_get_field(isa, opc, k, fmt, s, sbuf, &v)) continue;
            if (xtensa_operand_decode(isa, opc, k, &v)) continue;
            const char *sep = k ? ", " : " ";
            if (xtensa_operand_is_register(isa, opc, k) == 1) {
                xtensa_regfile rf = xtensa_operand_regfile(isa, opc, k);
                o += snprintf(out + o, outsz - o, "%s%s%u", sep, xtensa_regfile_shortname(isa, rf), v);
            } else if (xtensa_operand_is_PCrelative(isa, opc, k) == 1) {
                uint32 t = v;
                xtensa_operand_undo_reloc(isa, opc, k, &t, addr);
                o += snprintf(out + o, outsz - o, "%s0x%x", sep, t);
            } else {
                o += snprintf(out + o, outsz - o, "%s%d", sep, (int)v);
            }
            if (o >= outsz) break;
        }
    }
    if (ns > 1) o += snprintf(out + o, outsz - o, "}");
    return unknown;
}

static void hexs(const unsigned char *p, int n, char *o) { for (int i = 0; i < n; i++) sprintf(o + 2 * i, "%02x", p[i]); }

int main(int argc, char **argv) {
    char *dlls[] = {"/opt/aws/neuron/gpsimd/tools/ncore2gp/config/libisa-core-hw.so", "/opt/aws/neuron/gpsimd/tools/ncore2gp/config/libisa-core.so", NULL};
    xtensa_isa_status st; char *msg;
    isa = xtensa_isa_init(dlls, &st, &msg);
    if (!isa) { fprintf(stderr, "isa init failed: %s\n", msg); return 1; }
    ibuf = xtensa_insnbuf_alloc(isa); sbuf = xtensa_insnbuf_alloc(isa);
    char out[1024], hx[80];
    if (argc >= 3 && !strcmp(argv[1], "hex")) {
        unsigned char b[32]; int n = strlen(argv[2]) / 2;
        for (int i = 0; i < n && i < 32; i++) sscanf(argv[2] + 2 * i, "%2hhx", &b[i]);
        int len = xtensa_isa_length_from_chars(isa, b);
        int u = decode(b, len, 0, out, sizeof out);
        printf("len %d unknown %d: %s\n", len, u, out);
        return 0;
    }
    if (argc < 6 || strcmp(argv[1], "file")) { fprintf(stderr, "usage: xslot hex HEX | file BIN BASE START END\n"); return 1; }
    FILE *f = fopen(argv[2], "rb"); fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *code = malloc(sz + 32); memset(code, 0, sz + 32); fread(code, 1, sz, f); fclose(f);
    unsigned base = strtoul(argv[3], 0, 0), start = strtoul(argv[4], 0, 0), end = strtoul(argv[5], 0, 0);
    unsigned i = start - base, stop = end - base;
    while (i < stop && i < (unsigned)sz) {
        unsigned char *p = code + i;
        int len = xtensa_isa_length_from_chars(isa, p);
        if (len == XTENSA_UNDEFINED || len <= 0) len = 1;
        int u = decode(p, len, base + i, out, sizeof out);
        if ((p[0] & 0xF) == 0xF && u != 0) {          /* try the other bundle length */
            int alt = len == 16 ? 8 : 16; char out2[1024];
            int u2 = decode(p, alt, base + i, out2, sizeof out2);
            if (u2 >= 0 && (u < 0 || u2 < u)) { len = alt; u = u2; strcpy(out, out2); }
        }
        hexs(p, len, hx);
        printf("%08x: %-34s %s\n", base + i, hx, out);
        int is_ret = (len == 2 && (p[0] == 0x1d || p[0] == 0x0d) && p[1] == 0xf0) ||
                     (len == 3 && (p[0] == 0x90 || p[0] == 0x80) && p[1] == 0 && p[2] == 0);
        i += len;
        if (is_ret) while (i < stop && (i & 3) && code[i] == 0) i++;
    }
    return 0;
}
