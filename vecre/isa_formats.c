/* isa_formats: list the FLIX formats of the configured Xtensa core (ncore2gp): length, slots, the
 * bundle bits that each slot occupies, and the fixed format-identification bits.
 * usage: isa_formats [fmt_name]   (env XTENSA_CORE/XTENSA_SYSTEM select the core) */
#include <stdio.h>
#include <string.h>
#include <xtensa-isa.h>
static void clr(xtensa_isa isa, xtensa_insnbuf b) { memset(b, 0, sizeof(xtensa_insnbuf_word) * xtensa_insnbuf_size(isa)); }
#define xtensa_insnbuf_clear clr
int main(int argc, char **argv) {
    char *dlls[] = {"/opt/aws/neuron/gpsimd/tools/ncore2gp/config/libisa-core-hw.so", "/opt/aws/neuron/gpsimd/tools/ncore2gp/config/libisa-core.so", NULL};
    xtensa_isa_status st; char *msg;
    xtensa_isa isa = xtensa_isa_init(dlls, &st, &msg);
    if (!isa) { fprintf(stderr, "isa init failed: %s\n", msg); return 1; }
    xtensa_insnbuf ib = xtensa_insnbuf_alloc(isa), sb = xtensa_insnbuf_alloc(isa), z = xtensa_insnbuf_alloc(isa);
    int nf = xtensa_isa_num_formats(isa);
    printf("formats %d, max insn size %d, insnbuf words %d\n", nf, xtensa_isa_maxlength(isa), xtensa_insnbuf_size(isa));
    for (int f = 0; f < nf; f++) {
        const char *nm = xtensa_format_name(isa, f);
        if (argc > 1 && strcmp(argv[1], nm)) continue;
        int len = xtensa_format_length(isa, f), ns = xtensa_format_num_slots(isa, f);
        unsigned char ch[32];
        xtensa_insnbuf_clear(isa, ib);
        xtensa_format_encode(isa, f, ib);
        xtensa_insnbuf_to_chars(isa, ib, ch, len);
        printf("%-6s len %2d slots %d  format bits (memory order):", nm, len, ns);
        for (int i = 0; i < len; i++) printf(" %02x", ch[i]);
        printf("\n");
        for (int s = 0; s < ns; s++) {
            
            /* map bundle bit -> slot bit: set one bundle bit at a time */
            int width = 0; char map[512] = {0}; int mo = 0; int first = -1, last = -1, prevsb = -2, runstart = -1;
            for (int bit = 0; bit < len * 8; bit++) {
                xtensa_insnbuf_clear(isa, ib);
                xtensa_format_encode(isa, f, ib);
                ib[bit / 32] |= 1u << (bit % 32);
                xtensa_insnbuf_clear(isa, sb);
                xtensa_format_get_slot(isa, f, s, ib, sb);
                xtensa_insnbuf_clear(isa, z);
                xtensa_insnbuf_clear(isa, ib); xtensa_format_encode(isa, f, ib);
                xtensa_format_get_slot(isa, f, s, ib, z);
                for (int sbit = 0; sbit < 128; sbit++) {
                    if (((sb[sbit / 32] ^ z[sbit / 32]) >> (sbit % 32)) & 1) {
                        width++;
                        if (bit != last + 1 || sbit != prevsb + 1) {
                            if (runstart >= 0) mo += snprintf(map + mo, sizeof map - mo, "[%d..%d]", runstart, last);
                            runstart = bit;
                        }
                        last = bit; prevsb = sbit;
                        if (first < 0) first = bit;
                    }
                }
            }
            if (runstart >= 0) mo += snprintf(map + mo, sizeof map - mo, "[%d..%d]", runstart, last);
            printf("   slot %d %-14s width %3d bundle bits %s\n", s, "", width, map);
        }
    }
    return 0;
}
