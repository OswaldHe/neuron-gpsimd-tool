/* kprint.h - printf for trn3 GPSIMD kernel libraries.
 *
 * Each core keeps a log buffer in its own data RAM (.bss of the kernel library). The host reads
 * it over PCIe BAR0 after execution (runner/run_neff.cpp --logs: scans every core's data RAM 0x80000-0xBFFFF
 * for KLOG_MAGIC). No firmware services are used.
 *
 * Buffer layout (all little-endian u32 words):
 *   [0] KLOG_MAGIC0  [1] KLOG_MAGIC1  [2] capacity (bytes)  [3] write position (bytes, monotonic)
 *   [4] core id (PRID)  [5] number of kprintf calls  [6..7] reserved, then the text ring.
 * Kernel-library data RAM budget is small (8 KB total), keep KLOG_SIZE modest.
 */
#pragma once
#include <stdarg.h>
#include <stdint.h>

#ifndef KLOG_SIZE
#define KLOG_SIZE 1024
#endif
#define KLOG_MAGIC0 0x4b4c4f47u /* "GOLK" -> bytes 'G','O','L','K' */
#define KLOG_MAGIC1 0x31534d47u /* "GMS1" */

struct klog {
    volatile uint32_t magic0, magic1, cap, pos, core, calls, rsv[2];
    volatile char text[KLOG_SIZE];
};
static struct klog klog_buf __attribute__((aligned(16)));

static inline uint32_t kprid(void) { uint32_t p; __asm__ volatile("rsr.prid %0" : "=r"(p)); return p & 0xff; }

static void klog_init(void) {
    if (klog_buf.magic0 == KLOG_MAGIC0 && klog_buf.magic1 == KLOG_MAGIC1) return;
    klog_buf.cap = KLOG_SIZE; klog_buf.pos = 0; klog_buf.calls = 0; klog_buf.core = kprid();
    klog_buf.magic1 = KLOG_MAGIC1; klog_buf.magic0 = KLOG_MAGIC0;
}
static void klog_putc(char c) { klog_buf.text[klog_buf.pos % KLOG_SIZE] = c; klog_buf.pos = klog_buf.pos + 1; }
static void klog_puts(const char *s) { while (*s) klog_putc(*s++); }
static void klog_putu(uint32_t v, uint32_t base, int width, char pad, int upper) {
    char buf[12]; int n = 0;
    const char *dig = upper ? "0123456789ABCDEF" : "0123456789abcdef";
    do { buf[n++] = dig[v % base]; v /= base; } while (v && n < 12);
    while (n < width && n < 12) buf[n++] = pad;
    while (n) klog_putc(buf[--n]);
}
/* Supports %d %i %u %x %X %p %s %c %% with optional '0' flag and width, and 'l' (ignored). */
static void kvprintf(const char *fmt, va_list ap) {
    klog_init();
    klog_buf.calls = klog_buf.calls + 1;
    for (; *fmt; fmt++) {
        if (*fmt != '%') { klog_putc(*fmt); continue; }
        char pad = ' '; int width = 0;
        fmt++;
        if (*fmt == '0') { pad = '0'; fmt++; }
        while (*fmt >= '0' && *fmt <= '9') width = width * 10 + (*fmt++ - '0');
        while (*fmt == 'l') fmt++;
        switch (*fmt) {
        case 'd': case 'i': { int32_t v = va_arg(ap, int32_t);
            if (v < 0) { klog_putc('-'); v = -v; } klog_putu((uint32_t)v, 10, width, pad, 0); break; }
        case 'u': klog_putu(va_arg(ap, uint32_t), 10, width, pad, 0); break;
        case 'x': klog_putu(va_arg(ap, uint32_t), 16, width, pad, 0); break;
        case 'X': klog_putu(va_arg(ap, uint32_t), 16, width, pad, 1); break;
        case 'p': klog_puts("0x"); klog_putu(va_arg(ap, uint32_t), 16, 8, '0', 0); break;
        case 's': { const char *s = va_arg(ap, const char *); klog_puts(s ? s : "(null)"); break; }
        case 'c': klog_putc((char)va_arg(ap, int)); break;
        case '%': klog_putc('%'); break;
        default: klog_putc('%'); if (*fmt) klog_putc(*fmt); else fmt--; break;
        }
    }
}
static void kprintf(const char *fmt, ...) __attribute__((format(printf, 1, 2), unused));
static void kprintf(const char *fmt, ...) { va_list ap; va_start(ap, fmt); kvprintf(fmt, ap); va_end(ap); }
