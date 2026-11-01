/* rt.c - GPSIMD operator runtime: call dispatch, execution receipts, log ring, cross-core barrier and
 * reductions, dtype conversion. Linked into every operator library by gpsimd-cc.
 *
 * Memory: the library's .data/.bss live in each core's private data RAM (8 KB budget per library) and
 * persist across executions of the loaded NEFF. The host (gpsimd-run) finds the receipt and the log
 * ring by their magic words and reads them over PCIe BAR0.
 */
#include <stdarg.h>
#include <stdint.h>

#include <gpsimd/gpsimd.h>

#ifndef GP_LOG_SIZE
#define GP_LOG_SIZE 1024
#endif
#define GP_BARRIER_SPIN_LIMIT (1u << 22)

/* ------------------------------------------------------------------ receipt */
typedef struct gp_receipt {
    uint32_t magic0, magic1;      /* 0x4D495350 "PSIM", 0x32504347 "GCP2" (magic0 written last) */
    uint32_t prid;                /* processor id register of the core that wrote it */
    uint32_t calls;               /* KERNEL_CALLs executed by this core since the library was loaded */
    uint32_t site_mask;           /* bit s: call site s ran (s < 32) */
    uint32_t last_site, last_cycles;
    uint32_t total_cycles;        /* sum of cycles spent in operators (mod 2^32) */
    uint32_t entry_pc;            /* address of the last entry stub (GPSIMD instruction RAM) */
    uint32_t sp, ra;              /* stack pointer / return address inside the dispatcher */
    uint32_t status;              /* OR of gp_call.status (bit 0: barrier timeout) */
    uint32_t user[4];
} gp_receipt;
_Static_assert(sizeof(gp_receipt) == 64, "receipt layout is shared with gpsimd-run");
volatile gp_receipt gp_receipt_data __attribute__((aligned(16)));

void gp_receipt_user(uint32_t idx, uint32_t value) {
    if (idx < 4) gp_receipt_data.user[idx] = value;
}

/* ------------------------------------------------------------------ log ring */
struct gp_log {
    volatile uint32_t magic0, magic1, cap, pos, core, calls, rsv[2];
    volatile char text[GP_LOG_SIZE];
};
static struct gp_log gp_log_buf __attribute__((aligned(16)));

static void log_putc(char c) {
    gp_log_buf.text[gp_log_buf.pos % GP_LOG_SIZE] = c;
    gp_log_buf.pos = gp_log_buf.pos + 1;
}
static void log_field(const char *txt, int len, int width, int left, char pad) {
    if (!left)
        for (int k = len; k < width; ++k) log_putc(pad);
    for (int k = 0; k < len; ++k) log_putc(txt[k]);
    if (left)
        for (int k = len; k < width; ++k) log_putc(' ');
}
static int fmt_u(char *buf, uint32_t v, uint32_t base, int upper) {
    char tmp[12];
    int n = 0, m = 0;
    const char *dig = upper ? "0123456789ABCDEF" : "0123456789abcdef";
    do { tmp[n++] = dig[v % base]; v /= base; } while (v && n < 12);
    while (n) buf[m++] = tmp[--n];
    return m;
}
/* %d %i %u %x %X %p %s %c %%, flags '-' (left) and '0', field width, 'l' ignored */
void gp_printf(const char *fmt, ...) {
    if (gp_log_buf.magic0 != 0x4b4c4f47u) {
        gp_log_buf.cap = GP_LOG_SIZE;
        gp_log_buf.pos = 0;
        gp_log_buf.calls = 0;
        gp_log_buf.core = gp_core_id();
        gp_log_buf.magic1 = 0x31534d47u;
        gp_log_buf.magic0 = 0x4b4c4f47u;
    }
    gp_log_buf.calls = gp_log_buf.calls + 1;
    va_list ap;
    va_start(ap, fmt);
    for (; *fmt; fmt++) {
        if (*fmt != '%') { log_putc(*fmt); continue; }
        char pad = ' ', buf[16];
        int width = 0, left = 0, n;
        fmt++;
        for (;; fmt++) {
            if (*fmt == '-') left = 1;
            else if (*fmt == '0') pad = '0';
            else break;
        }
        while (*fmt >= '0' && *fmt <= '9') width = width * 10 + (*fmt++ - '0');
        while (*fmt == 'l') fmt++;
        if (left) pad = ' ';
        switch (*fmt) {
            case 'd': case 'i': {
                int32_t v = va_arg(ap, int32_t);
                if (v < 0) {
                    buf[0] = '-';
                    n = 1 + fmt_u(buf + 1, 0u - (uint32_t)v, 10, 0);
                    if (pad == '0' && !left) { log_putc('-'); log_field(buf + 1, n - 1, width - 1, 0, '0'); break; }
                } else {
                    n = fmt_u(buf, (uint32_t)v, 10, 0);
                }
                log_field(buf, n, width, left, pad);
                break;
            }
            case 'u': n = fmt_u(buf, va_arg(ap, uint32_t), 10, 0); log_field(buf, n, width, left, pad); break;
            case 'x': n = fmt_u(buf, va_arg(ap, uint32_t), 16, 0); log_field(buf, n, width, left, pad); break;
            case 'X': n = fmt_u(buf, va_arg(ap, uint32_t), 16, 1); log_field(buf, n, width, left, pad); break;
            case 'p': {
                log_putc('0'); log_putc('x');
                n = fmt_u(buf, va_arg(ap, uint32_t), 16, 0);
                log_field(buf, n, 8, 0, '0');
                break;
            }
            case 'c': buf[0] = (char)va_arg(ap, int); log_field(buf, 1, width, left, ' '); break;
            case 's': {
                const char *str = va_arg(ap, const char *);
                int len = 0;
                while (str[len]) len++;
                log_field(str, len, width, left, ' ');
                break;
            }
            case '%': log_putc('%'); break;
            case 0: fmt--; break;
            default: log_putc('%'); log_putc(*fmt); break;
        }
    }
    va_end(ap);
}

/* ------------------------------------------------------------------ dispatch */
void gp_dispatch(const gp_site *site, gp_op_fn fn, uint32_t *exec_counter, uint32_t entry_pc) {
    uint32_t ra, sp;
    __asm__ volatile("mov %0, a0" : "=r"(ra));
    __asm__ volatile("mov %0, a1" : "=r"(sp));
    gp_call c;
    c.site = site;
    c.in = &gp_tiles[site->first_tile];
    c.out = c.in + site->n_in;
    c.scratch = c.out + site->n_out;
    c.core = gp_core_id();
    c.ncores = GP_NUM_CORES;
    c.exec = ++*exec_counter;
    c.barrier_gen = 0;
    c.status = 0;
    const uint32_t t0 = gp_cycles();
    fn(&c);
    gp_dma_end(&c);                          /* wait for the operator's iDMA transfers, give the channel back */
    gp_memw();                               /* all SBUF/HBM stores complete before the call completes */
    const uint32_t t1 = gp_cycles();
    volatile gp_receipt *r = &gp_receipt_data;
    r->prid = c.core;
    r->calls = r->calls + 1;
    if (site->site < 32) r->site_mask = r->site_mask | (1u << site->site);
    r->last_site = site->site;
    r->last_cycles = t1 - t0;
    r->total_cycles = r->total_cycles + (t1 - t0);
    r->entry_pc = entry_pc;
    r->sp = sp;
    r->ra = ra;
    r->status = r->status | c.status;
    r->magic1 = 0x32504347u;
    r->magic0 = 0x4D495350u;
}

/* Called once by the GPSIMD firmware when the library is loaded. */
void *_start(void *ctx) {
    (void)ctx;
    return 0;
}

/* ------------------------------------------------------------------ iDMA
 * Each GPSIMD core has one iDMA channel (Xtensa integrated DMA, registers on ERI 0x110000). The firmware
 * uses it to load kernel libraries: a ring of 64-byte descriptors in data RAM, CURR_DESC = the last
 * descriptor it processed. The runtime borrows the channel during a call: on first use it saves the
 * firmware's settings and last descriptor and points the channel at its own ring; at the end of the call
 * (gp_dma_end) it replays a harmless descriptor at the firmware's last descriptor address (so CURR_DESC is
 * exactly what the firmware left) and continues the firmware's ring after it. Without that, the firmware
 * later waits on descriptors the hardware never fetches (model stop fails). */
#define IDMA_SETTINGS 0x110000u
#define IDMA_DESC_START 0x110008u
#define IDMA_NUM_DESC 0x11000Cu
#define IDMA_DESC_INC 0x110010u
#define IDMA_CONTROL 0x110014u
#define IDMA_STATUS 0x110040u
#define IDMA_CURR_DESC 0x110044u
#define IDMA_FETCH_START (1u << 7)
#define GP_DMA_RING 8u

static inline uint32_t eri_rd(uint32_t a) { uint32_t v; __asm__ volatile("rer %0, %1" : "=r"(v) : "r"(a) : "memory"); return v; }
static inline void eri_wr(uint32_t a, uint32_t v) { __asm__ volatile("wer %0, %1" : : "r"(v), "r"(a) : "memory"); }

typedef struct { uint32_t w[16]; } __attribute__((aligned(64))) gp_desc64;
static gp_desc64 dma_ring[GP_DMA_RING];
static uint32_t dma_scratch[8] __attribute__((aligned(16)));
static struct {
    uint32_t active, settings, start, fw_last;
    uint32_t head;          /* next ring slot */
    uint32_t issued;        /* descriptors scheduled in this call */
} dma;

static uint32_t dma_wait_idle(void) {          /* wait until no descriptor is pending; returns STATUS */
    uint32_t st, spins = 0;
    do {
        st = eri_rd(IDMA_STATUS);
        const uint32_t state = st & 7u;
        if (state == 5u || state == 4u) break;   /* error / halted */
        if ((state == 0u || state == 3u) && eri_rd(IDMA_NUM_DESC) == 0) break;
    } while (++spins < (1u << 26));
    return st;
}

static void dma_begin(gp_call *c) {
    const uint32_t st = dma_wait_idle();
    if ((st & 7u) == 5u) c->status |= 64;
    dma.settings = eri_rd(IDMA_SETTINGS);
    dma.start = eri_rd(IDMA_DESC_START);
    dma.fw_last = eri_rd(IDMA_CURR_DESC);
    dma.head = 0;
    dma.issued = 0;
    dma.active = 1;
    eri_wr(IDMA_SETTINGS, dma.settings | IDMA_FETCH_START);
    eri_wr(IDMA_DESC_START, (uint32_t)&dma_ring[0]);
}

uint32_t gp_dma_2d(gp_call *c, volatile void *dst, const volatile void *src, uint32_t row_bytes, uint32_t nrows,
                   int32_t src_pitch, int32_t dst_pitch) {
    if (!dma.active) dma_begin(c);
    if (dma.head == GP_DMA_RING) {               /* ring full: drain, restart at slot 0 */
        if ((dma_wait_idle() & 7u) == 5u) c->status |= 64;
        eri_wr(IDMA_SETTINGS, dma.settings | IDMA_FETCH_START);
        eri_wr(IDMA_DESC_START, (uint32_t)&dma_ring[0]);
        dma.head = 0;
    }
    volatile uint32_t *d = dma_ring[dma.head++].w;
    d[0] = 1u | (1u << 4);                       /* 64-byte descriptor, 2D */
    d[1] = (uint32_t)src;
    d[2] = (uint32_t)dst;
    d[3] = row_bytes;
    d[4] = (uint32_t)src_pitch;
    d[5] = (uint32_t)dst_pitch;
    d[6] = nrows;
    for (uint32_t k = 7; k < 16; ++k) d[k] = 0;
    gp_memw();                                    /* descriptor and source data visible to the iDMA */
    eri_wr(IDMA_CONTROL, 1);
    eri_wr(IDMA_DESC_INC, 1);
    return ++dma.issued;
}
uint32_t gp_dma_copy(gp_call *c, volatile void *dst, const volatile void *src, uint32_t bytes) {
    return gp_dma_2d(c, dst, src, bytes, 1, 0, 0);
}
void gp_dma_wait(gp_call *c, uint32_t ticket) {     /* descriptors complete in order */
    if (!dma.active) return;
    uint32_t spins = 0;
    for (;;) {
        const uint32_t st = eri_rd(IDMA_STATUS), state = st & 7u;
        if (state == 5u) { c->status |= 64; return; }
        const uint32_t pending = eri_rd(IDMA_NUM_DESC);
        if ((int32_t)(dma.issued - pending - ticket) >= 0) {
            if (pending > 0) return;                     /* a later descriptor is still in flight */
            if (state == 0u || state == 3u) return;       /* channel idle: everything finished */
        }
        if (++spins == (1u << 26)) { c->status |= 64; return; }
    }
}
void gp_dma_wait_all(gp_call *c) {
    if (dma.active && (dma_wait_idle() & 7u) == 5u) c->status |= 64;
}
uint32_t gp_dma_status(void) { return eri_rd(IDMA_STATUS); }

void gp_dma_end(gp_call *c) {
    if (!dma.active) return;
    if ((dma_wait_idle() & 7u) == 5u) c->status |= 64;
    if (dma.fw_last) {                            /* replay a 16-byte local copy at the firmware's last slot */
        volatile uint32_t *fw = (volatile uint32_t *)dma.fw_last;
        uint32_t keep[16];
        for (uint32_t k = 0; k < 16; ++k) keep[k] = fw[k];
        fw[0] = 1u;                               /* 64-byte descriptor, 1D */
        fw[1] = (uint32_t)&dma_scratch[0];
        fw[2] = (uint32_t)&dma_scratch[4];
        fw[3] = 16u;
        for (uint32_t k = 4; k < 16; ++k) fw[k] = 0;
        gp_memw();
        eri_wr(IDMA_SETTINGS, dma.settings | IDMA_FETCH_START);
        eri_wr(IDMA_DESC_START, dma.fw_last);
        eri_wr(IDMA_DESC_INC, 1);
        dma_wait_idle();
        for (uint32_t k = 0; k < 16; ++k) fw[k] = keep[k];
        gp_memw();
        eri_wr(IDMA_SETTINGS, dma.settings | IDMA_FETCH_START);
        eri_wr(IDMA_DESC_START, dma.fw_last + 64u);   /* the firmware's next slot (its ring ends in a JUMP) */
    } else {
        eri_wr(IDMA_SETTINGS, dma.settings | IDMA_FETCH_START);
        eri_wr(IDMA_DESC_START, dma.start);
    }
    eri_wr(IDMA_SETTINGS, dma.settings);
    dma.active = 0;
}

/* ------------------------------------------------------------------ cross-core cooperation */
/* scratch words (zeroed by the Pool engine before every call):
 *   [0..7]   barrier arrival counters        [8..15]  bakery lock: choosing[core]
 *   [16..23] bakery lock: ticket[core]       [24..55] reduction slots (4 sets x 8 cores)
 *   [64..255] shared with the operator (gp_shared_words) */
#define SCR_BARRIER 0u
#define SCR_CHOOSING 8u
#define SCR_TICKET 16u
#define SCR_REDUCE 24u
#define SCR_USER 64u

static inline volatile uint32_t *scratch(const gp_call *c) { return (volatile uint32_t *)gp_row(c->scratch, 0); }

volatile uint32_t *gp_shared_words(const gp_call *c, uint32_t *count) {
    if (count) *count = c->scratch->elems - SCR_USER;
    return scratch(c) + SCR_USER;
}

/* Arrival flags are polled with one iDMA read of all 8 words (~450 cycles per poll) instead of 8 blocking
 * SBUF loads (~300 cycles each). */
void gp_barrier(gp_call *c) {
    volatile uint32_t *s = scratch(c) + SCR_BARRIER;
    const uint32_t g = ++c->barrier_gen;
    gp_memw();                               /* everything written before the barrier ... */
    s[c->core] = g;                          /* ... is visible before the arrival flag */
    gp_memw();
    uint32_t flags[8];
    for (uint32_t spins = 0;; ++spins) {
        gp_dma_wait(c, gp_dma_copy(c, flags, s, c->ncores * 4u));
        uint32_t k = 0;
        while (k < c->ncores && (int32_t)(flags[k] - g) >= 0) ++k;
        if (k == c->ncores) break;
        if (spins == GP_BARRIER_SPIN_LIMIT / 16u) { c->status |= 1; break; }
    }
}

/* Lamport's bakery lock over SBUF words (the trn3 GPSIMD core has no working exclusive-access
 * atomics: L32EX/S32EX hang on both local data RAM and SBUF). memw orders each core's SBUF accesses. */
void gp_lock(gp_call *c) {
    volatile uint32_t *s = scratch(c), *choosing = s + SCR_CHOOSING, *ticket = s + SCR_TICKET;
    const uint32_t me = c->core;
    choosing[me] = 1;
    gp_memw();
    uint32_t mx = 0;
    for (uint32_t k = 0; k < c->ncores; ++k) mx = ticket[k] > mx ? ticket[k] : mx;
    ticket[me] = mx + 1;
    gp_memw();
    choosing[me] = 0;
    gp_memw();
    const uint32_t mine = mx + 1;
    for (uint32_t k = 0; k < c->ncores; ++k) {
        if (k == me) continue;
        uint32_t spins = 0;
        while (choosing[k]) if (++spins == GP_BARRIER_SPIN_LIMIT) { c->status |= 4; break; }
        for (;;) {
            const uint32_t t = ticket[k];
            if (t == 0 || t > mine || (t == mine && k > me)) break;
            if (++spins == GP_BARRIER_SPIN_LIMIT) { c->status |= 4; break; }
        }
    }
    gp_memw();
}
void gp_unlock(gp_call *c) {
    gp_memw();
    scratch(c)[SCR_TICKET + c->core] = 0;
    gp_memw();
}
uint32_t gp_atomic_add_u32(gp_call *c, volatile uint32_t *word, uint32_t v) {
    gp_lock(c);
    const uint32_t old = *word;
    *word = old + v;
    gp_unlock(c);
    return old;
}

static void reduce_publish(gp_call *c, uint32_t v, volatile uint32_t **set) {
    volatile uint32_t *slots = scratch(c) + SCR_REDUCE + 8 * (c->barrier_gen & 3);
    slots[c->core] = v;
    gp_barrier(c);
    *set = slots;
}

uint32_t gp_allreduce_add_u32(gp_call *c, uint32_t v) {
    volatile uint32_t *s;
    reduce_publish(c, v, &s);
    uint32_t r = 0;
    for (uint32_t k = 0; k < c->ncores; ++k) r += s[k];
    return r;
}
uint32_t gp_allreduce_max_u32(gp_call *c, uint32_t v) {
    volatile uint32_t *s;
    reduce_publish(c, v, &s);
    uint32_t r = 0;
    for (uint32_t k = 0; k < c->ncores; ++k) r = s[k] > r ? s[k] : r;
    return r;
}
int32_t gp_allreduce_max_i32(gp_call *c, int32_t v) {
    volatile uint32_t *s;
    reduce_publish(c, (uint32_t)v, &s);
    int32_t r = INT32_MIN;
    for (uint32_t k = 0; k < c->ncores; ++k) r = (int32_t)s[k] > r ? (int32_t)s[k] : r;
    return r;
}
int32_t gp_allreduce_min_i32(gp_call *c, int32_t v) {
    volatile uint32_t *s;
    reduce_publish(c, (uint32_t)v, &s);
    int32_t r = INT32_MAX;
    for (uint32_t k = 0; k < c->ncores; ++k) r = (int32_t)s[k] < r ? (int32_t)s[k] : r;
    return r;
}
uint32_t gp_allreduce_add_f32(gp_call *c, uint32_t f) {
    volatile uint32_t *s;
    reduce_publish(c, f, &s);
    uint32_t r = s[0];
    for (uint32_t k = 1; k < c->ncores; ++k) r = sf_add(r, s[k]);
    return r;
}
uint32_t gp_allreduce_max_f32(gp_call *c, uint32_t f) {
    volatile uint32_t *s;
    reduce_publish(c, f, &s);
    uint32_t r = s[0];
    for (uint32_t k = 1; k < c->ncores; ++k) r = sf_max(r, s[k]);
    return r;
}
uint32_t gp_exscan_add_u32(gp_call *c, uint32_t v, uint32_t *total) {
    volatile uint32_t *s;
    reduce_publish(c, v, &s);
    uint32_t before = 0, all = 0;
    for (uint32_t k = 0; k < c->ncores; ++k) {
        if (k < c->core) before += s[k];
        all += s[k];
    }
    if (total) *total = all;
    return before;
}

/* ------------------------------------------------------------------ dtype conversion */
uint32_t gp_ld_f32(const gp_tile *t, uint32_t p, uint32_t i) {
    const uint32_t b = gp_ld_bits(t, p, i);
    switch (t->dtype) {
        case GP_FP32: return b;
        case GP_BF16: return sf_from_bf16((uint16_t)b);
        case GP_FP16: return sf_from_f16((uint16_t)b);
        case GP_INT8: return sf_from_i32((int8_t)b);
        case GP_INT16: return sf_from_i32((int16_t)b);
        case GP_INT32: return sf_from_i32((int32_t)b);
        case GP_UINT32: return sf_from_u32(b);
        default: return sf_from_i32((int32_t)b);         /* uint8, uint16 (zero-extended) */
    }
}
void gp_st_f32(const gp_tile *t, uint32_t p, uint32_t i, uint32_t f) {
    uint32_t b;
    switch (t->dtype) {
        case GP_FP32: b = f; break;
        case GP_BF16: b = sf_to_bf16(f); break;
        case GP_FP16: b = sf_to_f16(f); break;
        case GP_UINT32: b = sf_to_u32(f); break;
        default: b = (uint32_t)sf_to_i32(f); break;     /* low bits kept by gp_st_bits */
    }
    gp_st_bits(t, p, i, b);
}
