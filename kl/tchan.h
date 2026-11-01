/* Timing side channel: each kernel invocation emits one 4-bit symbol as a busy-wait of
 * (symbol + 1) * TCHAN_UNIT iterations; the host recovers it from nrt_execute latency.
 * Protocol (persistent .bss counter): invocation 0 = warm-up; then calibration levels 0..15,
 * each TCHAN_REPEAT times; then data symbols, each TCHAN_REPEAT times. */
#pragma once
#include <stdint.h>
#define TCHAN_UNIT 30000
#define TCHAN_REPEAT 3
#define TCHAN_PREAMBLE (1 + 16 * TCHAN_REPEAT)
static inline void tchan_delay(uint32_t n) { for (volatile uint32_t i = 0; i < n; i++) { } }
static inline void tchan_emit(uint32_t sym) { tchan_delay((sym + 1) * TCHAN_UNIT); }
static inline int tchan_preamble(volatile uint32_t *counter, uint32_t *idx) {
    uint32_t c = (*counter)++;
    if (c == 0) { tchan_emit(0); return 1; }
    if (c < TCHAN_PREAMBLE) { tchan_emit((c - 1) / TCHAN_REPEAT); return 1; }
    *idx = (c - TCHAN_PREAMBLE) / TCHAN_REPEAT;
    return 0;
}
static inline void tchan_word_nibble(uint32_t v, uint32_t nib) { tchan_emit((v >> (4 * (7 - nib))) & 15); }
/* emit word i/8 of results[] */
static inline void tchan_emit_words(const volatile uint32_t *results, uint32_t n, uint32_t i) {
    uint32_t w = i >> 3;
    tchan_word_nibble(w < n ? results[w] : 0xFFFFFFFFu, i & 7);
}
