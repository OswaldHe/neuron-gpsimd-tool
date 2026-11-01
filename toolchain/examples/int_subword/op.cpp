// int_subword: 8/16-bit SBUF loads and stores (l8ui, l16ui, l16si, s8i, s16i) and element-interleaved
// work distribution (core k handles elements i with i % 8 == k: the 8 cores store into the same words).
#include <gpsimd/gpsimd.hpp>

static inline int32_t clamp(int32_t v, int32_t lo, int32_t hi) { return v < lo ? lo : v > hi ? hi : v; }

GPSIMD_OP(int_subword) {
    gp::Call c(call);
    auto a8 = c.in<int8_t>(0), b8 = c.in<int8_t>(1);
    auto u8 = c.in<uint8_t>(2), v8 = c.in<uint8_t>(3);
    auto a16 = c.in<int16_t>(4), b16 = c.in<int16_t>(5);
    auto u16 = c.in<uint16_t>(6);
    auto o0 = c.out<int8_t>(0);
    auto o1 = c.out<uint8_t>(1);
    auto o2 = c.out<int16_t>(2);
    auto o3 = c.out<uint16_t>(3);
    auto o4 = c.out<int32_t>(4);
    auto o5 = c.out<int16_t>(5);
    const uint32_t P = o0.partitions(), F = o0.elems();

    // element-interleaved: every partition, elements i = core, core + 8, ...
    for (uint32_t p = 0; p < P; ++p) {
        for (uint32_t i = c.core(); i < F; i += c.ncores()) {
            o0.row(p)[i] = static_cast<int8_t>(clamp(a8.row(p)[i] + b8.row(p)[i], -128, 127));
            o1.row(p)[i] = static_cast<uint8_t>((u8.row(p)[i] + v8.row(p)[i] + 1u) >> 1);
            o2.row(p)[i] = static_cast<int16_t>(clamp((a16.row(p)[i] * b16.row(p)[i] + (1 << 14)) >> 15, -32768, 32767));
            o5.row(p)[i] = static_cast<int16_t>(a8.row(p)[i] * u8.row(p)[i]);
        }
    }
    // partition split
    for (uint32_t p : c.partitions(o3)) {
        for (uint32_t i = 0; i < F; ++i) {
            uint32_t x = u16.row(p)[i];
            x ^= x >> 3;
            o3.row(p)[i] = static_cast<uint16_t>(((x & 0xFF) << 8) | (x >> 8));
            o4.row(p)[i] = a8.row(p)[i] + u8.row(p)[i] + a16.row(p)[i] + u16.row(p)[i];
        }
    }
}
