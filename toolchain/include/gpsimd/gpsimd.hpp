// gpsimd.hpp - C++ device API for trn3 GPSIMD custom operators (wraps gpsimd.h).
//
//     #include <gpsimd/gpsimd.hpp>
//     GPSIMD_OP(scale) {
//         gp::Call c(call);
//         auto x = c.in<gp::bf16>(0);
//         auto y = c.out<gp::f32>(0);
//         for (uint32_t p : c.partitions(x))                // this core's share of the partitions
//             for (uint32_t i = 0; i < x.elems(); ++i)
//                 y.store(p, i, gp::f32(x.load(p, i)) * GP_F32(0.5));
//     }
//
// Floating point: the trn3 GPSIMD cores cannot execute the toolchain's FPU/vector instructions, so
// `float` must not be used. gp::f32 is an IEEE binary32 value with soft-float arithmetic (round to
// nearest even, results bit-identical to x86 SSE / numpy float32); GP_F32(1.5) is a compile-time
// float32 constant; gp::bf16 / gp::f16 are storage types that convert exactly to gp::f32.
#pragma once
#include <stddef.h>
#include <stdint.h>

#include <gpsimd/gpsimd.h>

namespace gp {

// ------------------------------------------------------------------ compile-time float constants
// double -> float32 bits with round-to-nearest-even, evaluated by the compiler (constexpr only).
constexpr uint32_t f32_bits(double x) {
    if (x != x) return 0x7FC00000u;
    uint32_t sign = 0;
    if (x < 0 || (x == 0 && 1.0 / x < 0)) { sign = 0x80000000u; x = -x; }
    if (x == 0) return sign;
    if (x > 3.4028235677973366e38) return sign | 0x7F800000u;
    int e = 0;
    while (x >= 2.0) { x /= 2.0; ++e; }
    while (x < 1.0) { x *= 2.0; --e; }
    // x in [1, 2): value = x * 2^e
    double scaled = x;
    int k = e < -126 ? e + 149 : 23;                             // subnormal: units of 2^-149
    for (; k > 0; --k) scaled *= 2.0;
    for (; k < 0; ++k) scaled /= 2.0;
    uint64_t m = static_cast<uint64_t>(scaled);
    double frac = scaled - static_cast<double>(m);
    if (frac > 0.5 || (frac == 0.5 && (m & 1))) ++m;
    if (e < -126) return sign | static_cast<uint32_t>(m);        // subnormal (2^23 rounds up to the min normal)
    if (m == (1ull << 24)) { m >>= 1; ++e; }
    if (e > 127) return sign | 0x7F800000u;
    return sign | (static_cast<uint32_t>(e + 127) << 23) | (static_cast<uint32_t>(m) & 0x7FFFFFu);
}
template <uint32_t V> struct u32_const { static constexpr uint32_t value = V; };

// ------------------------------------------------------------------ soft-float types
struct f32 {
    uint32_t bits;
    f32() : bits(0) {}
    static f32 from_bits(uint32_t b) { f32 r; r.bits = b; return r; }
    explicit f32(int32_t v) : bits(sf_from_i32(v)) {}
    explicit f32(uint32_t v) : bits(sf_from_u32(v)) {}
    static f32 from_int(int32_t v) { return f32(v); }

    f32 operator+(f32 o) const { return from_bits(sf_add(bits, o.bits)); }
    f32 operator-(f32 o) const { return from_bits(sf_sub(bits, o.bits)); }
    f32 operator*(f32 o) const { return from_bits(sf_mul(bits, o.bits)); }
    f32 operator/(f32 o) const { return from_bits(sf_div(bits, o.bits)); }
    f32 operator-() const { return from_bits(bits ^ 0x80000000u); }
    f32 &operator+=(f32 o) { return *this = *this + o; }
    f32 &operator-=(f32 o) { return *this = *this - o; }
    f32 &operator*=(f32 o) { return *this = *this * o; }
    f32 &operator/=(f32 o) { return *this = *this / o; }
    bool operator<(f32 o) const { return sf_lt(bits, o.bits); }
    bool operator<=(f32 o) const { return sf_le(bits, o.bits); }
    bool operator>(f32 o) const { return sf_lt(o.bits, bits); }
    bool operator>=(f32 o) const { return sf_le(o.bits, bits); }
    bool operator==(f32 o) const { return sf_eq(bits, o.bits); }
    bool operator!=(f32 o) const { return !sf_eq(bits, o.bits); }
    bool is_nan() const { return (bits & 0x7FFFFFFFu) > 0x7F800000u; }
    int32_t to_i32() const { return sf_to_i32(bits); }        // truncate toward zero (x86 cvttss2si)
    uint32_t to_u32() const { return sf_to_u32(bits); }
};
inline f32 fma(f32 a, f32 b, f32 c) { return f32::from_bits(sf_fma(a.bits, b.bits, c.bits)); }
inline f32 sqrt(f32 a) { return f32::from_bits(sf_sqrt(a.bits)); }
inline f32 min(f32 a, f32 b) { return f32::from_bits(sf_min(a.bits, b.bits)); }   // x86 minss: a < b ? a : b
inline f32 max(f32 a, f32 b) { return f32::from_bits(sf_max(a.bits, b.bits)); }   // x86 maxss: b < a ? a : b
inline f32 abs(f32 a) { return f32::from_bits(a.bits & 0x7FFFFFFFu); }
inline f32 trunc(f32 a) { return f32::from_bits(sf_trunc(a.bits)); }
inline f32 floor(f32 a) { return f32::from_bits(sf_floor(a.bits)); }
inline f32 ceil(f32 a) { return f32::from_bits(sf_ceil(a.bits)); }
inline f32 rint(f32 a) { return f32::from_bits(sf_rint(a.bits)); }     // nearest, ties to even
inline f32 round(f32 a) { return f32::from_bits(sf_round(a.bits)); }   // nearest, ties away from zero
// a * 2^n (exact unless the result over/underflows; computed with a power-of-two multiply)
inline f32 ldexp(f32 a, int n) {
    f32 r = a;
    while (n > 127) { r = r * f32::from_bits(0x7F000000u); n -= 127; }
    while (n < -126) { r = r * f32::from_bits(0x00800000u); n += 126; }
    return r * f32::from_bits(static_cast<uint32_t>(n + 127) << 23);
}

struct bf16 {
    uint16_t bits;
    bf16() : bits(0) {}
    static bf16 from_bits(uint16_t b) { bf16 r; r.bits = b; return r; }
    bf16(f32 v) : bits(sf_to_bf16(v.bits)) {}                   // round to nearest even
    operator f32() const { return f32::from_bits(sf_from_bf16(bits)); }
};
struct f16 {
    uint16_t bits;
    f16() : bits(0) {}
    static f16 from_bits(uint16_t b) { f16 r; r.bits = b; return r; }
    f16(f32 v) : bits(sf_to_f16(v.bits)) {}                     // round to nearest even
    operator f32() const { return f32::from_bits(sf_from_f16(bits)); }
};

// ------------------------------------------------------------------ element traits
template <typename T> struct elem;
#define GP_ELEM(T, code)                                                     \
    template <> struct elem<T> {                                             \
        static constexpr uint8_t dtype = code;                               \
        using storage = T;                                                   \
        static T from_bits(uint32_t b) { return static_cast<T>(b); }         \
        static uint32_t to_bits(T v) { return static_cast<uint32_t>(v); }    \
    };
GP_ELEM(int8_t, GP_INT8)
GP_ELEM(uint8_t, GP_UINT8)
GP_ELEM(int16_t, GP_INT16)
GP_ELEM(uint16_t, GP_UINT16)
GP_ELEM(int32_t, GP_INT32)
GP_ELEM(uint32_t, GP_UINT32)
#undef GP_ELEM
template <> struct elem<f32> {
    static constexpr uint8_t dtype = GP_FP32;
    using storage = uint32_t;
    static f32 from_bits(uint32_t b) { return f32::from_bits(b); }
    static uint32_t to_bits(f32 v) { return v.bits; }
};
template <> struct elem<bf16> {
    static constexpr uint8_t dtype = GP_BF16;
    using storage = uint16_t;
    static bf16 from_bits(uint32_t b) { return bf16::from_bits(static_cast<uint16_t>(b)); }
    static uint32_t to_bits(bf16 v) { return v.bits; }
};
template <> struct elem<f16> {
    static constexpr uint8_t dtype = GP_FP16;
    using storage = uint16_t;
    static f16 from_bits(uint32_t b) { return f16::from_bits(static_cast<uint16_t>(b)); }
    static uint32_t to_bits(f16 v) { return v.bits; }
};

// ------------------------------------------------------------------ tiles
// Typed view of a tile. T must have the element size of the tile's dtype (e.g. uint32_t views a float32
// tile's bits); Call::in/out set call->status bit 1 on a size mismatch. AnyTile: dtype-generic code.
template <typename T> class Tile {
   public:
    using S = typename elem<T>::storage;
    explicit Tile(const gp_tile *t) : t_(t) {}
    const gp_tile *desc() const { return t_; }
    uint32_t partitions() const { return t_->nparts; }
    uint32_t elems() const { return t_->elems; }
    bool contiguous() const { return t_->contiguous; }
    // pointer to partition p's first element; element i at row(p)[i] when contiguous()
    volatile S *row(uint32_t p) const { return reinterpret_cast<volatile S *>(gp_row_addr(t_, p)); }
    volatile S *ptr(uint32_t p, uint32_t i) const { return row(p) + gp_elem_offset(t_, i); }
    T load(uint32_t p, uint32_t i) const { return elem<T>::from_bits(*ptr(p, i)); }
    void store(uint32_t p, uint32_t i, T v) const { *ptr(p, i) = static_cast<S>(elem<T>::to_bits(v)); }
    // element offset of multi-dimensional index (free dims in NKI order, outermost first)
    int32_t offset(uint32_t i0) const { return gp_elem_offset(t_, i0); }
    // elements of storage one partition of the view covers (dense tiles: elems())
    uint32_t span() const {
        uint32_t m = 0;
        for (uint32_t d = 0; d < t_->ndims; ++d) m += (t_->num[d] - 1u) * static_cast<uint32_t>(t_->step[d]);
        return m + 1u;
    }

    // Bulk staging through local data RAM with the iDMA (~100x faster than element access to SBUF):
    // partitions [p, p + n) of the view <-> `local`, a dense [n][span()] array. Element i of staged row r
    // is local[r * span() + offset(i)]  (contiguous views: local[r * elems() + i]). Asynchronous: returns a
    // ticket for Call::dma_wait. flush() writes back the whole span, so elements between strided view
    // elements are rewritten with the values fetched before.
    uint32_t fetch(gp_call *c, uint32_t p, uint32_t n, S *local) const {
        return gp_dma_2d(c, local, row(p), span() * sizeof(S), n, GP_PARTITION_STRIDE, span() * sizeof(S));
    }
    uint32_t flush(gp_call *c, uint32_t p, uint32_t n, const S *local) const {
        return gp_dma_2d(c, row(p), local, span() * sizeof(S), n, span() * sizeof(S), GP_PARTITION_STRIDE);
    }

   private:
    const gp_tile *t_;
};

// dtype-generic view: every element read/written as float32 (exact for all supported dtypes on load).
class AnyTile {
   public:
    explicit AnyTile(const gp_tile *t) : t_(t) {}
    const gp_tile *desc() const { return t_; }
    uint32_t partitions() const { return t_->nparts; }
    uint32_t elems() const { return t_->elems; }
    uint8_t dtype() const { return t_->dtype; }
    f32 load(uint32_t p, uint32_t i) const { return f32::from_bits(gp_ld_f32(t_, p, i)); }
    void store(uint32_t p, uint32_t i, f32 v) const { gp_st_f32(t_, p, i, v.bits); }
    uint32_t load_bits(uint32_t p, uint32_t i) const { return gp_ld_bits(t_, p, i); }
    void store_bits(uint32_t p, uint32_t i, uint32_t b) const { gp_st_bits(t_, p, i, b); }

   private:
    const gp_tile *t_;
};

// HBM access: gpsimd-run maps the device memory of the tensors named with --hbm (examples: HBM = [...])
// into the cores' address space and writes their core addresses into an int32 [1, 16] input tile
// (element k = k-th tensor, element 15 = 0x48424D31). Pass that tile to gp.call and use:
template <typename T> volatile T *hbm_ptr(const Tile<int32_t> &params, uint32_t k) {
    return reinterpret_cast<volatile T *>(static_cast<uintptr_t>(static_cast<uint32_t>(params.load(0, k))));
}
inline bool hbm_mapped(const Tile<int32_t> &params) { return static_cast<uint32_t>(params.load(0, 15)) == 0x48424D31u; }

// [lo, hi) range usable in range-for
struct Range {
    uint32_t lo, hi;
    struct It {
        uint32_t v;
        uint32_t operator*() const { return v; }
        It &operator++() { ++v; return *this; }
        bool operator!=(const It &o) const { return v != o.v; }
    };
    It begin() const { return {lo}; }
    It end() const { return {hi}; }
    uint32_t size() const { return hi - lo; }
};

// Blocks of at most `step` partitions covering [lo, hi): for (gp::Range b : c.blocks(t.partitions(), 8)) ...
struct Blocks {
    uint32_t lo, hi, step;
    struct It {
        uint32_t p, hi, step;
        Range operator*() const { return {p, p + step < hi ? p + step : hi}; }
        It &operator++() { p = p + step < hi ? p + step : hi; return *this; }
        bool operator!=(const It &o) const { return p != o.p; }
    };
    It begin() const { return {lo, hi, step}; }
    It end() const { return {hi, hi, step}; }
};

class Call {
   public:
    explicit Call(gp_call *c) : c_(c) {}
    gp_call *raw() const { return c_; }
    uint32_t core() const { return c_->core; }
    uint32_t ncores() const { return c_->ncores; }
    uint32_t exec() const { return c_->exec; }
    uint32_t n_in() const { return c_->site->n_in; }
    uint32_t n_out() const { return c_->site->n_out; }
    template <typename T> Tile<T> in(uint32_t k) const { return checked<T>(gp_in(c_, k)); }
    template <typename T> Tile<T> out(uint32_t k) const { return checked<T>(gp_out(c_, k)); }
    AnyTile in_any(uint32_t k) const { return AnyTile(gp_in(c_, k)); }
    AnyTile out_any(uint32_t k) const { return AnyTile(gp_out(c_, k)); }
    // this core's block of [0, n) / of a tile's partitions
    Range split(uint32_t n) const { gp_range r = gp_split(c_, n); return {r.lo, r.hi}; }
    Range split_aligned(uint32_t n, uint32_t align) const { gp_range r = gp_split_aligned(c_, n, align); return {r.lo, r.hi}; }
    template <typename Tl> Range partitions(const Tl &t) const { return split(t.partitions()); }
    // this core's partitions of an n-partition tile in blocks of <= rows partitions (staging buffers)
    Blocks blocks(uint32_t n, uint32_t rows) const { Range r = split(n); return {r.lo, r.hi, rows ? rows : 1}; }
    void barrier() const { gp_barrier(c_); }
    uint32_t allreduce_add(uint32_t v) const { return gp_allreduce_add_u32(c_, v); }
    int32_t allreduce_max(int32_t v) const { return gp_allreduce_max_i32(c_, v); }
    int32_t allreduce_min(int32_t v) const { return gp_allreduce_min_i32(c_, v); }
    f32 allreduce_add(f32 v) const { return f32::from_bits(gp_allreduce_add_f32(c_, v.bits)); }
    f32 allreduce_max(f32 v) const { return f32::from_bits(gp_allreduce_max_f32(c_, v.bits)); }
    uint32_t exscan_add(uint32_t v, uint32_t *total = nullptr) const { return gp_exscan_add_u32(c_, v, total); }
    volatile uint32_t *shared(uint32_t *count = nullptr) const { return gp_shared_words(c_, count); }
    void lock() const { gp_lock(c_); }
    void unlock() const { gp_unlock(c_); }
    uint32_t atomic_add(volatile uint32_t *word, uint32_t v) const { return gp_atomic_add_u32(c_, word, v); }
    void fail(uint32_t bit) const { c_->status |= bit; }
    // iDMA (see gpsimd.h): asynchronous bulk copies between local RAM, SBUF and HBM
    uint32_t dma(volatile void *dst, const volatile void *src, uint32_t bytes) const { return gp_dma_copy(c_, dst, src, bytes); }
    uint32_t dma_2d(volatile void *dst, const volatile void *src, uint32_t row_bytes, uint32_t nrows, int32_t src_pitch,
                    int32_t dst_pitch) const {
        return gp_dma_2d(c_, dst, src, row_bytes, nrows, src_pitch, dst_pitch);
    }
    void dma_wait(uint32_t ticket) const { gp_dma_wait(c_, ticket); }
    void dma_wait_all() const { gp_dma_wait_all(c_); }
    template <typename T> uint32_t fetch(const Tile<T> &t, uint32_t p, uint32_t n, typename Tile<T>::S *local) const {
        return t.fetch(c_, p, n, local);
    }
    template <typename T> uint32_t flush(const Tile<T> &t, uint32_t p, uint32_t n, const typename Tile<T>::S *local) const {
        return t.flush(c_, p, n, local);
    }

   private:
    template <typename T> Tile<T> checked(const gp_tile *t) const {
        if (t->elem_bytes != sizeof(typename elem<T>::storage)) c_->status |= 2;   // same-size views allowed
        return Tile<T>(t);
    }
    gp_call *c_;
};

}  // namespace gp

// compile-time float32 constant: GP_F32(0.1) == nearest float32 to 0.1
#define GP_F32(x) (::gp::f32::from_bits(::gp::u32_const<::gp::f32_bits(x)>::value))
