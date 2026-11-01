// sbuf_add.cpp - GPSIMD custom operator for trn3 (built with the Xtensa toolchain, xt-clang++).
//
// Operation (in place, in SBUF):   x[p][i] = x[p][i] + ADDEND      (int32, wrap-around)
// for every partition p of the tile and every element i. All 8 GPSIMD cores execute the kernel;
// core k handles partitions [k * P/8, (k + 1) * P/8). Each core logs what it did with kprintf; the
// host runner prints the logs.
//
// Build-time parameters (run_demo.sh extracts the tile location from the compiled NEFF):
//   TILE_SBUF_OFFSET   byte offset of the tile inside each SBUF partition
//   TILE_PARTITIONS    number of partitions (rows) of the tile
//   TILE_ELEMS         int32 elements per partition
//   ADDEND             the constant to add
//
// Constraints of the trn3 GPSIMD target (see README): plain C/C++ code only (no FLIX/vector TIE
// instructions, no floating point yet), no C++ runtime (exceptions, RTTI, static constructors,
// heap), SBUF reached through the window the host maps at 0xD0000000 (gpsimd.h).
#include <stdint.h>

#include "gpsimd.h"
#include "kprint.h"
#include "neuron_kl.h"

#ifndef TILE_SBUF_OFFSET
#define TILE_SBUF_OFFSET 0x3FE00u
#endif
#ifndef TILE_PARTITIONS
#define TILE_PARTITIONS 128u
#endif
#ifndef TILE_ELEMS
#define TILE_ELEMS 64u
#endif
#ifndef ADDEND
#define ADDEND 1000
#endif

namespace {

struct Tile {
    uint32_t sbuf_offset;
    uint32_t partitions;
    uint32_t elems;

    volatile uint32_t *row(uint32_t p) const { return sbuf_u32(p, sbuf_offset); }
};

constexpr Tile kTile{TILE_SBUF_OFFSET, TILE_PARTITIONS, TILE_ELEMS};
constexpr uint32_t kAddend = static_cast<uint32_t>(ADDEND);

// [first, last) partition range owned by `core`
struct Range {
    uint32_t first, last;
};

Range partitions_of(uint32_t core, uint32_t partitions) {
    const uint32_t per_core = (partitions + GPSIMD_NUM_CORES - 1) / GPSIMD_NUM_CORES;
    uint32_t first = core * per_core;
    uint32_t last = first + per_core;
    if (first > partitions) first = partitions;
    if (last > partitions) last = partitions;
    return {first, last};
}

// Adds `addend` to one partition row; returns the sums of the values read and written (mod 2^32).
struct Sums {
    uint32_t in, out;
};
Sums add_constant(const Tile &t, uint32_t p, uint32_t addend) {
    volatile uint32_t *x = t.row(p);
    Sums s{0, 0};
    for (uint32_t i = 0; i < t.elems; ++i) {
        const uint32_t v = x[i];
        const uint32_t r = v + addend;   // uint32: defined wrap-around, bit-identical to int32 add
        x[i] = r;
        s.in += v;
        s.out += r;
    }
    return s;
}

inline uint32_t read_ccount() { uint32_t c; __asm__ volatile("rsr.ccount %0" : "=r"(c)); return c; }
inline uint32_t read_sp() { uint32_t s; __asm__ volatile("mov %0, a1" : "=r"(s)); return s; }

}  // namespace

// Execution receipt, kept in this core's own data RAM (.bss) and read by the host over PCIe BAR0
// (runner/run_neff.cpp). It proves the operator ran on the GPSIMD cores: the processor id is read
// from the core's PRID register, the checksums can only be produced by reading/writing the SBUF
// tile, and the code/stack addresses lie in GPSIMD instruction/data RAM.
struct Receipt {
    uint32_t magic0, magic1;          // 0x4D495350, 0x43524444 ("PSIM", "DDRC")
    uint32_t prid;                    // hardware processor id (rsr.prid)
    uint32_t executions;              // incremented on every KERNEL_CALL
    uint32_t first_partition, end_partition, elems_per_partition;
    uint32_t input_checksum, output_checksum;   // sum of uint32 values read / written (mod 2^32)
    uint32_t ccount_start, ccount_end;          // core cycle counter around the work
    uint32_t kernel_pc;               // address of sbuf_add in GPSIMD instruction RAM
    uint32_t stack_pointer;           // a1 inside the kernel (GPSIMD data RAM)
    uint32_t return_address;          // a0: firmware dispatcher that called us
    uint32_t addend;
    uint32_t reserved;
};
static_assert(sizeof(Receipt) == 64, "receipt layout is shared with the host runner");
volatile Receipt g_receipt;

extern "C" void sbuf_add() {
    uint32_t ra;
    __asm__ volatile("mov %0, a0" : "=r"(ra));
    const uint32_t t0 = read_ccount();
    const uint32_t core = gpsimd_core_id();
    const Range r = partitions_of(core, kTile.partitions);

    const int32_t first_before = r.first < r.last ? static_cast<int32_t>(kTile.row(r.first)[0]) : 0;
    const int32_t last_before = r.first < r.last ? static_cast<int32_t>(kTile.row(r.last - 1)[kTile.elems - 1]) : 0;

    Sums total{0, 0};
    for (uint32_t p = r.first; p < r.last; ++p) {
        const Sums s = add_constant(kTile, p, kAddend);
        total.in += s.in;
        total.out += s.out;
    }
    const uint32_t t1 = read_ccount();

    g_receipt.prid = core;
    g_receipt.executions = g_receipt.executions + 1;
    g_receipt.first_partition = r.first;
    g_receipt.end_partition = r.last;
    g_receipt.elems_per_partition = kTile.elems;
    g_receipt.input_checksum = total.in;
    g_receipt.output_checksum = total.out;
    g_receipt.ccount_start = t0;
    g_receipt.ccount_end = t1;
    g_receipt.kernel_pc = reinterpret_cast<uint32_t>(&sbuf_add);
    g_receipt.stack_pointer = read_sp();
    g_receipt.return_address = ra;
    g_receipt.addend = kAddend;
    g_receipt.magic1 = 0x43524444u;   // "DDRC"
    g_receipt.magic0 = 0x4D495350u;   // "PSIM"  (magic written last)

    if (r.first == r.last) return;
    kprintf("sbuf_add core %u: partitions %u..%u (+%d): x[%u][0] %d -> %d, x[%u][%u] %d -> %d\n",
            core, r.first, r.last - 1, ADDEND,
            r.first, first_before, static_cast<int32_t>(kTile.row(r.first)[0]),
            r.last - 1, kTile.elems - 1, last_before,
            static_cast<int32_t>(kTile.row(r.last - 1)[kTile.elems - 1]));
}

// Register the kernel under the name used in the NEFF's gpsimd_kernels_def ("sbuf_add").
NEURON_KERNEL(sbuf_add);

// Called once by the GPSIMD firmware when the kernel library is loaded (argument: firmware context).
extern "C" void *_start(void *) { return nullptr; }
