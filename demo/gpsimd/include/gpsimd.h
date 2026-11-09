/* gpsimd.h - device-side helpers for trn3 GPSIMD kernel libraries.
 *
 * SBUF access requires the host to map SBUF into the cores' address space before execution
 * (runner/run_neff.cpp does this after nrt_load): translation entry 38 makes
 *     core address SBUF_WINDOW + p * SBUF_PARTITION_STRIDE + byte_offset  ==  SBUF partition p
 * using the same partition/offset addressing as NKI/NEFF (tile at SBUF offset X -> byte X).
 * All 8 GPSIMD cores run every kernel; core k conventionally owns partitions 16k..16k+15.
 */
#pragma once
#include <stdint.h>

#define GPSIMD_NUM_CORES 8
#define SBUF_PARTITIONS 128
#define SBUF_PARTITIONS_PER_CORE (SBUF_PARTITIONS / GPSIMD_NUM_CORES)
#define SBUF_WINDOW 0xD0000000u
#define SBUF_PARTITION_STRIDE 0x40000u
#define HBM_WINDOW 0x80000000u       /* 16 MB onto this NC's HBM area (runtime-programmed) */

static inline uint32_t gpsimd_core_id(void) { uint32_t p; __asm__ volatile("rsr.prid %0" : "=r"(p)); return p & 0xff; }
static inline volatile void *sbuf_ptr(uint32_t partition, uint32_t offset) {
    return (volatile void *)(SBUF_WINDOW + partition * SBUF_PARTITION_STRIDE + offset);
}
static inline volatile int32_t *sbuf_i32(uint32_t partition, uint32_t offset) { return (volatile int32_t *)sbuf_ptr(partition, offset); }
static inline volatile uint32_t *sbuf_u32(uint32_t partition, uint32_t offset) { return (volatile uint32_t *)sbuf_ptr(partition, offset); }
