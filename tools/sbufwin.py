"""Map a TPB's SBUF into the GPSIMD cores' address space (trn3 / mariana).

The GPSIMD (Pool engine) cores reach SoC memory through an address-translation block at
TPB window + 0x3062000 (BAR0-visible). It holds 40 entries of 7 little-endian u32 words:
    {ctrl, mask_lo, mask_hi, local_lo, local_hi, soc_lo, soc_hi}
ctrl bit0 = valid; a core address `a` hits the entry if (a & mask) == local; it is then translated
to soc + (a & ~mask). Entries 0-39 exist (reads past them return 0xbad0add1).
Runtime/firmware-programmed entries seen on TPB4 (device 0):
    0: 0x80000000 16 MB -> HBM 0x8008c0000000      4: 0x90000000 32 MB -> HBM 0x8008c0000000
    1: 0x84000000 32 MB -> 0x801002000000           6/7: 0x98000000/0x9c000000 -> 0x8050../0x8080..
    8,12,..,36: 0x2N000000 ctrl 0xNc, soc 0        9,13,..,37: 0x1N000000 ctrl 5 -> 0x0?00..0803000000
Free entries include 38 and 39; we use entry 38 at core address 0xD0000000 by default.

SBUF SoC bases (libnrt `mariana_sbuf_base_addresses`, also in GPSIMD firmware data RAM at 0x81f80):
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from devmem import Dev  # noqa: E402

XLAT_OFF = 0x3062000
ENTRY_WORDS = 7
NUM_ENTRIES = 40
SBUF_SOC_BASE = [0x2000000000, 0x3000000000, 0x6000000000, 0x7000000000,
                 0x802000000000, 0x803000000000, 0x806000000000, 0x807000000000]
SBUF_BYTES = 128 * 0x40000          # 128 partitions x 256 KB address stride = 32 MB
DEFAULT_ENTRY = 38
DEFAULT_LOCAL = 0xD0000000


def read_entries(d, tpb):
    w = d.tpb_read(tpb, XLAT_OFF, NUM_ENTRIES * ENTRY_WORDS)
    out = []
    for k in range(NUM_ENTRIES):
        e = w[ENTRY_WORDS * k:ENTRY_WORDS * (k + 1)]
        out.append(dict(idx=k, ctrl=e[0], mask=e[1] | e[2] << 32, local=e[3] | e[4] << 32, soc=e[5] | e[6] << 32))
    return out


def map_window(d, tpb, entry, local, soc, size):
    assert size & (size - 1) == 0 and local % size == 0 and soc % size == 0
    mask = (~(size - 1)) & 0xFFFFFFFFFF
    off = XLAT_OFF + 4 * ENTRY_WORDS * entry
    d.tpb_write(tpb, off, [0])                                        # invalidate first
    d.tpb_write(tpb, off + 4, [mask & 0xFFFFFFFF, mask >> 32, local & 0xFFFFFFFF, local >> 32,
                               soc & 0xFFFFFFFF, soc >> 32])
    d.tpb_write(tpb, off, [1])                                        # valid
    back = d.tpb_read(tpb, off, ENTRY_WORDS).tolist()
    return back


def map_sbuf(tpb=4, entry=DEFAULT_ENTRY, local=DEFAULT_LOCAL, dev=0, d=None):
    d = d or Dev(dev)
    cur = read_entries(d, tpb)[entry]
    if cur["ctrl"] & 1 and cur["soc"] != SBUF_SOC_BASE[tpb]:
        raise RuntimeError(f"translation entry {entry} on TPB{tpb} is in use: {cur}")
    return map_window(d, tpb, entry, local, SBUF_SOC_BASE[tpb], SBUF_BYTES)


def main():
    a = sys.argv[1:]
    tpb = int(a[1]) if len(a) > 1 else 4
    d = Dev(0)
    if a and a[0] == "show":
        for e in read_entries(d, tpb):
            if e["ctrl"] or e["soc"] or e["local"]:
                size = ((~e["mask"]) & 0xFFFFFFFFFF) + 1
                print(f"entry {e['idx']:2d}: ctrl {e['ctrl']:#04x} local 0x{e['local']:08x} {size >> 20:3d} MB -> soc 0x{e['soc']:016x}")
    elif a and a[0] == "map":
        print("entry now:", [hex(x) for x in map_sbuf(tpb, d=d)])
    else:
        print("usage: sbufwin.py show|map [tpb]")


if __name__ == "__main__":
    main()
