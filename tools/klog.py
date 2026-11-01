"""Read kprint.h log buffers from every GPSIMD core's data RAM over BAR0 and print them.

usage: klog.py [--dev N] [--tpb T[,T...]] [--clear]
  default: scan all 8 TPBs of device N; --clear zeroes the magic so the next run starts fresh.
GPSIMD core k of TPB t: local address space at TPB window + 0x3100000 + k*0x100000
(data RAM = local 0x80000-0xBFFFF).
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from devmem import Dev  # noqa: E402

Q7_BASE = 0x3100000
Q7_STRIDE = 0x100000
DRAM_LO, DRAM_HI = 0x80000, 0xC0000
MAGIC = struct.pack("<II", 0x4B4C4F47, 0x31534D47)


def find_logs(d, tpb, core):
    base = Q7_BASE + core * Q7_STRIDE
    dram = d.tpb_bytes(tpb, base + DRAM_LO, DRAM_HI - DRAM_LO)
    out, i = [], dram.find(MAGIC)
    while i >= 0:
        if i % 4 == 0:
            cap, pos, prid, calls = struct.unpack_from("<IIII", dram, i + 8)
            if 0 < cap <= 0x10000:
                text = dram[i + 32:i + 32 + cap]
                if pos <= cap:
                    s = text[:pos]
                else:   # ring wrapped: oldest byte at pos % cap
                    k = pos % cap
                    s = text[k:] + text[:k]
                out.append(dict(addr=DRAM_LO + i, cap=cap, pos=pos, prid=prid, calls=calls, text=s, tpb=tpb,
                                core=core, bar_off=base + DRAM_LO + i))
        i = dram.find(MAGIC, i + 4)
    return out


def dump(dev=0, tpbs=range(8), clear=False, stream=sys.stdout):
    d = Dev(dev)
    found = []
    for t in tpbs:
        for c in range(8):
            for lg in find_logs(d, t, c):
                found.append(lg)
                hdr = (f"--- TPB{t} GPSIMD core{c} (prid {lg['prid']}) log @0x{lg['addr']:05x}: "
                       f"{lg['calls']} kprintf calls, {lg['pos']} bytes{' (wrapped)' if lg['pos'] > lg['cap'] else ''}")
                print(hdr, file=stream)
                print(lg["text"].decode("ascii", "replace").rstrip("\n"), file=stream)
                if clear:
                    d.tpb_write(t, lg["bar_off"], [0, 0])
    if not found:
        print("(no kprint log buffers found)", file=stream)
    return found


def main():
    a = sys.argv[1:]
    dev, tpbs, clear = 0, range(8), False
    while a:
        if a[0] == "--dev":
            dev, a = int(a[1]), a[2:]
        elif a[0] == "--tpb":
            tpbs, a = [int(x) for x in a[1].split(",")], a[2:]
        elif a[0] == "--clear":
            clear, a = True, a[1:]
        else:
            raise SystemExit(__doc__)
    dump(dev, tpbs, clear)


if __name__ == "__main__":
    main()
