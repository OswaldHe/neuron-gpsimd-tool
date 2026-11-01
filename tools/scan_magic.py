"""Scan TPB regions over BAR0 for 32-bit magic values. usage: scan_magic.py TPB hexmagic[,hexmagic...]"""
import ctypes as C, fcntl, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from devmem import Dev, BarRw, IOCTL_BAR_READ, TPB_BAR0_OFFSET, TPB_BAR0_DIST  # noqa: E402

REGIONS = [(0x2400000, 0x2C00000), (0x3000000, 0x3100000)] + \
          [(0x3100000 + k * 0x100000 + a, 0x3100000 + k * 0x100000 + b) for k in range(8)
           for a, b in ((0x0, 0x40000), (0x80000, 0xC0000))]


def scan(d, tpb, magics, regions=REGIONS):
    hits = []
    for lo, hi in regions:
        w = d.tpb_read(tpb, lo, (hi - lo) // 4)
        for i, v in enumerate(w):
            if v in magics:
                hits.append((lo + 4 * i, v))
    return hits


if __name__ == "__main__":
    tpb = int(sys.argv[1]); magics = {int(x, 16) for x in sys.argv[2].split(",")}
    d = Dev(0)
    for off, v in scan(d, tpb, magics):
        ctx = d.tpb_read(tpb, max(0, off - 16), 16)
        print(f"TPB{tpb}+0x{off:07x}: {v:08x}   context: " + " ".join(f"{x:08x}" for x in ctx))
