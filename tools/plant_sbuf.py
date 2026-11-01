"""Plant SBUF markers over BAR0: p0+0x20000 = 0x5B0F5B0F, p1+0x20000 = 0x5B0F0001 (TPB given as arg)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from devmem import Dev
tpb = int(sys.argv[1]) if len(sys.argv) > 1 else 4
d = Dev(0)
d.tpb_write(tpb, 0x20000, [0x5B0F5B0F]); d.tpb_write(tpb, 0x60000, [0x5B0F0001])
print("planted:", hex(d.tpb_read(tpb, 0x20000, 1)[0]), hex(d.tpb_read(tpb, 0x60000, 1)[0]))
