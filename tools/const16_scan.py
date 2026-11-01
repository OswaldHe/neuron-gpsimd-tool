"""Find CONST16 pairs (hi then lo into the same register) in raw Xtensa code and histogram the
32-bit constants. Works on stripped code with undecodable FLIX bundles (pattern match on bytes).
usage: const16_scan.py code.bin [base_addr] [--all]"""
import collections, sys
b = open(sys.argv[1], "rb").read()
base = int(sys.argv[2], 0) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else 0
def c16(i):  # CONST16 at, imm16: byte0 = (t<<4)|4, imm16 = byte1 | byte2<<8
    if i + 3 > len(b) or (b[i] & 0xF) != 4: return None
    return b[i] >> 4, b[i + 1] | (b[i + 2] << 8)
cnt, where = collections.Counter(), collections.defaultdict(list)
for i in range(len(b) - 6):
    a = c16(i)
    if not a: continue
    for gap in (3, 5, 6):  # second half directly after, or after a narrow/24-bit instruction
        c = c16(i + gap)
        if c and c[0] == a[0]:
            v = (a[1] << 16) | c[1]
            cnt[v] += 1; where[v].append(base + i)
            break
for v, n in sorted(cnt.items(), key=lambda x: -x[1]):
    if n >= 2 or "--all" in sys.argv:
        print(f"0x{v:08x} x{n:3d}  first at 0x{where[v][0]:05x}")
