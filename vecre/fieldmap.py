"""fieldmap: map the operand fields of a known vector instruction by flipping single bits (vector mode).
usage: fieldmap.py <hex> [first_bit last_bit]   prints, per flipped bit, the status and the inferred operation"""
import sys, json
import numpy as np
sys.path.insert(0, '.')
import vrun, infer
h = sys.argv[1]
b = bytearray(bytes.fromhex(h))
lo, hi = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (4, len(b) * 8 - 1)
skip = {0, 1, 2, 3, 24, 25}            # op0 and the length bits
tests, bits = [(h, {}, None, 1)], [None]
for k in range(lo, hi + 1):
    if k in skip: continue
    c = bytearray(b); c[k // 8] ^= 1 << (k % 8)
    tests.append((c.hex(), {}, None, 1)); bits.append(k)
res = vrun.run(tests, timeout=1800)
def desc(r):
    if r['status'] != 1: return {2: f"EXC{r['cause'] & 0xf}", 4: 'HANG', 5: 'CORRUPT', 6: 'BREAKS'}.get(r['status'], str(r['status']))
    if not r.get('vmask'): return f"no vreg change" + (f", mem {r['nchanged']}B" if r['nchanged'] else '')
    ks = [k for k in range(16) if r['vmask'] >> k & 1]
    out = []
    for n, k in enumerate(r['vregs'][:min(2, len(ks))]):
        R = np.array(r['mem'][16 * n:16 * n + 16], dtype=np.uint32).view(np.uint8)
        e = infer.explain(R, r['sbase'])
        out.append(f"v{k}: " + (' | '.join(e[:3]) if e else '?'))
    return '; '.join(out)
for k, r in zip(bits, res):
    print(f"{'base' if k is None else f'bit {k:3d} (byte {k // 8} bit {k % 8})':22} {desc(r)}")
