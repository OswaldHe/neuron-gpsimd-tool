"""Learn the trn3 FLIX bundle-length rule from firmware code: segments between aligned function
entries must tile exactly; bundles in uniquely-tiled segments become labelled samples; then search
for the bit(s) that separate 8- from 16-byte bundles. usage: learn_flix_len.py code.bin"""
import collections, itertools, sys
code = open(sys.argv[1], "rb").read()
BASE = {**{o: 3 for o in range(8)}, **{o: 2 for o in range(8, 14)}}
def is_entry(i):  # entry a1, N : byte0 0x36, byte1 low nibble 1
    return code[i] == 0x36 and code[i + 1] & 0xF == 0x1 and i % 4 == 0
entries = [i for i in range(0, len(code) - 3, 4) if is_entry(i)]
samples = []
uniq = amb = none = 0
for a, b in zip(entries, entries[1:]):
    if b - a > 600: continue
    # end: last instruction must be a return followed by zero padding to b
    ends = set()
    for e in range(b - 3, b + 1):
        if e < a: continue
        if all(x == 0 for x in code[e:b]):
            for rl in (2, 3):
                r = code[e - rl:e]
                if r in (b"\x1d\xf0", b"\x0d\xf0", b"\x90\x00\x00", b"\x80\x00\x00"):
                    ends.add(e)
    if not ends: continue
    # DP: ways[i] = list of bundle-length choices sequences (cap to detect uniqueness)
    ways = {a + 3: [()]}
    order = [a + 3]
    res = []
    stack = [(a + 3, ())]
    while stack and len(res) < 3:
        i, ch = stack.pop()
        if i in ends: res.append(ch); continue
        if i >= b: continue
        op0 = code[i] & 0xF
        if op0 in BASE: stack.append((i + BASE[op0], ch))
        else:
            for n in ((8, 16) if op0 == 0xF else (16, 8)):
                stack.append((i + n, ch + ((i, n),)))
    if len(res) == 1:
        uniq += 1; samples.extend(res[0])
    elif res: amb += 1
    else: none += 1
print(f"segments: unique {uniq}, ambiguous {amb}, impossible {none}; labelled bundles {len(samples)}")
lab = [(code[i:i + 8], n, code[i] & 0xF) for i, n in samples]
for op0 in (0xE, 0xF):
    L = [(bb, n) for bb, n, o in lab if o == op0]
    print(f"op0={op0:x}: {collections.Counter(n for _, n in L)}")
    best = []
    for byte in range(8):
        for bit in range(8):
            t = collections.Counter((bb[byte] >> bit & 1, n) for bb, n in L)
            err = min(t[(0, 8)] + t[(1, 16)], t[(0, 16)] + t[(1, 8)])
            best.append((err, byte, bit, dict(t)))
    best.sort()
    for e in best[:4]: print("   ", e)
    # pairs of bits (AND/OR/XOR) for the best few single bits
    cand = [(by, bi) for _, by, bi, _ in best[:12]]
    best2 = []
    for (b1, i1), (b2, i2) in itertools.combinations(cand, 2):
        for opn, fn in (("and", lambda x, y: x & y), ("or", lambda x, y: x | y), ("xor", lambda x, y: x ^ y)):
            t = collections.Counter((fn(bb[b1] >> i1 & 1, bb[b2] >> i2 & 1), n) for bb, n in L)
            err = min(t[(0, 8)] + t[(1, 16)], t[(0, 16)] + t[(1, 8)])
            best2.append((err, f"byte{b1}.bit{i1} {opn} byte{b2}.bit{i2}", dict(t)))
    best2.sort(key=lambda x: x[0])
    for e in best2[:3]: print("   2-bit:", e)
