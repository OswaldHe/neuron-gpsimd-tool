"""verify the ALU macros of toolchain/include/gpsimd/vec.h on the device: random vd/va/vb in v0..v15
(observable registers), vector mode, compare with numpy on the vpat inputs."""
import re, sys, random, collections
import numpy as np
sys.path.insert(0, '.')
import vrun
from vrun import vpat
H = open('../toolchain/include/gpsimd/vec.h').read()
alu = re.findall(r'#define (GPV_\w+)\((\w+), (\w+), (\w+)\)\s+GPV__ALU\(([^()]*)\)', H)
alu.append(('GPV_SRLI16', 'd', 'b', 'imm', 'd, imm, b, 2, 1, 0xac'))
def enc(d, a, b, op3, lo, op):
    return bytes([0x2f, 0x45, ((a & 31) << 1) | ((d & 1) << 6) | (((d >> 1) & 1) << 7),
                  1 | (((d >> 2) & 7) << 2) | (op3 << 5), ((b & 31) << 3) | lo, op, 0x50, 0x32]).hex()
V = np.array([[vpat(k, i) for i in range(16)] for k in range(16)], dtype=np.uint32).view(np.uint8).reshape(16, 64)
def L(x, w, s=False): return x.view({8: np.int8, 16: np.int16, 32: np.int32}[w] if s else {8: np.uint8, 16: np.uint16, 32: np.uint32}[w])
def ref(name, A, B, imm):
    w = int(re.search(r'(\d+)$', name).group(1)) if re.search(r'\d+$', name) else 8
    U = lambda x: L(x, w); S = lambda x: L(x, w, True)
    f = {'ADD': lambda: U(A) + U(B), 'SUB': lambda: U(A) - U(B), 'AND': lambda: U(A) & U(B), 'OR': lambda: U(A) | U(B),
         'XOR': lambda: U(A) ^ U(B), 'MAXS': lambda: np.maximum(S(A), S(B)), 'MAXU': lambda: np.maximum(U(A), U(B)),
         'MINS': lambda: np.minimum(S(A), S(B)), 'MINU': lambda: np.minimum(U(A), U(B)),
         'AVGU': lambda: ((U(A).astype(np.uint64) + U(B)) >> 1), 'AVGRU': lambda: ((U(A).astype(np.uint64) + U(B) + 1) >> 1),
         'ABSDU': lambda: np.maximum(U(A), U(B)) - np.minimum(U(A), U(B)),
         'SRLI': lambda: U(B) >> imm, 'SRAI': lambda: S(B) >> imm, 'SLLI': lambda: (U(B).astype(np.uint64) << imm)}
    base = re.match(r'GPV_([A-Z]+?)(\d+)?$', name).group(1)
    r = f[base]()
    return np.asarray(r).astype({8: np.uint8, 16: np.uint16, 32: np.uint32}[w] if base not in ('MAXS', 'MINS', 'SRAI') else {8: np.int8, 16: np.int16, 32: np.int32}[w]).view(np.uint8)
random.seed(5)
tests, meta = [], []
for name, p1, p2, p3, args in alu:
    a_ = [x.strip() for x in args.split(',')]
    op3, lo, op = int(a_[3]), int(a_[4]), int(a_[5], 16)
    shift = p3 == 'imm'
    w = int(re.search(r'(\d+)$', name).group(1)) if re.search(r'\d+$', name) else 8
    for rep in range(4):
        d, x, y = random.randrange(16), random.randrange(16), random.randrange(16)
        if shift:
            imms = list(range(w)) if name == 'GPV_SRLI16' and rep == 0 else ([0, 1, w // 2 - 1, w - 1][rep:rep + 1] if name != 'GPV_SRLI16' else [])
            for imm in imms:
                fld = ((imm & 3) | (((imm >> 2) & 3) << 3)) if name == 'GPV_SRLI16' else imm
                tests.append((enc(d, fld, x, op3, lo, op), {}, None, 1)); meta.append((name, d, None, x, imm))
        else:
            tests.append((enc(d, x, y, op3, lo, op), {}, None, 1)); meta.append((name, d, x, y, None))
res = vrun.run(tests, timeout=900)
ok = collections.Counter(); bad = []
for (name, d, x, y, imm), r in zip(meta, res):
    want = ref(name, V[x] if x is not None else None, V[y], imm)
    if r['status'] != 1: bad.append((name, d, x, y, imm, f"status {r['status']}")); continue
    exp_mask = 0 if np.array_equal(want, V[d]) else (1 << d)
    got = None
    if r['vmask'] & (1 << d):
        n = [k for k in range(16) if r['vmask'] >> k & 1].index(d)
        got = np.array(r['mem'][16 * n:16 * n + 16], dtype=np.uint32).view(np.uint8) if n < 2 else None
    else:
        got = V[d]
    if r['vmask'] & ~(1 << d): bad.append((name, d, x, y, imm, f"other regs changed {r['vmask']:#x}")); continue
    if got is not None and np.array_equal(got, want): ok[name] += 1
    else: bad.append((name, d, x, y, imm, 'mismatch'))
print('verified (4 random cases each):', len(ok), 'macros;', 'all ok' if not bad else f'{len(bad)} failures')
for b in bad: print('  FAIL', b)
print(dict(ok))
