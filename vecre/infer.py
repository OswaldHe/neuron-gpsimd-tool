"""infer: explain vector-register results of vector-mode sandbox runs (sweepv.jsonl).

Each result reports up to two changed vector registers (index + 64-byte value). The inputs are known:
v0..v15 = vpat (see vrun.vpat), a3..a15 = scratch + 0x1000 + 0x100 k, a0 = scratch + 0x2800, a2 = scratch + 0x3000,
scratch word i = 0xA5000000 | i (bytes 0x000..0x3ff hold the vpat images, 0x400..0x7ff the dumps).
For a result R we try: loads from scratch, scalar broadcasts, lane-wise unary/binary ops on 8/16/32-bit lanes
(with every pair of source registers and every scalar register), shifts by immediates, and byte permutations.
usage: infer.py [sweepv.jsonl] [--hex <bundle>]
"""
import json, sys, collections
import numpy as np
sys.path.insert(0, '.')
from vrun import vpat

V = np.array([[vpat(k, i) for i in range(16)] for k in range(16)], dtype=np.uint32)   # 16 x 16 words
VB = V.view(np.uint8).reshape(16, 64)
W = {8: np.uint8, 16: np.uint16, 32: np.uint32}
S = {8: np.int8, 16: np.int16, 32: np.int32}


def lanes(x, w, signed=False):
    return x.view(S[w] if signed else W[w])


def scalars(sb):
    r = {f'a{k}': (sb + 0x1000 + 0x100 * k) & 0xFFFFFFFF for k in range(3, 16)}
    r['a0'] = (sb + 0x2800) & 0xFFFFFFFF
    r['a2'] = (sb + 0x3000) & 0xFFFFFFFF
    return r


def binops(w):
    m = (1 << w) - 1
    def u(x): return lanes(x, w)
    def s(x): return lanes(x, w, True)
    return {
        'add': lambda a, b: (u(a) + u(b)), 'sub': lambda a, b: (u(a) - u(b)),
        'and': lambda a, b: u(a) & u(b), 'or': lambda a, b: u(a) | u(b), 'xor': lambda a, b: u(a) ^ u(b),
        'andn': lambda a, b: u(a) & ~u(b), 'nand': lambda a, b: ~(u(a) & u(b)), 'nor': lambda a, b: ~(u(a) | u(b)),
        'xnor': lambda a, b: ~(u(a) ^ u(b)),
        'minu': lambda a, b: np.minimum(u(a), u(b)), 'maxu': lambda a, b: np.maximum(u(a), u(b)),
        'mins': lambda a, b: np.minimum(s(a), s(b)).view(W[w]), 'maxs': lambda a, b: np.maximum(s(a), s(b)).view(W[w]),
        'mul': lambda a, b: (u(a).astype(np.uint64) * u(b).astype(np.uint64) & m).astype(W[w]),
        'mulhu': lambda a, b: ((u(a).astype(np.uint64) * u(b).astype(np.uint64)) >> w).astype(W[w]),
        'mulhs': lambda a, b: ((s(a).astype(np.int64) * s(b).astype(np.int64)) >> w).astype(W[w]),
        'avgu': lambda a, b: ((u(a).astype(np.uint64) + u(b)) >> 1).astype(W[w]),
        'avgru': lambda a, b: ((u(a).astype(np.uint64) + u(b) + 1) >> 1).astype(W[w]),
        'absdu': lambda a, b: (np.maximum(u(a), u(b)) - np.minimum(u(a), u(b))),
        'sll': lambda a, b: (u(a).astype(np.uint64) << (u(b) & (w - 1))).astype(np.uint64).__and__(m).astype(W[w]),
        'srl': lambda a, b: (u(a) >> (u(b) & (w - 1))),
        'sra': lambda a, b: (s(a) >> (u(b) & (w - 1)).astype(S[w])).view(W[w]),
        'eq': lambda a, b: np.where(u(a) == u(b), m, 0).astype(W[w]),
        'ltu': lambda a, b: np.where(u(a) < u(b), m, 0).astype(W[w]),
        'lts': lambda a, b: np.where(s(a) < s(b), m, 0).astype(W[w]),
        'leu': lambda a, b: np.where(u(a) <= u(b), m, 0).astype(W[w]),
        'les': lambda a, b: np.where(s(a) <= s(b), m, 0).astype(W[w]),
    }


def unops(w):
    m = (1 << w) - 1
    def u(x): return lanes(x, w)
    def s(x): return lanes(x, w, True)
    ops = {'mov': lambda a: u(a).copy(), 'not': lambda a: ~u(a), 'neg': lambda a: (0 - u(a)).astype(W[w]),
           'abs': lambda a: np.abs(s(a)).view(W[w])}
    for sh in range(1, w):
        ops[f'slli{sh}'] = lambda a, sh=sh: (u(a).astype(np.uint64) << sh & m).astype(W[w])
        ops[f'srli{sh}'] = lambda a, sh=sh: u(a) >> sh
        ops[f'srai{sh}'] = lambda a, sh=sh: (s(a) >> sh).view(W[w])
    return ops


def explain(R, sb):
    """R: 64-byte np.uint8 array. Returns a list of explanations (strings)."""
    out = []
    Rw = R.view(np.uint32)
    # loads from scratch (word pattern 0xA50xxxxx) or from the vpat image area
    if all((x >> 20) == 0xA50 for x in Rw):
        offs = [(x & 0xFFFFF) * 4 for x in Rw]
        if all(offs[i] == offs[0] + 4 * i for i in range(16)):
            out.append(f'load scratch+{offs[0]:#x}')
        else:
            out.append(f'load-like (scratch words {[hex(o) for o in offs[:4]]}...)')
    for k in range(16):
        if np.array_equal(R, VB[k]): out.append(f'= v{k}')
    sc = scalars(sb)
    for w in (8, 16, 32):
        for name, val in sc.items():
            if np.array_equal(lanes(R, w), np.full(64 * 8 // w, val & ((1 << w) - 1), dtype=W[w])):
                out.append(f'broadcast{w} {name}')
    if out: return out
    for w in (8, 16, 32):
        Rl = lanes(R, w)
        for name, f in unops(w).items():
            for a in range(16):
                if np.array_equal(f(VB[a]).astype(W[w]), Rl): out.append(f'{name}.{w} v{a}')
        for name, f in binops(w).items():
            for a in range(16):
                for b in range(16):
                    with np.errstate(all='ignore'):
                        if np.array_equal(f(VB[a], VB[b]).astype(W[w]), Rl): out.append(f'{name}.{w} v{a}, v{b}')
                for sname, val in sc.items():
                    B = np.full(64 * 8 // w, val & ((1 << w) - 1), dtype=W[w]).view(np.uint8)
                    with np.errstate(all='ignore'):
                        if np.array_equal(f(VB[a], B).astype(W[w]), Rl): out.append(f'{name}.{w} v{a}, {sname}')
                        if np.array_equal(f(B, VB[a]).astype(W[w]), Rl): out.append(f'{name}.{w} {sname}, v{a}')
        if out: return out
    # byte permutation / selection from the source registers
    src = {}
    for i in range(64):
        hits = [(k, j) for k in range(16) for j in range(64) if VB[k, j] == R[i]]
        src[i] = hits
    if all(src[i] for i in range(64)):
        regs = collections.Counter(k for i in range(64) for k, j in src[i])
        out.append(f'byte-select from {dict(regs.most_common(3))}')
    return out


def main():
    path = 'sweepv.jsonl'
    only = None
    args = sys.argv[1:]
    if args and not args[0].startswith('--'): path = args.pop(0)
    if args and args[0] == '--hex': only = args[1]
    rows = [json.loads(l) for l in open(path)]
    st = collections.Counter()
    for r in rows:
        if only and r['hex'] != only: continue
        if r['status'] != 1 or not r.get('vmask'):
            continue
        mask = r['vmask']
        ks = [k for k in range(16) if mask >> k & 1]
        exps = []
        for n, k in enumerate(r['vregs'][:min(2, len(ks))]):
            R = np.array(r['mem'][16 * n:16 * n + 16], dtype=np.uint32).view(np.uint8)
            e = explain(R, r['sbase'])
            exps.append(f'v{k}: ' + (' | '.join(e[:4]) if e else '?'))
            st['explained' if e else 'unexplained'] += 1
        print(f"{r['hex']:34} mask {mask:#06x}  " + ' ;; '.join(exps))
    print(st, file=sys.stderr)


if __name__ == '__main__':
    main()
