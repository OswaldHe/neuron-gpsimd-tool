"""learn the trn3 FLIX length rule from lenprobe.json labels (decision tree over bits of bytes 0..3)"""
import json, collections, itertools
D = json.load(open('lenprobe.json'))
S = []
for d in D:
    h = bytes.fromhex(d['hdr'])
    if d.get('len'): S.append((h, d['len']))
print('labelled', len(S), collections.Counter(n for _, n in S), 'unlabelled', len(D) - len(S))
def bit(h, k): return h[k // 8] >> (k % 8) & 1
for op0 in (0xE, 0xF):
    L = [(h, n) for h, n in S if h[0] & 0xF == op0]
    print(f'op0={op0:x}', collections.Counter(n for _, n in L))
    if not L or len({n for _, n in L}) < 2: continue
    def tree(L, depth, used=()):
        cnt = collections.Counter(n for _, n in L)
        if len(cnt) <= 1 or depth == 0: return cnt.most_common(1)[0][0], sum(cnt.values()) - cnt.most_common(1)[0][1]
        best = None
        for k in range(4, 32):
            if k in used: continue
            a = [(h, n) for h, n in L if bit(h, k) == 0]; b = [(h, n) for h, n in L if bit(h, k)]
            if not a or not b: continue
            ea = len(a) - collections.Counter(n for _, n in a).most_common(1)[0][1]
            eb = len(b) - collections.Counter(n for _, n in b).most_common(1)[0][1]
            if best is None or ea + eb < best[0]: best = (ea + eb, k, a, b)
        if best is None: return cnt.most_common(1)[0][0], 0
        _, k, a, b = best
        ta, ea = tree(a, depth - 1, used + (k,)); tb, eb = tree(b, depth - 1, used + (k,))
        return (k, ta, tb), ea + eb
    for depth in (1, 2, 3, 4):
        t, err = tree(L, depth)
        print(f'  depth {depth}: errors {err}: {t}')
    for h, n in sorted(L, key=lambda x: (x[1], x[0][3])): print('   ', n, h.hex(' '), f'b3={h[3]:08b}')
