"""lenprobe: measure trn3 bundle lengths on the device. For candidate bundle starts taken from trn3 code,
run prefix(L) alone and prefix(L) + 'addi a5,a5,1; addi a5,a5,1; nop.n' for L = 8, 16. The bundle is L bytes
when the two runs complete and a5 differs by exactly 2. Results (full) are appended to lenprobe.json."""
import json, os, random, sys
sys.path.insert(0, '.')
import vrun
code = open('code/trn3_kernels.bin', 'rb').read()
ADD2 = "52c50152c5013df0"
N, seed = int(sys.argv[1]), int(sys.argv[2])
random.seed(seed)
cand = [i for i in range(len(code) - 32) if code[i] & 0xF in (0xE, 0xF)]
done = {d['off'] for d in json.load(open('lenprobe.json'))} if os.path.exists('lenprobe.json') else set()
seen, pick = set(), []
for i in random.sample(cand, len(cand)):
    h = code[i:i + 4]
    if h in seen or i in done: continue
    seen.add(h); pick.append(i)
pick = pick[:N]
tests = []
for i in pick:
    for L in (8, 16):
        tests += [(code[i:i + L].hex(), {}, None), (code[i:i + L].hex() + ADD2, {}, None)]
res = vrun.run(tests, timeout=900)
def s(r):
    return {'st': r['status'], 'cause': r['cause'], 'epc': r['epc'], 'a5': (r['regs'][5] - r['sbase'] - 0x1500) if r['sbase'] else None}
out = json.load(open('lenprobe.json')) if os.path.exists('lenprobe.json') else []
for k, i in enumerate(pick):
    a8, b8, a16, b16 = (s(r) for r in res[4 * k:4 * k + 4])
    lab = None
    if a8['st'] == b8['st'] == 1 and b8['a5'] - a8['a5'] == 2: lab = 8
    elif a16['st'] == b16['st'] == 1 and b16['a5'] - a16['a5'] == 2: lab = 16
    out.append({'off': i, 'hdr': code[i:i + 4].hex(), 'len': lab, 'r': [a8, b8, a16, b16]})
json.dump(out, open('lenprobe.json', 'w'))
import collections
print('labels', collections.Counter(o['len'] for o in out))
