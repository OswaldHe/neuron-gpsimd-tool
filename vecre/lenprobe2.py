"""lenprobe2: lengths of the uncertain bundle classes (op0=E, or op0=F with byte3[1:0] in {10, 11}) at REAL
bundle starts: tile each trn3 function from its entry with the known rules and stop at the first uncertain
bundle; test prefix lengths 12..32 (+ addi pair). Appends to lenprobe2.json."""
import json, os, sys, collections
sys.path.insert(0, '.')
import vrun
code = open('code/trn3_kernels.bin', 'rb').read() + open('../re/fw/tpb4_core0_iram_0x0.bin', 'rb').read()
ADD2 = "52c50152c5013df0"
LENS = tuple(int(x) for x in os.environ.get("LENS", "12,16,20,24,28,32").split(","))
ONLY = os.environ.get("ONLY")
def known(i):
    op0 = code[i] & 0xF
    if op0 < 8: return 3
    if op0 < 14: return 2
    if op0 == 0xF and code[i + 3] & 3 == 1: return 8
    if op0 == 0xF and code[i + 3] & 3 == 0: return 16
    return None
entries = [i for i in range(0, len(code) - 3, 4) if code[i] == 0x36 and code[i + 1] & 0xF == 1]
starts = []
for a in entries:
    i = a
    for _ in range(64):
        n = known(i)
        if n is None: starts.append(i); break
        i += n
cls = collections.defaultdict(list)
for i in starts:
    cls[(code[i] & 0xF, code[i + 3] & 3)].append(i)
print({f'{k[0]:x}/{k[1]:02b}': len(v) for k, v in cls.items()})
pick = []
for k, v in cls.items():
    if ONLY and f'{k[0]:x}/{k[1]:02b}' != ONLY: continue
    seen = set()
    for i in v:
        if code[i:i + 4] in seen: continue
        seen.add(code[i:i + 4]); pick.append(i)
        if len([p for p in pick if (code[p] & 0xF, code[p + 3] & 3) == k]) >= int(sys.argv[1]): break
tests, meta = [], []
for i in pick:
    for L in LENS:
        tests += [(code[i:i + L].hex(), {}, None), (code[i:i + L].hex() + ADD2, {}, None)]
        meta.append((i, L))
res = vrun.run(tests, timeout=1800)
lab = collections.defaultdict(list)
for k, (i, L) in enumerate(meta):
    a, b = res[2 * k], res[2 * k + 1]
    ok = a['status'] == b['status'] == 1 and b['regs'][5] - a['regs'][5] == 2
    if ok: lab[i].append(L)
out = [{'off': i, 'hdr': code[i:i + 4].hex(), 'cls': f'{code[i] & 0xF:x}/{code[i + 3] & 3:02b}', 'ok_lengths': lab[i]} for i in pick]
old = json.load(open('lenprobe2.json')) if os.path.exists('lenprobe2.json') else []
json.dump(old + out, open('lenprobe2.json', 'w'), indent=0)
for o in out: print(o['cls'], o['hdr'], o['ok_lengths'])
