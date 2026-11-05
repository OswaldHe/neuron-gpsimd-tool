"""summarize sandbox results: stores (scratch changes), register updates, exceptions"""
import json, sys, collections
sys.path.insert(0, '.')
import vrun
res = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'n0_results.json'))
st = collections.Counter()
rows = []
for r in res:
    tc = (r['hex'], {}, None)
    if r['status'] == 4: st['hang'] += 1; continue
    if r['status'] == 2: st[f"exc{r['cause'] & 0xf}"] += 1; continue
    init = vrun.regs_init(r, r['sbase'], tc)
    ch = {k: (init[k] - r['sbase'], r['regs'][k] - r['sbase']) for k in range(3, 16) if r['regs'][k] != init[k]}
    if r['nchanged']:
        base_reg = (r['first'] - 0x1000) // 0x100
        st['store'] += 1
        rows.append(('STORE', r['hex'], f"bytes {r['nchanged']} at +{r['first']:#x}..{r['last']:#x} (a{base_reg}+{(r['first'] - 0x1000) % 0x100:#x})",
                     {f'a{k}': f'{a:+#x}->{b:+#x}' for k, (a, b) in ch.items()}))
    elif ch:
        st['reg'] += 1
        rows.append(('REG', r['hex'], '', {f'a{k}': f'{a:+#x}->{b:+#x}' if abs(b) < 0x10000 else f'{a:+#x}->{b + r["sbase"]:#x}' for k, (a, b) in ch.items()}))
    else:
        st['none'] += 1
print(st)
for x in sorted(rows): print(*x)
