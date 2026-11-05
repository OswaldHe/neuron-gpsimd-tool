"""sweepv: run bundles in the sandbox with vector context (v0..v15 = vpat before, dumped after); resumable.
usage: sweepv.py <lengths e.g. 8,10,16> <out.jsonl>  (skips bundles that hung in earlier sweeps)"""
import json, os, sys, glob
sys.path.insert(0, '.')
import vrun
lens = {int(x) for x in sys.argv[1].split(',')}
out = sys.argv[2]
skip = set()
for f in glob.glob('sweep*.jsonl'):
    for l in open(f):
        r = json.loads(l)
        if r['status'] in (4, 5, 6): skip.add(r['hex'])
        if f == out: skip.add(r['hex'])
todo = [h for h, n in json.load(open('all_bundles.json')) if len(h) // 2 in lens and h not in skip]
print('todo', len(todo), 'skipped', len(skip), flush=True)
for k in range(0, len(todo), 254):
    chunk = todo[k:k + 254]
    res = vrun.run([(h, {}, None, 1) for h in chunk], timeout=1800)
    with open(out, 'a') as f:
        for r in res: f.write(json.dumps(r) + '\n')
    print('done', k + len(chunk), flush=True)
