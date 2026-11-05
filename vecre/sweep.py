"""sweep: run bundles from all_bundles.json in the sandbox, appending one JSON line per result to <out.jsonl>
(resumable). usage: sweep.py <lengths e.g. 8,10> <out.jsonl>"""
import json, os, sys
sys.path.insert(0, '.')
import vrun
lens = {int(x) for x in sys.argv[1].split(',')}
out = sys.argv[2]
done = set()
if os.path.exists(out):
    for l in open(out): done.add(json.loads(l)['hex'])
todo = [h for h, n in json.load(open('all_bundles.json')) if len(h) // 2 in lens and h not in done]
print('todo', len(todo), flush=True)
for k in range(0, len(todo), 254):
    chunk = todo[k:k + 254]
    res = vrun.run([(h, {}, None) for h in chunk], timeout=1800)
    with open(out, 'a') as f:
        for r in res: f.write(json.dumps(r) + '\n')
    print('done', k + len(chunk), flush=True)
