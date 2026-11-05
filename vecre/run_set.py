"""run_set: run a list of hex bundles (json list or [[hex, count], ...]) in the sandbox; save results.
usage: run_set.py <in.json> <out.json> [filter: odd3]"""
import json, sys, collections
sys.path.insert(0, '.')
import vrun
src = json.load(open(sys.argv[1]))
hexes = [x[0] if isinstance(x, list) else x for x in src]
if len(sys.argv) > 3 and sys.argv[3] == 'odd3':
    hexes = [h for h in hexes if int(h[6:8], 16) & 1]
tests = [(h, {}, None) for h in hexes]
res = vrun.run(tests, timeout=900)
json.dump(res, open(sys.argv[2], 'w'))
st = collections.Counter({4: 'hang', 2: 'exc', 1: 'ok'}[r['status']] for r in res)
print(len(res), st)
