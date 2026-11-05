"""collect distinct trn3 bundles of the N0-like formats (first bytes 2f45 / 2f35) from the trn3 kernels"""
import collections, json, sys
code = open('code/trn3_kernels.bin', 'rb').read()
c = collections.Counter()
for i in range(len(code) - 8):
    if code[i] == 0x2f and code[i + 1] in (0x45, 0x35):
        c[code[i:i + 8].hex()] += 1
json.dump(c.most_common(), open('n0_bundles.json', 'w'))
print(len(c), sum(c.values()))
