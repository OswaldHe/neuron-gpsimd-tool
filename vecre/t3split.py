"""t3split: split trn3 GPSIMD code into instructions/bundles with the learned length rule
(op0 0-7: 3 B, 8-13: 2 B, 0xF: 8 B if byte3 bit0 else 16 B, 0xE: ELEN env (default 16)).
usage: t3split.py <bin> <start> <end>"""
import os, sys
ELEN = int(os.environ.get("ELEN", "16"))
def length(code, i):
    op0 = code[i] & 0xF
    if op0 < 8: return 3
    if op0 < 14: return 2
    if op0 == 0xF: return {0: 16, 1: 8, 2: 16, 3: 10}[code[i + 3] & 3]
    return ELEN
def split(code, a, b):
    i = a
    while i < b:
        n = length(code, i); yield i, code[i:i + n]; i += n
if __name__ == "__main__":
    code = open(sys.argv[1], "rb").read()
    a, b = int(sys.argv[2], 0), int(sys.argv[3], 0)
    for i, ins in split(code, a, b):
        print(f"{i:05x}: {ins.hex(' ')}")
