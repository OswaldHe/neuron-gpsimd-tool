"""opsweep: sweep the slot-3 opcode space of the 8-byte '2f45 .. .. .. .. 50 32' format (slots 0-2 nop) with
fixed operands vd=v6, va=v1, vb=v3 in vector mode; resumable JSONL.
Field map: va = bits 17..21, vd = bits 22,23,26,27,28, vb = bits 35..39; opcode = b3[7:5], b4[2:0], b5, b6 bit 0.
usage: opsweep.py <out.jsonl> <b4lo list e.g. 1> [b6 list e.g. 0x50]"""
import json, os, sys
sys.path.insert(0, '.')
import vrun
def enc(d, a, b, op3, oplo, b5, b6=0x50, b7=0x32):
    b2 = (a << 1) | ((d & 1) << 6) | (((d >> 1) & 1) << 7)
    b3 = 0x01 | (((d >> 2) & 7) << 2) | (op3 << 5)
    b4 = (b << 3) | oplo
    return bytes([0x2f, 0x45, b2, b3, b4, b5, b6, b7]).hex()
if __name__ == '__main__':
    out = sys.argv[1]
    los = [int(x, 0) for x in sys.argv[2].split(',')]
    b6s = [int(x, 0) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else [0x50]
    done = {json.loads(l)['hex'] for l in open(out)} if os.path.exists(out) else set()
    todo = [enc(6, 1, 3, op3, lo, b5, b6) for b6 in b6s for lo in los for op3 in range(8) for b5 in range(256)]
    todo = [h for h in todo if h not in done]
    print('todo', len(todo), flush=True)
    for k in range(0, len(todo), 254):
        chunk = todo[k:k + 254]
        res = vrun.run([(h, {}, None, 1) for h in chunk], timeout=1800)
        with open(out, 'a') as f:
            for r in res: f.write(json.dumps(r) + '\n')
        print('done', k + len(chunk), flush=True)
