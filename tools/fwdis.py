"""Sync-preserving disassembler for trn3 GPSIMD code with ncore2gp tools.

The trn3 GPSIMD core shares the Xtensa NX base ISA with ncore2gp but has different FLIX slot
encodings. Instruction length uses the same rule as ncore2gp (verified: all 92 .xt.prop instruction
segments of the trn3 builtin library decode in sync):
  op0 0-7 -> 3 bytes, 8-0xd -> 2 bytes, 0xe -> 16-byte bundle,
  0xf -> 16-byte bundle if bit 0 of byte 3 is set, else 8-byte bundle.
Zero padding after a return (retw.n/ret.n/retw/ret) up to the next 4-byte boundary is skipped.
Bundles are replaced by same-length NOP padding so xt-objdump stays in sync; they are printed as
`<flix16 ...>` / `<flix8 ...>` with their raw bytes.

usage: fwdis.py code.bin base_addr start end [--quality]
  (start/end are addresses; decoding starts at `start`, which must be an instruction boundary)
"""
import os
import re
import subprocess
import sys
import tempfile

LEN = {**{o: 3 for o in range(8)}, **{o: 2 for o in range(8, 14)}, 0xE: 16, 0xF: 8}
NOP3, NOPN = bytes.fromhex("f02000"), bytes.fromhex("3df0")


def pad(n):
    out = b""
    while n - len(out) >= 3 and (n - len(out)) != 4:
        out += NOP3
    while len(out) < n:
        out += NOPN
    return out


def insn_len(code, i):
    op0 = code[i] & 0xF
    if op0 == 0xF:
        return 16 if code[i + 3] & 1 else 8
    return LEN[op0]


RETURNS = (b"\x1d\xf0", b"\x0d\xf0", b"\x90\x00\x00", b"\x80\x00\x00")


def walk(code, start, end):
    i, items = start, []
    while i < end:
        n = insn_len(code, i)
        items.append((i, n))
        ret = bytes(code[i:i + n]) in RETURNS
        i += n
        if ret:
            while i < end and i % 4 and code[i] == 0:
                items.append((i, -1))   # padding
                i += 1
    return items


def disassemble(code, base, start, end, env):
    items = walk(code, start - base, end - base)
    patched = bytearray(code)
    flix = {}
    for k, (off, n) in enumerate(items):
        if n == -1:
            flix[base + off] = None
            if items[k - 1][1] != -1:      # first padding byte: rewrite return + padding
                roff, rlen = items[k - 1]
                npad = 1
                while k + npad < len(items) and items[k + npad][1] == -1:
                    npad += 1
                is_w = code[roff] == 0x1d or code[roff] == 0x90
                if rlen == 2 and npad == 1:
                    patched[roff:roff + 3] = b"\x90\x00\x00" if is_w else b"\x80\x00\x00"
                else:
                    patched[off:off + npad] = pad(npad)
        elif n >= 8:
            flix[base + off] = bytes(code[off:off + n])
            patched[off:off + n] = pad(n)
    with tempfile.TemporaryDirectory() as td:
        bin_, obj = os.path.join(td, "c.bin"), os.path.join(td, "c.o")
        open(bin_, "wb").write(bytes(patched[start - base:end - base]))
        subprocess.run(["xt-objcopy", "-I", "binary", "-O", "elf32-xtensa-le", "-B", "xtensa",
                        "--rename-section", ".data=.text,alloc,code,readonly,contents",
                        "--add-symbol", f"code=.text:0,function,global", bin_, obj], check=True, env=env)
        out = subprocess.run(["xt-objdump", "-d", obj], capture_output=True, text=True, env=env).stdout
    lines = []
    skip_until = -1
    for ln in out.splitlines():
        m = re.match(r"^\s*([0-9a-f]+):\s+([0-9a-f]+)\s+(.*)$", ln)
        if not m:
            continue
        a = int(m.group(1), 16) + start
        if a < skip_until:
            continue
        if a in flix and flix[a] is None:
            continue
        if a in flix:
            b = flix[a]
            lines.append((a, f"<flix{len(b)} {b.hex()}>"))
            skip_until = a + len(b)
            continue
        t = re.sub(r"\s+", " ", m.group(3)).strip()
        t = re.sub(r"\b([0-9a-f]+) <(code|_binary_\S+?)(\+0x[0-9a-f]+)?>", lambda mm: f"0x{(int(mm.group(1), 16) + start) & 0xffffffff:x}", t)
        lines.append((a, t))
    return lines


def main():
    path, base, start, end = sys.argv[1], int(sys.argv[2], 0), int(sys.argv[3], 0), int(sys.argv[4], 0)
    code = open(path, "rb").read()
    env = dict(os.environ)
    lines = disassemble(code, base, start, end, env)
    if "--quality" in sys.argv:
        bad = sum(1 for _, t in lines if t.startswith(".byte") or t.startswith("ill"))
        print(f"{len(lines)} instructions, {bad} undecodable/ill ({100.0 * bad / max(1, len(lines)):.1f}%)")
        return
    for a, t in lines:
        print(f"{a:08x}: {t}")


if __name__ == "__main__":
    main()
