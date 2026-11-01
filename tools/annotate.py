# Annotate objdump (intel syntax) lines: for "# <hexaddr>" comments pointing into .rodata,
# append the C string found there. usage: annotate.py <elf> < disasm > annotated
import re
import struct
import sys

elf = open(sys.argv[1], "rb").read()
# parse section headers (ELF64 LE)
shoff, = struct.unpack_from("<Q", elf, 0x28)
shentsize, shnum, shstrndx = struct.unpack_from("<HHH", elf, 0x3A)
secs = []
for i in range(shnum):
    name, typ, flags, addr, off, size = struct.unpack_from("<IIQQQQ", elf, shoff + i * shentsize)
    secs.append([name, addr, off, size])
strtab_off = secs[shstrndx][2]


def secname(n):
    e = elf.index(b"\0", strtab_off + n)
    return elf[strtab_off + n:e].decode()


ro = [(a, o, s) for (n, a, o, s) in secs if secname(n) in (".rodata", ".data.rel.ro", ".data")]


def cstr(addr):
    for a, o, s in ro:
        if a <= addr < a + s:
            p = o + (addr - a)
            e = elf.find(b"\0", p, p + 200)
            raw = elf[p:e if e >= 0 else p + 200]
            if len(raw) >= 2 and all(32 <= c < 127 or c in (9, 10) for c in raw):
                return raw.decode().replace("\n", "\\n")
    return None


pat = re.compile(r"# ([0-9a-f]+)")
for line in sys.stdin:
    line = line.rstrip("\n")
    m = pat.search(line)
    if m:
        s = cstr(int(m.group(1), 16))
        if s:
            line += f'   ; "{s}"'
    print(line)
