"""Post-link fix-up for a GPSIMD kernel library: fill each 'Neuron' type-2 note's entry
address with the link address of the function it names. usage: klpatch.py <elf> <nm-output>"""
import struct, sys
elf = bytearray(open(sys.argv[1], "rb").read())
syms = {}
for line in open(sys.argv[2]):
    p = line.split()
    if len(p) == 3:
        syms[p[2]] = int(p[0], 16)
phoff, = struct.unpack_from("<I", elf, 0x1c)
phentsize, phnum = struct.unpack_from("<HH", elf, 0x2a)
for i in range(phnum):
    p_type, p_off, _, _, p_filesz = struct.unpack_from("<IIIII", elf, phoff + i * phentsize)
    if p_type != 4:
        continue
    o, end = p_off, p_off + p_filesz
    while o + 12 <= end:
        namesz, descsz, typ = struct.unpack_from("<III", elf, o)
        name = bytes(elf[o + 12:o + 12 + namesz]).rstrip(b"\0")
        d = o + 12 + ((namesz + 3) & ~3)
        if name == b"Neuron" and typ == 2:
            kname = bytes(elf[d + 4:d + descsz]).split(b"\0")[0].decode()
            addr = syms[kname]
            struct.pack_into("<I", elf, d, addr)
            print(f"note: kernel '{kname}' entry = 0x{addr:08x}")
        elif name == b"GNU" and typ == 3:
            print("note: GNU build-id", bytes(elf[d:d + descsz]).hex())
        o = d + ((descsz + 3) & ~3)
open(sys.argv[1], "wb").write(elf)
