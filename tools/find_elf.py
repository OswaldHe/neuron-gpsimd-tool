# Scan a binary for embedded 32-bit LE Xtensa ELF images; print offset, type, size estimate; optionally dump.
import sys, struct, os
data = open(sys.argv[1], "rb").read()
outdir = sys.argv[2] if len(sys.argv) > 2 else None
i = 0; n = 0
while True:
    i = data.find(b"\x7fELF\x01\x01", i)
    if i < 0: break
    e_type, e_machine = struct.unpack_from("<HH", data, i + 16)
    if e_machine == 94:  # EM_XTENSA
        shoff, = struct.unpack_from("<I", data, i + 32)
        shentsize, shnum = struct.unpack_from("<HH", data, i + 46)
        size = shoff + shentsize * shnum
        print(f"off=0x{i:x} type={e_type} size={size}")
        if outdir:
            open(os.path.join(outdir, f"emb_{n:02d}_0x{i:x}_t{e_type}.elf"), "wb").write(data[i:i+size])
        n += 1
    i += 4
