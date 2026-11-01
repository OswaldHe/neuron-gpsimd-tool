"""Inject a trn3 GPSIMD *kernel library* call into the NKI passthrough NEFF.

trn3 path (reverse-engineered from libnrt 2.35.30 / libnrtucode_extisa):
  * sg00/Pool0.json gets  "gpsimd_kernels_def": "<file>.json"
  * that file is a JSON array of {"name": <kernel name>, "lib": "/<in-NEFF path>"}
  * the library is an ELF built with the pisplitload LSP + GNU build-id + 'Neuron' type-2 notes
  * KERNEL_CALL (opcode 0x85, 64 bytes): byte 12 = index into gpsimd_kernels_def; the runtime
    replaces it with the firmware's global kernel id (nrtucode_ll_builder_add_kl_call).

Pool stream produced:
  SET_ORDERING_MODE | DMA in (signals S[8]+=16) | KERNEL_CALL (wait S[8]>=16, S[3]+=1 on done)
  | [extra 64-byte argument slots] | DMA out (wait S[3]>=1) | NOP (wait S[9]>=16)

usage: inject_kl.py in.neff lib.so kernel_name out.neff [--arg-hex HEX] [--out-from-kernel]
  --arg-hex: bytes 13..63 of the KERNEL_CALL (kernel-defined argument area), hex string
"""
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inject_customop import read_neff, write_neff, SLOT  # noqa: E402


def kernel_call(kernel_idx, arg=b""):
    s = bytearray(SLOT)
    s[0], s[1] = 0x85, 0x10
    s[4], s[5] = 5, 8                     # WAIT_FOR_SEM_GE_IMM on S[8]
    s[6], s[7] = 19, 3                    # SEM_INC_COMPLETE on S[3]
    struct.pack_into("<I", s, 8, 16)      # wait value
    s[12] = kernel_idx
    s[13:13 + len(arg)] = arg
    return bytes(s)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    src_neff, lib, kname, out_neff = args[:4]
    arg = b""
    if "--arg-hex" in sys.argv:
        arg = bytes.fromhex(sys.argv[sys.argv.index("--arg-hex") + 1])
        assert len(arg) <= 51
    header, members = read_neff(src_neff)
    files = {n: i for i, (n, _) in enumerate(members)}

    pool = members[files["sg00/Pool0.bin"]][1]
    slots = [bytearray(pool[i:i + SLOT]) for i in range(0, len(pool), SLOT)]
    assert [s[0] for s in slots] == [0xB1, 0xD4, 0xD4, 0xA4], [hex(s[0]) for s in slots]
    dma_out = slots[2]
    dma_out[4], dma_out[5] = 5, 3          # wait S[3] >= 1 (the kernel call finished)
    struct.pack_into("<I", dma_out, 8, 1)
    extra = []
    if "--payload-hex" in sys.argv:   # extra 64-byte slots after the KERNEL_CALL (opcode byte included)
        raw = bytes.fromhex(sys.argv[sys.argv.index("--payload-hex") + 1])
        assert len(raw) % SLOT == 0
        extra = [raw[i:i + SLOT] for i in range(0, len(raw), SLOT)]
    new_pool = b"".join([bytes(slots[0]), bytes(slots[1]), kernel_call(0, arg), *extra,
                         bytes(dma_out), bytes(slots[3])])
    members[files["sg00/Pool0.bin"]][1] = new_pool

    libname = os.path.basename(lib)
    pj = json.loads(members[files["sg00/Pool0.json"]][1])
    pj["gpsimd_kernels_def"] = "gpsimd_kernels_def.json"
    members[files["sg00/Pool0.json"]][1] = json.dumps(pj).encode()
    kdef = [{"name": kname, "lib": f"/sg00/{libname}"}]
    members.append(["sg00/gpsimd_kernels_def.json", json.dumps(kdef).encode()])
    members.append([f"sg00/{libname}", open(lib, "rb").read()])
    write_neff(out_neff, header, members)
    print(f"wrote {out_neff}: Pool0.bin {len(pool)} -> {len(new_pool)} B, kernel '{kname}' "
          f"from {libname} ({os.path.getsize(lib)} B), arg={arg.hex() or '-'}")


if __name__ == "__main__":
    main()
