"""Inject a GPSIMD custom-op library + KERNEL_CALL into an NKI-compiled NEFF.

Starts from the NKI passthrough kernel (Pool stream: SET_ORDERING_MODE, DMA in, DMA out, NOP)
and produces the same layout the trn1 compiler emits for an AwsNeuronCustomOp:

  SET_ORDERING_MODE
  DMA   HBM a        -> SB p0  @0x28000 (1 x 32768 B)           signals S[8] += 16
  KERNEL_CALL fn=0 nargs=1, wait S[8] >= 16, then S[3] += 1
  CUSTOM_OP_PAYLOAD  out: SBUF p32 @0x28000, fp32 [128,64], 1 partition x 8192 elem
  CUSTOM_OP_PAYLOAD  in : SBUF p0  @0x28000, fp32 [128,64], 1 partition x 8192 elem
  DMA   SB p32 @0x28000 -> HBM output_0, wait S[3] >= 1        signals S[9] += 16
  NOP   wait S[9] >= 16

usage: inject_customop.py in.neff lib.so function_name out.neff [--legacy-versions]
"""
import gzip
import hashlib
import io
import json
import os
import struct
import sys
import tarfile

SLOT = 64
SB_IN = 0x00028000            # partition 0, offset 0x28000
SB_OUT = 0x00828000           # partition 32, offset 0x28000 (same as the trn1 compiler)
NBYTES = 128 * 64 * 4
FP32 = 0x0A


def read_neff(path):
    raw = open(path, "rb").read()
    header, payload = bytearray(raw[:1024]), raw[1024:]
    members = []
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(payload)), mode="r:") as tf:
        for m in tf.getmembers():
            if m.isfile():
                members.append([m.name, tf.extractfile(m).read()])
    return header, members


def write_neff(path, header, members):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tf:
        for name, data in members:
            ti = tarfile.TarInfo(name)
            ti.size, ti.mode, ti.mtime = len(data), 0o644, 0
            tf.addfile(ti, io.BytesIO(data))
    payload = gzip.compress(buf.getvalue(), compresslevel=1, mtime=0)
    md5 = hashlib.md5(payload).digest()
    struct.pack_into("<Q", header, 0x10, len(payload))
    header[0xAC:0xBC] = md5
    header[0xCC:0xDC] = md5
    open(path, "wb").write(bytes(header) + payload)


def hdr(opcode, wait=(0, 0, 0), update=(0, 0)):
    s = bytearray(SLOT)
    s[0], s[1] = opcode, 0x10                       # 16 words = 64 bytes
    s[4], s[5] = wait[0], wait[1]
    s[6], s[7] = update[0], update[1]
    struct.pack_into("<I", s, 8, wait[2])
    return s


def kernel_call(function_id, nargs, npayloads):
    # wait S[8] >= 16 (input DMA complete); SEM_INC_COMPLETE (19) on S[3] when done
    s = hdr(0x85, wait=(5, 8, 16), update=(19, 3))
    struct.pack_into("<H", s, 12, npayloads)
    s[14], s[15] = function_id, nargs
    return s


def sbuf_payload(addr, shape, dtype=FP32):
    s = hdr(0x86)
    s[15] = 1          # arg_type TENSOR
    s[16] = 1          # location SBUF
    s[17] = 0          # INLINE_SHAPE8D
    s[18] = dtype
    for i, d in enumerate(shape):
        struct.pack_into("<H", s, 24 + 2 * i, d)
    n = 1
    for d in shape:
        n *= d
    struct.pack_into("<I", s, 40, addr)     # tpb addr
    struct.pack_into("<I", s, 44, n)        # observed = total elements in the trn1 reference
    s[48] = 1                               # num_partitions
    struct.pack_into("<I", s, 52, n)        # num_elem_per_block
    return s


def retarget_dma(slot, *, src=None, dst=None, wait=None):
    """Rewrite a PSEUDO_DMA_DIRECT2D as a single contiguous NBYTES row."""
    s = bytearray(slot)
    assert s[0] == 0xD4, hex(s[0])
    if wait is not None:
        s[4], s[5] = wait[0], wait[1]
        struct.pack_into("<I", s, 8, wait[2])
    if src is not None:
        s[16:24] = src
    if dst is not None:
        s[40:48] = dst
    struct.pack_into("<iiHHH", s, 24, NBYTES, 1, 1, 1, NBYTES & 0xFFFF)   # src step/num/elem
    struct.pack_into("<iiHHH", s, 48, NBYTES, 1, 1, 1, NBYTES & 0xFFFF)   # dst step/num/elem
    return s


def sb_var(offset):
    # PSEUDO_ADDR8 VAR_ID form: var_id 0 (state buffer), marker 0x10
    return struct.pack("<IHBB", offset, 0, 0, 0x10)


def main():
    src_neff, lib, fn, out_neff = sys.argv[1:5]
    header, members = read_neff(src_neff)
    files = {n: i for i, (n, _) in enumerate(members)}

    pool = members[files["sg00/Pool0.bin"]][1]
    slots = [pool[i:i + SLOT] for i in range(0, len(pool), SLOT)]
    ops = [s[0] for s in slots]
    assert ops == [0xB1, 0xD4, 0xD4, 0xA4], [hex(o) for o in ops]
    dma_in = retarget_dma(slots[1], dst=sb_var(SB_IN))
    if "--control" in sys.argv:
        # DMA edits + repack only: output DMA reads the input tile, original wait S[8] >= 16.
        members[files["sg00/Pool0.bin"]][1] = b"".join(
            [slots[0], dma_in, retarget_dma(slots[2], src=sb_var(SB_IN)), slots[3]])
        write_neff(out_neff, header, members)
        print(f"wrote control {out_neff}")
        return
    dma_out = retarget_dma(slots[2], src=sb_var(SB_OUT), wait=(5, 3, 1))
    new_pool = b"".join([
        slots[0], dma_in,
        kernel_call(function_id=0, nargs=1, npayloads=2),
        sbuf_payload(SB_OUT, (128, 64)),
        sbuf_payload(SB_IN, (128, 64)),
        dma_out, slots[3],
    ])
    members[files["sg00/Pool0.bin"]][1] = new_pool

    libname = os.path.basename(lib)
    ucode = {"ucode_lib": [{
        "cpu_id": 0,
        "functions": [{"name": fn, "opcode": 133, "sub_opcode": 0}],
        "is_builtin": False,
        "library": libname,
        "total_cpus": 1,
        "ulib_to_isa_version": "1.0.2520.0",
        "ulib_to_ucode_version": "1.25.36.0",
    }]}
    d = json.loads(members[files["sg00/def.json"]][1])
    d["ucode_lib"] = "ucode_lib.json"
    d["neff_features"] = sorted(set(d.get("neff_features", [])) | {"neff_feature_custom_ops"})
    members[files["sg00/def.json"]][1] = json.dumps(d).encode()
    members.append(["sg00/ucode_lib.json", json.dumps(ucode, indent=4).encode()])
    if "--kernels-def" in sys.argv:
        # trn2+/trn3 interface: the *engine* json (Pool0.json) names a JSON array of
        # {name, lib: "/<in-NEFF path>"} entries (parse_one_engine_instr in libnrt).
        pj = json.loads(members[files["sg00/Pool0.json"]][1])
        pj["gpsimd_kernels_def"] = "gpsimd_kernels_def.json"
        members[files["sg00/Pool0.json"]][1] = json.dumps(pj).encode()
        kdef = [{"name": fn, "lib": f"/sg00/{libname}"}]
        members.append(["sg00/gpsimd_kernels_def.json", json.dumps(kdef).encode()])
    members.append([f"sg00/{libname}", open(lib, "rb").read()])

    feat, = struct.unpack_from("<I", header, 0x220)
    struct.pack_into("<I", header, 0x220, feat | (1 << 2))   # custom-ops feature bit
    write_neff(out_neff, header, members)
    print(f"wrote {out_neff}: Pool0.bin {len(pool)} -> {len(new_pool)} bytes, "
          f"feature 0x{feat:x} -> 0x{feat | 4:x}, + {libname} ({os.path.getsize(lib)} B)")


if __name__ == "__main__":
    main()
