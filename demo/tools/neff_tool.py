"""neff_tool.py - inspect and modify trn3 NEFFs to call a GPSIMD custom operator.

Commands
  info   <in.neff> [--shell]
      Find the SBUF tile the Pool-engine DMAs load and store, print its location.
      --shell prints KEY=VALUE lines for build scripts (TILE_SBUF_OFFSET, TILE_PARTITIONS, ...).
  inject <in.neff> <kernel_lib.so> <kernel_name> <out.neff>
      Add the kernel library to the NEFF and a KERNEL_CALL between the load and store DMAs.

NEFF container: 1024-byte header + gzip(tar). Header: data_size at 0x10, MD5 of the payload at 0xAC
and 0xCC, feature word at 0x220. Engine programs are 64-byte instruction slots (sgNN/<Engine>0.bin).

What `inject` changes (trn3 GPSIMD kernel-library mechanism, reverse-engineered from libnrt 2.35):
  * sg00/Pool0.json: "gpsimd_kernels_def": "gpsimd_kernels_def.json"
  * sg00/gpsimd_kernels_def.json: [{"name": <kernel_name>, "lib": "/sg00/<lib>"}]
  * sg00/<lib>: the kernel library (gpsimd/build_kernel_lib.sh output)
  * sg00/Pool0.bin: KERNEL_CALL (opcode 0x85, 64 bytes; byte 12 = index into gpsimd_kernels_def)
    inserted right after the load DMA. The KERNEL_CALL takes over the store DMA's wait (load DMA
    complete); the store DMA now waits for the KERNEL_CALL (S[3] >= 1, incremented on completion).
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
OP_KERNEL_CALL = 0x85
OP_DMA_DIRECT2D = 0xD4          # PSEUDO_DMA_DIRECT2D (DGE DMA)
MARKER_VAR = 0x10               # PSEUDO_ADDR8 marker: {u32 offset, u16 var_id_lo, u8 var_id_hi, 0x10}
SBUF_VAR_ID = 0                 # def.json var "SB" (state buffer)
SBUF_PARTITION_STRIDE = 0x40000
DTYPES = {0x0A: ("float32", 4), 0x08: ("int32", 4), 0x09: ("uint32", 4), 0x06: ("bfloat16", 2),
          0x07: ("float16", 2), 0x03: ("uint8", 1), 0x02: ("int8", 1)}
WAIT_SEM_GE = 5                 # WAIT_MODE WAIT_FOR_SEM_GE_IMM
UPDATE_SEM_INC_COMPLETE = 19    # UPDATE_MODE SEM_INC_COMPLETE
POOL_SEM = 3                    # semaphore incremented by the KERNEL_CALL on completion


# ---------------------------------------------------------------- container
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


# ---------------------------------------------------------------- Pool program analysis
def slots_of(blob):
    return [bytearray(blob[i:i + SLOT]) for i in range(0, len(blob), SLOT)]


def addr8(slot, off):
    """Decode a PSEUDO_ADDR8 at `off`; returns ('var', var_id, offset) or ('other', marker, raw)."""
    raw = bytes(slot[off:off + 8])
    if raw[7] == MARKER_VAR:
        offset, lo, hi = struct.unpack_from("<IHB", raw, 0)
        return ("var", lo | hi << 16, offset)
    return ("other", raw[7], raw)


def dma_info(slot):
    src, dst = addr8(slot, 16), addr8(slot, 40)
    s_step, _, s_num, _, s_elem = struct.unpack_from("<iiHHH", slot, 24)
    d_step, _, d_num, _, d_elem = struct.unpack_from("<iiHHH", slot, 48)
    return dict(src=src, dst=dst, src_pattern=(s_step, s_num, s_elem), dst_pattern=(d_step, d_num, d_elem),
                semaphore=slot[13], sem_increment=slot[14], in_dtype=slot[62], out_dtype=slot[63])


def is_sbuf(a):
    return a[0] == "var" and a[1] == SBUF_VAR_ID


def find_load_store(slots):
    loads = [i for i, s in enumerate(slots) if s[0] == OP_DMA_DIRECT2D and is_sbuf(dma_info(s)["dst"])]
    stores = [i for i, s in enumerate(slots) if s[0] == OP_DMA_DIRECT2D and is_sbuf(dma_info(s)["src"])]
    if len(loads) != 1 or len(stores) != 1 or stores[0] < loads[0]:
        raise SystemExit(f"expected one HBM->SBUF DMA followed by one SBUF->HBM DMA on the Pool queue, "
                         f"found loads at {loads}, stores at {stores} (opcodes {[hex(s[0]) for s in slots]})")
    return loads[0], stores[0]


def tile_of(slots):
    li, si = find_load_store(slots)
    ld, st = dma_info(slots[li]), dma_info(slots[si])
    off = ld["dst"][2]
    step, parts, row_bytes = ld["dst_pattern"]
    if step != SBUF_PARTITION_STRIDE:
        raise SystemExit(f"unexpected SBUF partition stride 0x{step:x} in the load DMA")
    if st["src"][2] != off:
        raise SystemExit(f"load writes SBUF offset 0x{off:x} but store reads 0x{st['src'][2]:x}: the "
                         "custom operator would have to copy between tiles (not supported by this demo)")
    first_part, off_in_part = divmod(off, SBUF_PARTITION_STRIDE)
    dname, dsize = DTYPES.get(ld["out_dtype"], (f"dtype_0x{ld['out_dtype']:02x}", 0))
    return dict(sbuf_offset=off_in_part, first_partition=first_part, partitions=parts, row_bytes=row_bytes,
                dtype=dname, elem_bytes=dsize, elems=row_bytes // dsize if dsize else 0, load_slot=li, store_slot=si)


def pool_program(members):
    files = {n: i for i, (n, _) in enumerate(members)}
    if "sg00/Pool0.bin" not in files:
        raise SystemExit("NEFF has no Pool engine program (sg00/Pool0.bin)")
    return files, slots_of(members[files["sg00/Pool0.bin"]][1])


# ---------------------------------------------------------------- commands
def cmd_info(path, shell=False):
    _, members = read_neff(path)
    _, slots = pool_program(members)
    t = tile_of(slots)
    if shell:
        print(f"TILE_SBUF_OFFSET=0x{t['sbuf_offset']:x}")
        print(f"TILE_FIRST_PARTITION={t['first_partition']}")
        print(f"TILE_PARTITIONS={t['partitions']}")
        print(f"TILE_ELEMS={t['elems']}")
        print(f"TILE_DTYPE={t['dtype']}")
        return
    print(f"Pool program: {len(slots)} instructions {[hex(s[0]) for s in slots]}")
    print(f"SBUF tile: partitions {t['first_partition']}..{t['first_partition'] + t['partitions'] - 1}, "
          f"offset 0x{t['sbuf_offset']:x}, {t['row_bytes']} bytes/partition = {t['elems']} x {t['dtype']}")
    print(f"load DMA = instruction {t['load_slot']}, store DMA = instruction {t['store_slot']}")


def kernel_call_slot(kernel_idx, wait_sem, wait_value, args=b""):
    s = bytearray(SLOT)
    s[0], s[1] = OP_KERNEL_CALL, SLOT // 4
    s[4], s[5] = WAIT_SEM_GE, wait_sem
    s[6], s[7] = UPDATE_SEM_INC_COMPLETE, POOL_SEM
    struct.pack_into("<I", s, 8, wait_value)
    s[12] = kernel_idx
    s[13:13 + len(args)] = args
    return bytes(s)


def cmd_inject(src, lib, kname, out):
    header, members = read_neff(src)
    files, slots = pool_program(members)
    t = tile_of(slots)
    li, si = t["load_slot"], t["store_slot"]
    load, store = slots[li], slots[si]
    if store[4] != WAIT_SEM_GE or store[5] != load[13]:
        raise SystemExit("store DMA does not wait on the load DMA's semaphore; unsupported program shape")
    wait_value = struct.unpack_from("<I", store, 8)[0]
    call = kernel_call_slot(0, load[13], wait_value)
    store[4], store[5] = WAIT_SEM_GE, POOL_SEM
    struct.pack_into("<I", store, 8, 1)
    new_slots = [bytes(s) for s in slots[:li + 1]] + [call] + [bytes(s) for s in slots[li + 1:]]
    members[files["sg00/Pool0.bin"]][1] = b"".join(new_slots)

    libname = os.path.basename(lib)
    pool_json = json.loads(members[files["sg00/Pool0.json"]][1])
    pool_json["gpsimd_kernels_def"] = "gpsimd_kernels_def.json"
    members[files["sg00/Pool0.json"]][1] = json.dumps(pool_json).encode()
    members.append(["sg00/gpsimd_kernels_def.json", json.dumps([{"name": kname, "lib": f"/sg00/{libname}"}]).encode()])
    members.append([f"sg00/{libname}", open(lib, "rb").read()])
    write_neff(out, header, members)
    print(f"injected KERNEL_CALL '{kname}' after Pool instruction {li} "
          f"(Pool program {len(slots)} -> {len(new_slots)} instructions), library {libname} "
          f"({os.path.getsize(lib)} bytes): {out}")


def main():
    a = sys.argv[1:]
    if len(a) >= 2 and a[0] == "info":
        cmd_info(a[1], "--shell" in a)
    elif len(a) == 5 and a[0] == "inject":
        cmd_inject(*a[1:])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
