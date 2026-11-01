"""Host-side reader/writer for Neuron device registers and on-chip memory through BAR0
(NEURON_IOCTL_BAR_READ / _WRITE of the aws-neuronx driver, 32-bit accesses).

trn3 (v4) BAR0 layout (driver v4/address_map.h):
  TPB i window: BAR0 + 0xd0000000 + i * 0x4000000 (64 MB), mirrors SoC V4_MMAP_TPB_i_BASE
                (0x20_0000_0000, 0x30_..., 0x60_..., 0x70_..., 0x8020_..., 0x8030_..., 0x8060_..., 0x8070_...)
                first 32 MB = SBUF (partition p, byte o -> p * 0x40000 + o, per NEFF/ISA addressing)

usage as a CLI:
  devmem.py [--dev N] read  <bar0_offset> <nwords>
  devmem.py [--dev N] tpb   <tpb_idx> <tpb_offset> <nwords>
  devmem.py [--dev N] dump  <tpb_idx> <tpb_offset> <nbytes> <out.bin>
"""
import array
import ctypes as C
import fcntl
import os
import struct
import sys

IOCTL_BAR_READ = 0x80084E0B     # _IOR('N', 11, struct neuron_ioctl_bar_rw *)
IOCTL_BAR_WRITE = 0x40084E0C    # _IOW('N', 12, struct neuron_ioctl_bar_rw *)
TPB_BAR0_OFFSET = 0xD0000000
TPB_BAR0_DIST = 0x4000000
CHUNK = 100  # driver readless path: FW_IO_MAX_READLESS_READ_REGISTER_COUNT (100) registers per ioctl


class BarRw(C.Structure):
    _fields_ = [("bar", C.c_uint32), ("address", C.c_void_p), ("data", C.c_void_p), ("count", C.c_uint32)]


def _pci_bdf(dev):
    link = os.readlink(f"/sys/class/neuron_device/neuron{dev}/device") if os.path.exists(
        f"/sys/class/neuron_device/neuron{dev}/device") else None
    if link:
        return os.path.basename(link)
    # fall back to neuron-ls order
    import subprocess
    out = subprocess.run(["neuron-ls"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        p = [x.strip() for x in line.split("|")]
        if len(p) > 6 and p[1] == str(dev):
            return p[6]
    raise RuntimeError(f"cannot find PCI BDF of neuron{dev}")


class Dev:
    def __init__(self, dev=0):
        self.fd = os.open(f"/dev/neuron{dev}", os.O_RDWR)
        bdf = _pci_bdf(dev)
        if not bdf.startswith("0000:"):
            bdf = "0000:" + bdf
        self.bar0_pa = int(open(f"/sys/bus/pci/devices/{bdf}/resource").readline().split()[0], 16)

    def read(self, bar0_off, nwords, chunk=CHUNK):
        out = array.array("I")
        for base in range(0, nwords, chunk):
            n = min(chunk, nwords - base)
            addrs = (C.c_uint64 * n)(*[self.bar0_pa + bar0_off + 4 * (base + k) for k in range(n)])
            data = (C.c_uint32 * n)()
            arg = BarRw(0, C.cast(addrs, C.c_void_p), C.cast(data, C.c_void_p), n)
            fcntl.ioctl(self.fd, IOCTL_BAR_READ, arg)
            out.extend(data)
        return out

    def write(self, bar0_off, words):
        n = len(words)
        addrs = (C.c_uint64 * n)(*[self.bar0_pa + bar0_off + 4 * k for k in range(n)])
        data = (C.c_uint32 * n)(*words)
        arg = BarRw(0, C.cast(addrs, C.c_void_p), C.cast(data, C.c_void_p), n)
        fcntl.ioctl(self.fd, IOCTL_BAR_WRITE, arg)

    def tpb_read(self, tpb, off, nwords):
        return self.read(TPB_BAR0_OFFSET + tpb * TPB_BAR0_DIST + off, nwords)

    def tpb_write(self, tpb, off, words):
        return self.write(TPB_BAR0_OFFSET + tpb * TPB_BAR0_DIST + off, words)

    def tpb_bytes(self, tpb, off, nbytes):
        return self.tpb_read(tpb, off, (nbytes + 3) // 4).tobytes()[:nbytes]


def main():
    args = sys.argv[1:]
    dev = 0
    if args and args[0] == "--dev":
        dev, args = int(args[1]), args[2:]
    d = Dev(dev)
    cmd = args[0]
    if cmd == "read":
        off, n = int(args[1], 0), int(args[2], 0)
        w = d.read(off, n)
    elif cmd in ("tpb", "dump"):
        tpb, off, n = int(args[1], 0), int(args[2], 0), int(args[3], 0)
        if cmd == "dump":
            open(args[4], "wb").write(d.tpb_bytes(tpb, off, n))
            print(f"wrote {n} bytes of TPB{tpb}+0x{off:x} to {args[4]}")
            return
        w = d.tpb_read(tpb, off, n)
        off = TPB_BAR0_OFFSET + tpb * TPB_BAR0_DIST + off
    else:
        raise SystemExit(__doc__)
    for i in range(0, len(w), 4):
        print(f"0x{off + 4 * i:09x}: " + " ".join(f"{x:08x}" for x in w[i:i + 4]))


if __name__ == "__main__":
    main()
