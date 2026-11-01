"""Load a NEFF with libnrt (ctypes), feed input0 = arange (float32: *0.25, int32: as is), execute,
compare the output with the expectation, optionally map SBUF for GPSIMD kernels and print their logs.

usage: run_neff.py file.neff [--vnc N] [--repeat N] [--expect identity|add:K] [--map-sbuf [TPB]] [--klog]
  --map-sbuf : after nrt_load, program GPSIMD address-translation entry 38 so core address
               0xD0000000 + p*0x40000 + off = SBUF partition p (tools/sbufwin.py). Default TPB 4.
  --klog     : after execution, print kprint.h logs of all GPSIMD cores (tools/klog.py, BAR0).
"""
import ctypes as C
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
nrt = C.CDLL("/opt/aws/neuron/lib/libnrt.so")
DT = {0xA: np.float32, 0x8: np.int32, 0x9: np.uint32}


class TensorInfo(C.Structure):
    _fields_ = [("name", C.c_char * 256), ("usage", C.c_int), ("size", C.c_size_t),
                ("dtype", C.c_int), ("shape", C.POINTER(C.c_uint32)), ("ndim", C.c_uint32)]


class TensorInfoArray(C.Structure):
    _fields_ = [("tensor_count", C.c_uint64), ("tensor_array", TensorInfo * 0)]


def check(rc, what):
    if rc != 0:
        raise SystemExit(f"{what} failed: NRT_STATUS={rc}")


def opt(name, default=None, has_value=True):
    if name not in sys.argv:
        return default
    i = sys.argv.index(name)
    if not has_value:
        return True
    return sys.argv[i + 1] if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--") else default


def main():
    path = sys.argv[1]
    vnc = int(opt("--vnc", 0))
    repeat = int(opt("--repeat", 1))
    expect = opt("--expect", "identity")
    neff = open(path, "rb").read()
    check(nrt.nrt_init(1, b"", b""), "nrt_init")
    model = C.c_void_p()
    check(nrt.nrt_load(neff, C.c_size_t(len(neff)), vnc, 1, C.byref(model)), "nrt_load")
    if "--map-sbuf" in sys.argv:
        import sbufwin
        tpb = int(opt("--map-sbuf", 4))
        sbufwin.map_sbuf(tpb)
        print(f"[mapped TPB{tpb} SBUF at GPSIMD core address 0x{sbufwin.DEFAULT_LOCAL:08x} (xlat entry {sbufwin.DEFAULT_ENTRY})]")

    info_p = C.POINTER(TensorInfoArray)()
    check(nrt.nrt_get_model_tensor_info(model, C.byref(info_p)), "nrt_get_model_tensor_info")
    infos = C.cast(C.addressof(info_p.contents.tensor_array), C.POINTER(TensorInfo))
    ins, outs = C.c_void_p(), C.c_void_p()
    check(nrt.nrt_allocate_tensor_set(C.byref(ins)), "alloc in set")
    check(nrt.nrt_allocate_tensor_set(C.byref(outs)), "alloc out set")
    inputs, out_tensors = {}, {}
    for i in range(info_p.contents.tensor_count):
        ti = infos[i]
        name = ti.name.decode()
        dt = DT.get(ti.dtype, np.float32)
        t = C.c_void_p()
        check(nrt.nrt_tensor_allocate(0, vnc, C.c_size_t(ti.size), ti.name, C.byref(t)), f"alloc {name}")
        if ti.usage == 0:
            data = np.arange(ti.size // 4).astype(dt)
            if dt == np.float32:
                data *= 0.25
            check(nrt.nrt_tensor_write(t, data.ctypes.data, C.c_size_t(0), C.c_size_t(ti.size)), "write")
            inputs[name] = data
            check(nrt.nrt_add_tensor_to_tensor_set(ins, ti.name, t), "add in")
        else:
            out_tensors[name] = (t, ti.size, dt)
            check(nrt.nrt_add_tensor_to_tensor_set(outs, ti.name, t), "add out")
        print(f"tensor {name}: {'input' if ti.usage == 0 else 'output'} {ti.size} bytes {np.dtype(dt).name}")

    for _ in range(repeat):
        check(nrt.nrt_execute(model, ins, outs), "nrt_execute")
    ref = next(iter(inputs.values()))
    if expect.startswith("add:"):
        k = float(expect[4:]) if ref.dtype == np.float32 else int(expect[4:])
        exp = ref + np.array(k, dtype=ref.dtype)   # each execution re-DMAs the input
    else:
        exp = ref
    ok = True
    for name, (t, size, dt) in out_tensors.items():
        buf = np.empty(size // 4, dtype=dt)
        check(nrt.nrt_tensor_read(t, buf.ctypes.data, C.c_size_t(0), C.c_size_t(size)), "read")
        bad = np.flatnonzero(buf != exp)
        ok &= bad.size == 0
        print(f"{name}: first values {buf[:6]}  expected ({expect}) {exp[:6]}  "
              f"{'MATCH' if bad.size == 0 else f'{bad.size} mismatches, first at {bad[:4]}'}")
        if "--max-err" in sys.argv:
            print(f"   max|out-in| = {float(np.abs(buf.astype(np.float64) - ref).max())}")
    if "--klog" in sys.argv:
        import klog
        klog.dump(clear=True)
    nrt.nrt_unload(model)
    nrt.nrt_close()
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
