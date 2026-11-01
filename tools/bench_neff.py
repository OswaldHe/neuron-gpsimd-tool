# Load each NEFF (one nrt_init session), execute it N times, report latency stats per NEFF.
# usage: bench_neff.py N file1.neff [file2.neff ...]   (prints JSON lines)
import ctypes as C
import json
import sys
import time

import numpy as np

nrt = C.CDLL("/opt/aws/neuron/lib/libnrt.so")


class TensorInfo(C.Structure):
    _fields_ = [("name", C.c_char * 256), ("usage", C.c_int), ("size", C.c_size_t),
                ("dtype", C.c_int), ("shape", C.POINTER(C.c_uint32)), ("ndim", C.c_uint32)]


class TensorInfoArray(C.Structure):
    _fields_ = [("tensor_count", C.c_uint64), ("tensor_array", TensorInfo * 0)]


def check(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} failed: NRT_STATUS={rc}")


def run_one(path, n, vnc=0):
    neff = open(path, "rb").read()
    model = C.c_void_p()
    check(nrt.nrt_load(neff, C.c_size_t(len(neff)), vnc, 1, C.byref(model)), "nrt_load")
    info_p = C.POINTER(TensorInfoArray)()
    check(nrt.nrt_get_model_tensor_info(model, C.byref(info_p)), "tensor_info")
    infos = C.cast(C.addressof(info_p.contents.tensor_array), C.POINTER(TensorInfo))
    ins, outs = C.c_void_p(), C.c_void_p()
    check(nrt.nrt_allocate_tensor_set(C.byref(ins)), "set")
    check(nrt.nrt_allocate_tensor_set(C.byref(outs)), "set")
    out_t = []
    for i in range(info_p.contents.tensor_count):
        ti = infos[i]
        t = C.c_void_p()
        check(nrt.nrt_tensor_allocate(0, vnc, C.c_size_t(ti.size), ti.name, C.byref(t)), "alloc")
        if ti.usage == 0:
            data = np.arange(ti.size // 4, dtype=np.float32) * 0.25
            check(nrt.nrt_tensor_write(t, data.ctypes.data, C.c_size_t(0), C.c_size_t(ti.size)), "w")
            check(nrt.nrt_add_tensor_to_tensor_set(ins, ti.name, t), "add")
        else:
            out_t.append((t, ti.size))
            check(nrt.nrt_add_tensor_to_tensor_set(outs, ti.name, t), "add")
    lat = []
    status = 0
    for _ in range(n):
        t0 = time.perf_counter()
        status = nrt.nrt_execute(model, ins, outs)
        lat.append((time.perf_counter() - t0) * 1e6)
        if status:
            break
    out = None
    if out_t and not status:
        t, size = out_t[0]
        buf = np.empty(size // 4, dtype=np.uint32)
        check(nrt.nrt_tensor_read(t, buf.ctypes.data, C.c_size_t(0), C.c_size_t(size)), "read")
        out = buf
    nrt.nrt_unload(model)
    return lat, status, out


def main():
    n = int(sys.argv[1])
    check(nrt.nrt_init(1, b"", b""), "nrt_init")
    for p in sys.argv[2:]:
        lat, status, _ = run_one(p, n)
        a = np.array(lat)
        print(json.dumps({"neff": p, "status": status, "n": len(a), "median_us": round(float(np.median(a)), 1),
                          "p10_us": round(float(np.percentile(a, 10)), 1),
                          "p90_us": round(float(np.percentile(a, 90)), 1),
                          "first_us": [round(x, 1) for x in lat[:6]]}), flush=True)
    nrt.nrt_close()


if __name__ == "__main__":
    main()
