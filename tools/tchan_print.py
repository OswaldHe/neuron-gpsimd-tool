# Run a NEFF whose GPSIMD kernel transmits a NUL-terminated message over the timing channel
# (protocol in kl/tchan.h, 2 symbols per byte) and print it.  usage: tchan_print.py file.neff [maxlen]
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_neff as B  # noqa: E402
import tchan_run as T  # noqa: E402


def main():
    path = sys.argv[1]
    maxlen = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    B.check(B.nrt.nrt_init(1, b"", b""), "nrt_init")
    try:
        lat = T.collect(path, 2 * maxlen, 3)
    finally:
        B.nrt.nrt_close()
    levels, syms, margin = T.decode(lat)
    data = bytes((syms[k] << 4) | syms[k + 1] for k in range(0, len(syms) - 1, 2))
    text = data.split(b"\0")[0].decode("ascii", "replace")
    print(f"[timing channel: min margin {min(margin):.0f}us]")
    print(f"[GPSIMD] {text}")


if __name__ == "__main__":
    main()
