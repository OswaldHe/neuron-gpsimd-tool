# Run a timing-channel NEFF (protocol in kl/tchan.h) and decode 32-bit words.
# usage: tchan_run.py file.neff nwords [repeats]
# Each repeat reloads the NEFF (fresh kernel state); per-execution medians across repeats are used.
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_neff as B  # noqa: E402

REPEAT = 3                    # TCHAN_REPEAT
PRE = 1 + 16 * REPEAT         # TCHAN_PREAMBLE


def collect(path, nsyms, reps):
    runs = []
    for _ in range(reps):
        lat, status, _ = B.run_one(path, PRE + REPEAT * nsyms)
        if status:
            raise RuntimeError(f"execution status {status} after {len(lat)} executions")
        runs.append(lat)
    return np.median(np.array(runs), axis=0)


def decode(lat):
    lv = np.array(lat[1:PRE]).reshape(16, REPEAT)
    levels = np.median(lv, axis=1)
    d = np.array(lat[PRE:])
    data = np.median(d[: len(d) // REPEAT * REPEAT].reshape(-1, REPEAT), axis=1)
    syms = [int(np.argmin(np.abs(levels - x))) for x in data]
    margin = [float(np.min(np.abs(np.delete(levels, s) - x)) - abs(levels[s] - x)) for s, x in zip(syms, data)]
    return levels, syms, margin


def to_words(syms):
    return [int("".join(f"{s:x}" for s in syms[i:i + 8]), 16) for i in range(0, len(syms) - 7, 8)]


def read_words(path, nwords, reps=3):
    B.check(B.nrt.nrt_init(1, b"", b""), "nrt_init")
    try:
        lat = collect(path, 8 * nwords, reps)
    finally:
        B.nrt.nrt_close()
    levels, syms, margin = decode(lat)
    return to_words(syms), levels, margin


def main():
    path, nwords = sys.argv[1], int(sys.argv[2])
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    words, levels, margin = read_words(path, nwords, reps)
    print(f"levels(us): {[round(x) for x in levels]}  min margin {min(margin):.0f}us")
    for i, w in enumerate(words):
        print(f"word[{i}] = 0x{w:08x}")


if __name__ == "__main__":
    main()
