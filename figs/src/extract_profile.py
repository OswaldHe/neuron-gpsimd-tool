"""Turn a neuron-explorer JSON export into a compact timeline CSV for the blog figures.

    neuron-explorer view -n final.neff -s profile.ntff --output-format json --output-file p.json
    python extract_profile.py p.json out.csv

Rows: kind (inst / dma), lane (engine name or DMA queue), opcode, start_us, dur_us; times relative to the
first recorded instruction.
"""
import csv
import json
import sys


def main(src, dst):
    d = json.load(open(src))
    ins = d["instruction"]
    t0 = min(i["timestamp"] for i in ins)
    rows = []
    for i in ins:
        rows.append(("inst", i["subgroup"], i["opcode"], (i["timestamp"] - t0) / 1000, i["duration"] / 1000))
    for x in d["dma"]:
        rows.append(("dma", x["subgroup"].split("-")[0], x.get("variable", ""), (x["timestamp"] - t0) / 1000,
                     x["duration"] / 1000))
    rows.sort(key=lambda r: r[3])
    with open(dst, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "lane", "opcode", "start_us", "dur_us"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], f"{r[3]:.3f}", f"{r[4]:.3f}"])
    print(f"{dst}: {len(rows)} rows, span {max(r[3] + r[4] for r in rows):.1f} us")


if __name__ == "__main__":
    main(*sys.argv[1:3])
