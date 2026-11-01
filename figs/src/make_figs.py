"""Figures of BLOG.md.  python figs/src/make_figs.py  (writes figs/*.png)

Numbers come from nkiopt/README.md (median total_exec_time of 5 profiled executions) and the timeline CSVs
in figs/data/ (extract_profile.py on neuron-explorer JSON exports of one profiled execution).
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.dirname(HERE)
DATA = os.path.join(FIGS, "data")

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
NKI, GP, DMA, OTHER = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
NKI_L, GP_L, DMA_L, OTHER_L = "#e3eefc", "#fde5da", "#d9f2e8", "#f0efec"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": SURF,
    "figure.facecolor": SURF, "savefig.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name), dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", name)


def box(ax, x, y, w, h, text, fc, ec, tc=INK, fs=10, bold=False, r=0.12, lw=1.2, ha="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))
    tx = x + w / 2 if ha == "center" else x + 0.15
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, color=tc,
            weight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, color=INK2, lw=1.3, style="-|>"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=0, shrinkB=0))


def canvas(w, h, scale=1.0):
    fig = plt.figure(figsize=(w * scale, h * scale))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


# ----------------------------------------------------------------------------------------------- figure 1
def fig_speedups():
    rows = [  # label, stage speedup (GPSIMD-assisted stage alone) or None, whole-computation speedup
        ("Hashing · decode, 10 sequences (T=60)", None, 86.95 / 48.45),
        ("Hashing · prefill chunk (T=1024)", None, 531.1 / 449.7),
        ("Hashing · prefill chunk (T=2048)", None, 1055.6 / 890.6),
        ("Read-out · decode, 10 sequences (T=60)", 62.1 / 24.7, 697.1 / 681.0),
        ("Read-out · prefill chunk (T=1024)", 506.9 / 102.4, 3362.9 / 2968.9),
        ("Read-out · prefill chunk (T=2048)", 980.5 / 200.7, 6766.4 / 5926.4),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    h = 0.36
    for i, (lab, st, wh) in enumerate(rows):
        y = len(rows) - 1 - i
        if st is None:
            ax.barh(y, wh, height=h, color=NKI, edgecolor=SURF, linewidth=2)
            ax.text(wh + 0.06, y, f"{wh:.2f}x", va="center", fontsize=10, color=INK)
        else:
            ax.barh(y + h / 2, st, height=h, color=GP, edgecolor=SURF, linewidth=2)
            ax.barh(y - h / 2, wh, height=h, color=NKI, edgecolor=SURF, linewidth=2)
            ax.text(st + 0.06, y + h / 2, f"{st:.2f}x", va="center", fontsize=10, color=INK)
            ax.text(wh + 0.06, y - h / 2, f"{wh:.2f}x", va="center", fontsize=10, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=10.5, color=INK)
    ax.axvline(1.0, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(1.03, len(rows) - 0.45, "hand-optimized NKI = 1.0x", fontsize=9, color=INK2, va="bottom")
    ax.set_xlim(0, 5.6)
    ax.set_xlabel("speedup over the hand-optimized NKI port (median total_exec_time)")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=NKI, label="whole Engram computation"),
                       Patch(color=GP, label="retrieval (gather) stage alone")],
              loc="upper right", frameon=False, fontsize=10)
    save(fig, "fig01_speedups.png")


# ----------------------------------------------------------------------------------------------- figure 2
def fig_neuroncore():
    fig, ax = canvas(13.2, 6.2)
    # NeuronCore
    box(ax, 0.2, 0.2, 6.0, 5.8, "", "#ffffff", GRID, r=0.2)
    ax.text(0.45, 5.62, "One Trainium 3 NeuronCore", fontsize=12.5, weight="bold", va="center")
    engines = [("Tensor Engine\n128 x 128\nsystolic array", NKI_L, NKI), ("Vector Engine", NKI_L, NKI),
               ("Scalar Engine", NKI_L, NKI), ("Sync Engine", NKI_L, NKI)]
    for k, (t, fc, ec) in enumerate(engines):
        box(ax, 0.45 + k * 1.42, 4.15, 1.3, 1.0, t, fc, ec, fs=9.2)
    box(ax, 0.45, 2.95, 5.55, 0.95, "GPSIMD Engine (\"Pool\"): 8 general-purpose cores", GP_L, GP, fs=10.5, bold=True)
    box(ax, 0.45, 1.6, 3.9, 1.05, "SBUF: on-chip SRAM, 128 partitions\n(operands of every engine)", OTHER_L, OTHER, fs=9.5)
    box(ax, 4.5, 1.6, 1.5, 1.05, "PSUM\n(matmul\naccumulators)", OTHER_L, OTHER, fs=9)
    box(ax, 0.45, 0.45, 5.55, 0.85, "DMA engines and queues  <->  HBM", DMA_L, DMA, fs=10)
    # zoom lines
    ax.plot([6.0, 6.75], [3.9, 5.75], color=GP, lw=1, ls=(0, (3, 3)))
    ax.plot([6.0, 6.75], [2.95, 0.35], color=GP, lw=1, ls=(0, (3, 3)))
    # GPSIMD detail
    box(ax, 6.75, 0.2, 6.25, 5.8, "", "#fffaf7", GP, r=0.2)
    ax.text(7.0, 5.62, "Inside the GPSIMD Engine", fontsize=12.5, weight="bold", va="center")
    for c in range(8):
        cx = 7.0 + (c % 4) * 1.48
        cy = 4.25 if c < 4 else 3.05
        box(ax, cx, cy, 1.36, 1.0, f"core {c}\nXtensa, 32-bit\n1.2 GHz", "#ffffff", GP, fs=8.6)
    lines = [
        "each core:  256 KB local data RAM,  one iDMA channel;  custom code up to 64 KB",
        "integer ISA with 32 x 32 -> 64-bit multiply (MULL / MULUH);  no usable FPU or vector unit",
        "SBUF window at core address 0xD0000000 (partition p at + p x 0x40000)",
        "HBM window at 0xE0000000 (512 MB, set up by the host runner)",
        "NKI uses the cores only through built-in instructions",
        "(tensor ops on engine=gpsimd, iota, indirect-DMA descriptor generation)",
    ]
    for k, t in enumerate(lines):
        ax.text(7.0, 2.6 - k * 0.38, t, fontsize=9.3, color=INK if k < 4 else INK2, va="center")
    save(fig, "fig02_neuroncore.png")


# ----------------------------------------------------------------------------------------------- figure 3
def fig_toolchain():
    fig, ax = canvas(14.9, 6.0)
    W, H = 2.45, 1.05
    y1, y2, y3 = 4.55, 2.55, 0.45
    X = [0.2, 3.2, 6.2, 9.2, 12.2]
    # row 1: the NKI kernel
    box(ax, X[0], y1, W, H, "kernel.py\nNKI kernel + gp.call(...)", NKI_L, NKI, fs=9.5)
    box(ax, X[1], y1, W, H, "neuronx-cc\n(NKI compiler)", "#ffffff", INK2, fs=9.5)
    box(ax, X[2], y1, W, H, "kernel.neff\nPool0.bin holds the\ngp.call marker instructions", NKI_L, NKI, fs=9.0)
    box(ax, X[3], y1, W, H, "gpsimd-neff analyze\nfind the markers, read\neach tile's SBUF location", "#ffffff", INK2, fs=9.0)
    for k in range(3):
        arrow(ax, X[k] + W, y1 + H / 2, X[k + 1], y1 + H / 2)
    # row 2: the operator
    box(ax, X[0], y2, W, H, "op.cpp\nGPSIMD_OP(name) { ... }", GP_L, GP, fs=9.5)
    box(ax, X[1], y2, W, H, "gpsimd-cc\nxt-clang (base ISA) + runtime\n+ call-site table", "#ffffff", INK2, fs=8.9)
    box(ax, X[2], y2, W, H, "sites_resolved.json\npartitions, byte offset,\ndtype, access pattern", OTHER_L, OTHER, fs=9.0)
    arrow(ax, X[0] + W, y2 + H / 2, X[1], y2 + H / 2)
    arrow(ax, X[2], y2 + H / 2, X[1] + W, y2 + H / 2)
    arrow(ax, X[3] + W / 2, y1, X[2] + W * 0.8, y2 + H)
    # row 3: injection and run
    box(ax, X[1], y3, W, H, "ops.so\nXtensa ELF library", GP_L, GP, fs=9.5)
    box(ax, X[2], y3, W, H, "gpsimd-neff inject\nkernel.neff: markers ->\none KERNEL_CALL a site", "#ffffff", INK2, fs=8.9)
    box(ax, X[3], y3, W, H, "final.neff\nKERNEL_CALL (0x85) in Pool0.bin,\ngpsimd_kernels_def.json, ops.so", NKI_L, NKI, fs=8.4)
    box(ax, X[4], y3, W, H, "gpsimd-run (libnrt)\nmaps SBUF / HBM windows,\nruns, checks, reads receipts", "#ffffff", INK2, fs=8.6)
    arrow(ax, X[1] + W / 2, y2, X[1] + W / 2, y3 + H)
    arrow(ax, X[2] + W / 2, y2, X[2] + W / 2, y3 + H)
    arrow(ax, X[1] + W, y3 + H / 2, X[2], y3 + H / 2)
    arrow(ax, X[2] + W, y3 + H / 2, X[3], y3 + H / 2)
    arrow(ax, X[3] + W, y3 + H / 2, X[4], y3 + H / 2)
    ax.text(X[0], y3 + H / 2, "gpsimd-build runs\nall steps; gp.compile\ndoes the same from\nPython", fontsize=9.3,
            color=INK2, va="center")
    save(fig, "fig03_toolchain.png")


# ----------------------------------------------------------------------------------------------- figure 4
def fig_engram_split():
    fig, ax = canvas(14.4, 6.4)
    cols = [("1. Prepare Memory", 0.2), ("2. Compute Relevancy", 3.75), ("3. Retrieval", 7.3), ("4. Apply to Inference", 10.85)]
    cw = 3.35
    for t, x in cols:
        box(ax, x, 0.25, cw, 5.6, "", "#ffffff", GRID, r=0.15)
        ax.text(x + cw / 2, 5.5, t, ha="center", va="center", fontsize=12, weight="bold")
    # prepare
    box(ax, 0.4, 3.75, 2.95, 1.35, "Engram table, trained offline\n384M rows x 256 bf16,\nsharded by row over 64 ranks\n(6.0M rows = 3.1 GB a rank)", OTHER_L, OTHER, fs=9)
    box(ax, 0.4, 1.95, 2.95, 1.5, "n-gram windows\ncompressed id of each token\nand the 3 tokens before it\n(token map + token history)", NKI_L, NKI, fs=9)
    ax.text(1.875, 1.7, "NKI: indirect DMA + Vector", ha="center", fontsize=8.8, color=NKI)
    # relevancy
    box(ax, 3.95, 1.95, 2.95, 3.15, "n-gram hash\nper 2-, 3-, 4-gram:\nid x 64-bit multiplier,\nXOR, mod a prime\n(8 primes per n),\n+ column offset\n\n-> 24 row ids per token\n    per Engram layer", GP_L, GP, fs=9.2)
    ax.text(5.425, 1.7, "GPSIMD custom op", ha="center", fontsize=8.8, color=GP, weight="bold")
    # retrieval
    box(ax, 7.5, 3.95, 2.95, 1.15, "hit compaction\nkeep the ~1.5 % of ids\nthat this rank holds", GP_L, GP, fs=9.2)
    ax.text(8.975, 3.72, "GPSIMD custom op", ha="center", fontsize=8.8, color=GP, weight="bold")
    box(ax, 7.5, 2.3, 2.95, 1.15, "gather the hit rows from\nthis rank's table shard\ninto a zero-filled emb\n[T, 24 x 256]", NKI_L, NKI, fs=9.2)
    ax.text(8.975, 2.07, "NKI: indirect DMA", ha="center", fontsize=8.8, color=NKI)
    box(ax, 7.5, 0.6, 2.95, 1.05, "all-reduce of emb\nover the 64 ranks", OTHER_L, OTHER, fs=9.2)
    # apply
    box(ax, 11.05, 3.6, 2.95, 1.5, "fp8 MX GEMM\nemb [T, 6144] x wkv\n-> [T, 5 x 5120]:\n4 keys + 1 value", NKI_L, NKI, fs=9.2)
    ax.text(12.525, 3.37, "NKI: Tensor Engine", ha="center", fontsize=8.8, color=NKI)
    box(ax, 11.05, 1.55, 2.95, 1.5, "gate: per residual stream,\nnormalized dot product of\nhidden state and key,\nscales the value added", NKI_L, NKI, fs=9.2)
    ax.text(12.525, 1.32, "NKI: Vector + Scalar", ha="center", fontsize=8.8, color=NKI)
    arrow(ax, 3.35, 2.7, 3.95, 2.7)
    arrow(ax, 6.9, 4.5, 7.5, 4.5)
    arrow(ax, 10.1, 3.95, 10.1, 3.45)
    arrow(ax, 10.1, 2.3, 10.1, 1.65)
    arrow(ax, 10.45, 1.1, 11.05, 4.0)
    arrow(ax, 13.65, 3.6, 13.65, 3.05)
    save(fig, "fig04_engram_split.png")


# ----------------------------------------------------------------------------------------------- figure 6
def fig_gather():
    import numpy as np
    rng = np.random.default_rng(3)
    nrow, ncol = 16, 24
    hits = np.zeros((nrow, ncol), bool)
    for r in range(nrow):        # rank 21 holds rows of hash columns 7 and 8 only (~0.375 hits a token)
        for c in (7, 8):
            if rng.random() < (0.19 if c == 7 else 0.185) * 1.0:
                hits[r, c] = True
    fig = plt.figure(figsize=(13.6, 5.6))
    for side in range(2):
        ax = fig.add_axes([0.02 + side * 0.5, 0.05, 0.46, 0.83])
        ax.set_xlim(-0.5, 30)
        ax.set_ylim(-4.6, nrow + 2.2)
        ax.axis("off")
        title = ("NKI only: fetch every lookup, mask the misses" if side == 0
                 else "GPSIMD + NKI: list the hits, fetch only those")
        ax.text(-0.3, nrow + 1.6, title, fontsize=12.5, weight="bold", va="center")
        for r in range(nrow):
            for c in range(ncol):
                y = nrow - 1 - r
                if side == 0:
                    fc = NKI if hits[r, c] else "#bcd6f6"
                else:
                    fc = GP if hits[r, c] else "#f4f3f0"
                ax.add_patch(Rectangle((c + 0.06, y + 0.06), 0.88, 0.88, fc=fc, ec="none"))
        ax.text(12, -0.75, "24 hash columns (3 n-gram sizes x 8 hashes) of one Engram layer", ha="center",
                fontsize=9.5, color=INK2)
        ax.text(-0.35, nrow / 2, "tokens", rotation=90, ha="right", va="center", fontsize=9.5, color=INK2)
        if side == 0:
            ax.text(24.6, nrow - 1.2, "light blue:\nfetched, then\nmasked to 0\n\ndark blue:\nfetched, kept", fontsize=9.2,
                    color=INK2, va="top")
            ax.text(-0.3, -2.0, "T=1024: 192 indirect DMAs x 128 rows = 24,576 row fetches,", fontsize=10.3)
            ax.text(-0.3, -3.0, "of which 372 (1.5 %) are rows this rank holds", fontsize=10.3)
            ax.text(-0.3, -4.0, "every fetched index costs ~5.5 ns of descriptor generation", fontsize=10.3, color=INK2)
        else:
            ax.text(24.6, nrow - 1.2, "orange:\nhits found by\nthe GPSIMD scan\n\nonly 2 of 24\ncolumns overlap\nthis rank's\n6.0M rows", fontsize=9.2,
                    color=INK2, va="top")
            ax.text(-0.3, -2.0, "GPSIMD writes 372 (table row, emb row) pairs + a count;", fontsize=10.3)
            ax.text(-0.3, -3.0, "NKI gathers them with 3 indirect DMAs of 128 rows", fontsize=10.3)
            ax.text(-0.3, -4.0, "into a zero-filled emb (all other rows stay 0)", fontsize=10.3, color=INK2)
    save(fig, "fig06_gather.png")


# ----------------------------------------------------------------------------------------------- timelines
LANES = [("GPSIMD custom op", "GpSimd", True), ("GPSIMD built-ins", "GpSimd", False), ("Vector", "Vector", False),
         ("Scalar", "Scalar", False), ("Tensor", "Tensor", False), ("Sync", "Sync", False)]
SKIP = {"EVENT_SEMAPHORE", "NOP", "DRAIN", "EVENT_SEMAPHORE_RANGE_CLEAR"}


def load_tl(name):
    import gzip
    path = os.path.join(DATA, name)
    f = gzip.open(path + ".gz", "rt") if os.path.exists(path + ".gz") else open(path)
    rows = list(csv.DictReader(f))
    for r in rows:
        r["start_us"] = float(r["start_us"])
        r["dur_us"] = float(r["dur_us"])
    return rows


def draw_tl(ax, rows, title, xmax):
    lanes = LANES + [("DMA transfers", None, None)]
    n = len(lanes)
    for k, (lab, eng, custom) in enumerate(lanes):
        y = n - 1 - k
        if eng is None:
            spans = [(r["start_us"], max(r["dur_us"], xmax / 900)) for r in rows if r["kind"] == "dma"]
            col = DMA
        else:
            spans = [(r["start_us"], max(r["dur_us"], xmax / 900)) for r in rows
                     if r["kind"] == "inst" and r["lane"] == eng and r["opcode"] not in SKIP
                     and ((r["opcode"] == "CUSTOM_OP") == custom)]
            col = GP if custom else NKI
        if spans:
            ax.broken_barh(spans, (y - 0.34, 0.68), facecolors=col, edgecolor="none")
        busy = sum(d for _, d in spans)
    ax.set_yticks(range(n))
    ax.set_yticklabels([l[0] for l in lanes][::-1], fontsize=9.5, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.7, n - 0.3)
    end = max(r["start_us"] + r["dur_us"] for r in rows)
    ax.axvline(end, color=INK2, lw=1, ls=(0, (3, 2)))
    ax.text(end, n - 0.35, f" ends at {end:.0f} us", fontsize=9, color=INK2, va="bottom", ha="left")
    ax.set_title(title, loc="left", fontsize=11.5, weight="bold", color=INK, pad=14)
    ax.xaxis.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def fig_timeline(prefix, out, titles, xmax=None):
    b = load_tl(f"{prefix}_baseline.csv")
    g = load_tl(f"{prefix}_gpsimd.csv")
    if xmax is None:
        xmax = 1.08 * max(r["start_us"] + r["dur_us"] for r in b + g)
    fig, axs = plt.subplots(2, 1, figsize=(11.5, 6.6), sharex=True)
    draw_tl(axs[0], b, titles[0], xmax)
    draw_tl(axs[1], g, titles[1], xmax)
    axs[1].set_xlabel("time since the first instruction of the profiled execution (us)")
    fig.tight_layout(h_pad=1.5)
    save(fig, out)


# ----------------------------------------------------------------------------------------------- figure 9
def fig_amdahl():
    data = [  # label, retrieval stage measured alone, total
        ("NKI, T=1024", 506.9, 3362.9), ("GPSIMD + NKI, T=1024", 102.4, 2968.9),
        ("NKI, T=2048", 980.5, 6766.4), ("GPSIMD + NKI, T=2048", 200.7, 5926.4),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    ys = [3.3, 2.5, 1.0, 0.2]
    for (lab, g, t), y in zip(data, ys):
        ax.barh(y, g, height=0.6, color=GP, edgecolor=SURF, linewidth=2)
        ax.barh(y, t - g, left=g, height=0.6, color=NKI, edgecolor=SURF, linewidth=2)
        ax.text(t + 60, y, f"{t:,.0f} us  (retrieval {100 * g / t:.0f} %)", va="center", fontsize=9.8, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([d[0] for d in data], fontsize=10, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 8600)
    ax.set_xlabel("Engram read-out time on one NeuronCore (us)")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GP, label="retrieval stage (gather), measured alone"),
                       Patch(color=NKI, label="the rest: fp8 GEMM + gate (unchanged NKI)")],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=9.6, ncol=2)
    save(fig, "fig09_amdahl.png")


if __name__ == "__main__":
    fig_speedups()
    fig_neuroncore()
    fig_toolchain()
    fig_engram_split()
    fig_gather()
    fig_amdahl()
    if os.path.exists(os.path.join(DATA, "hash_decode_bs10_gpsimd.csv")):
        fig_timeline("hash_decode_bs10", "fig05_timeline_hash.png",
                     ["NKI only: n-gram hashing, decode, 10 sequences (T=60)",
                      "GPSIMD + NKI: the same hashing with one GPSIMD custom op"])
    if os.path.exists(os.path.join(DATA, "gather_t1024_gpsimd.csv.gz")):
        fig_timeline("gather_t1024", "fig07_timeline_gather.png",
                     ["NKI only: Engram retrieval stage, prefill chunk (T=1024)",
                      "GPSIMD + NKI: hit compaction on GPSIMD, gather of the hits by NKI"])
    if os.path.exists(os.path.join(DATA, "readout_t1024_gpsimd.csv.gz")):
        fig_timeline("readout_t1024", "fig08_timeline_readout.png",
                     ["NKI only: whole Engram read-out, prefill chunk (T=1024)",
                      "GPSIMD + NKI: whole Engram read-out, prefill chunk (T=1024)"])
