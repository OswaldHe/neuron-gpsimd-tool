"""Figures of BLOG.md.  python figs/src/make_figs.py  (writes figs/*.png)

Figure 2 (fig02_pipeline.png) is figures/memory_manage_flow.pdf of arXiv:2603.29002, rendered at 300 dpi.
Figure 5 (fig05_trainium_arch.png) is the AWS Neuron documentation diagram, copied from neuron-neff-doc/figs.

Figures 3-4 use figs/data/engram_gpu_fpga.csv (GPU-FPGA Engram experiment of HeteroLLM, Engram reference configuration).
The other numbers come from nkiopt/README.md (median total_exec_time of 5 profiled executions) and the timeline CSVs
in figs/data/ (extract_profile.py on neuron-explorer JSON exports of one profiled execution).
Diagrams use few words, symbols and large fonts; details live in the blog text.
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Patch, Polygon, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.dirname(HERE)
DATA = os.path.join(FIGS, "data")

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
NKI, GP, DMA, OTHER = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
NKI_L, GP_L, DMA_L, OTHER_L = "#e3eefc", "#fde5da", "#d9f2e8", "#f0efec"
GOOD, BAD = "#1baf7a", "#e34948"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": SURF,
    "figure.facecolor": SURF, "savefig.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 13, "xtick.labelsize": 12.5, "ytick.labelsize": 12.5, "legend.fontsize": 12.5,
})


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name), dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", name)


def box(ax, x, y, w, h, text, fc, ec, tc=INK, fs=14, bold=False, r=0.12, lw=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
                weight="bold" if bold else "normal", linespacing=1.3)


def doc(ax, x, y, w, h, text, fc, ec, fs=14, c=0.22):
    """A file: a rectangle with a folded corner."""
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h - c), (x + w - c, y + h), (x, y + h)],
                         closed=True, fc=fc, ec=ec, lw=1.6))
    ax.plot([x + w - c, x + w - c, x + w], [y + h, y + h - c, y + h - c], color=ec, lw=1.2)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.3)


def badge(ax, x, y, n, r=0.19):
    ax.add_patch(Circle((x, y), r, fc=INK, ec=SURF, lw=1.5, zorder=5))
    ax.text(x, y, str(n), ha="center", va="center", fontsize=12.5, color=SURF, weight="bold", zorder=6)


def arrow(ax, x0, y0, x1, y1, color=INK2, lw=1.8, style="-|>", ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0,
                                mutation_scale=16))


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def legend_row(ax, x, y, items, fs=12.5, gap=0.35):
    """Inline legend for diagrams: [(face, edge, label), ...] left to right."""
    for fc, ec, lab in items:
        ax.add_patch(FancyBboxPatch((x, y - 0.11), 0.32, 0.22, boxstyle="round,pad=0,rounding_size=0.05",
                                    fc=fc, ec=ec, lw=1.4))
        ax.text(x + 0.45, y, lab, va="center", fontsize=fs, color=INK)
        x += 0.45 + 0.105 * fs / 12.5 * len(lab) + gap


# ----------------------------------------------------------------------------------------------- figure 1
def fig_speedups():
    # GPSIMD + NKI over the hand-optimized NKI port, one NeuronCore (median total_exec_time, us).
    # Whole Engram per step = n-gram hashing (ids of both Engram layers) + read-out of 2 layers
    # (table retrieval + fp8 GEMM + gate), summed from the separate measurements.
    hash_nki, hash_gp = [86.95, 531.1, 1055.6], [48.45, 449.7, 890.6]
    ret_nki, ret_gp = [62.1, 506.9, 980.5], [24.7, 102.4, 200.7]
    ro_nki, ro_gp = [697.1, 3362.9, 6766.4], [681.0, 2968.9, 5926.4]
    whole = [(h + 2 * r) / (hg + 2 * rg) for h, r, hg, rg in zip(hash_nki, ro_nki, hash_gp, ro_gp)]
    groups = ["Decode", "Prefill, 1024 tokens", "Prefill, 2048 tokens"]
    series = [  # term of the high-level summary, color, speedup per group
        ("n-gram hashing", NKI, [n / g for n, g in zip(hash_nki, hash_gp)]),
        ("table retrieval", GP, [n / g for n, g in zip(ret_nki, ret_gp)]),
        ("whole Engram (n-gram hashing + table retrieval + attention + gate)", DMA, whole),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    h = 0.27
    for k, (name, color, vals) in enumerate(series):
        for g, v in enumerate(vals):
            y = (len(groups) - 1 - g) + (1 - k) * h
            ax.barh(y, v, height=h, color=color, edgecolor=SURF, linewidth=1.5, label=name if g == 0 else None)
            ax.text(v + 0.07, y, f"{v:.2f}x", ha="left", va="center", fontsize=13.5, weight="bold", color=INK)
    ax.axvline(1.0, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=3)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups[::-1], fontsize=13.5, color=INK)
    ax.set_ylim(-0.55, len(groups) - 0.5)
    ax.set_xlim(0, 5.6)
    ax.set_xticks(range(0, 6))
    ax.set_xticklabels([f"{t}x" for t in range(0, 6)])
    ax.set_xlabel("Speedup over baseline without GPSIMD custom op")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=1, frameon=False, fontsize=12.5,
              handlelength=1.2, borderaxespad=0, labelspacing=0.35)
    save(fig, "fig01_speedups.png")


# ----------------------------------------------------------------------------------------------- figure 3
def fig_fpga_split():
    """GPU-FPGA split of the Engram module (Engram reference configuration, HeteroLLM kernels/deepseek_engram)."""
    fig, ax = canvas(12.6, 3.9)
    yb, hb = 1.05, 1.35
    ym = yb + hb / 2
    # devices
    box(ax, 1.2, 0.5, 5.85, 3.0, "", "#fff8f4", GP, r=0.2, lw=1.8)
    ax.text(1.42, 3.12, "FPGA", fontsize=16, weight="bold", va="center")
    box(ax, 8.05, 0.5, 3.55, 3.0, "", "#f5f9fe", NKI, r=0.2, lw=1.8)
    ax.text(8.27, 3.12, "GPU", fontsize=16, weight="bold", va="center")
    fpga = [(1.45, "hash\n× ⊕ mod"), (3.35, "table\nlookup ▦"), (5.25, "K/V proj.\nW · e")]
    gpu = [(8.25, "gate\nσ(h·k)·v"), (9.95, "conv\n+ residual")]
    for x, t in fpga:
        box(ax, x, yb, 1.55, hb, t, "#ffffff", GP, fs=14.5)
    for x, t in gpu:
        box(ax, x, yb, 1.45, hb, t, "#ffffff", NKI, fs=14.5)
    arrow(ax, 3.0, ym, 3.35, ym)
    arrow(ax, 4.9, ym, 5.25, ym)
    arrow(ax, 9.7, ym, 9.95, ym)
    # PCIe hand-off
    arrow(ax, 6.8, ym, 8.25, ym, color=INK, lw=3)
    ax.text(7.52, ym + 0.2, "PCIe", ha="center", va="bottom", fontsize=14.5, weight="bold")
    # inputs and output
    ax.text(0.1, ym, "tokens", ha="left", va="center", fontsize=14)
    arrow(ax, 0.95, ym, 1.45, ym)
    arrow(ax, 8.975, 0.08, 8.975, yb)
    ax.text(8.8, 0.25, "h", ha="right", va="center", fontsize=15, style="italic")
    arrow(ax, 11.4, ym, 11.85, ym)
    ax.text(11.92, ym, "attn", ha="left", va="center", fontsize=14)
    save(fig, "fig03_fpga_split.png")


# ----------------------------------------------------------------------------------------------- figure 3
def load_fpga():
    with open(os.path.join(DATA, "engram_gpu_fpga.csv")) as f:
        rows = list(csv.DictReader(l for l in f if not l.startswith("#")))
    return [{k: float(v) for k, v in r.items()} for r in rows]


def fig_fpga_results():
    """Only the token counts where the GPU + 2 FPGAs estimate is faster than the GPU alone."""
    rows = [r for r in load_fpga() if r["engram_gpu"] > r["boards2"]]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    w = 0.36
    top = max(r["engram_gpu"] for r in rows)
    for i, r in enumerate(rows):
        g, f2 = r["engram_gpu"], r["boards2"]
        ax.bar(i - w / 2, g, width=w, color=NKI, edgecolor=SURF, linewidth=2)
        ax.bar(i + w / 2, f2, width=w, color=GP, edgecolor=SURF, linewidth=2)
        sp = g / f2
        ax.text(i, max(g, f2) + top * 0.04, f"{sp:.2f}x" if sp < 1.5 else f"{sp:.1f}x",
                ha="center", va="bottom", fontsize=15, color=INK, weight="bold")
    ax.set_ylim(0, top * 1.35)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{int(r['tokens']):,}" for r in rows], color=INK, fontsize=14)
    ax.set_xlabel("Tokens")
    ax.set_ylabel("Engram time (ms)")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.legend(handles=[Patch(color=NKI, label="GPU only"), Patch(color=GP, label="GPU + 2 FPGAs")],
              loc="upper left", frameon=False, ncol=2, fontsize=13.5)
    save(fig, "fig04_fpga_results.png")


# ----------------------------------------------------------------------------------------------- figure 5
def fig_toolchain():
    fig, ax = canvas(13.4, 4.4)
    w, h = 1.9, 0.95
    yt, ym, yb = 3.1, 1.72, 0.35
    tool = dict(fc="#ffffff", ec=INK2, r=0.3)
    # NKI lane
    doc(ax, 0.45, yt, w, h, "kernel.py\ngp.call(…)", NKI_L, NKI)
    box(ax, 2.85, yt, w, h, "⚙ neuronx-cc", fs=14, **tool)
    doc(ax, 5.25, yt, w, h, "kernel.neff", NKI_L, NKI)
    arrow(ax, 2.35, yt + h / 2, 2.85, yt + h / 2)
    arrow(ax, 4.75, yt + h / 2, 5.25, yt + h / 2)
    # operator lane
    doc(ax, 0.45, yb, w, h, "op.cpp", GP_L, GP)
    box(ax, 2.85, yb, w, h, "⚙ gpsimd-cc", fs=14, **tool)
    doc(ax, 5.25, yb, w, h, "ops.so", GP_L, GP)
    arrow(ax, 2.35, yb + h / 2, 2.85, yb + h / 2)
    arrow(ax, 4.75, yb + h / 2, 5.25, yb + h / 2)
    # tile addresses from the compiled NEFF into the operator build
    arrow(ax, 5.55, yt, 4.45, yb + h)
    ax.text(5.15, 2.2, "tile\naddresses", fontsize=12.5, color=INK2, va="center", ha="left")
    # inject, final NEFF, run
    box(ax, 7.65, ym, 1.7, h, "⚙ inject", fs=14, **tool)
    arrow(ax, 7.15, yt + 0.3, 8.6, ym + h)
    arrow(ax, 7.15, yb + h - 0.3, 8.6, ym)
    doc(ax, 9.75, ym, 2.0, h, "final.neff\n+ KERNEL_CALL", NKI_L, NKI, fs=13.5)
    arrow(ax, 9.35, ym + h / 2, 9.75, ym + h / 2)
    box(ax, 12.15, ym, 1.15, h, "⚡ run", fs=14, **tool)
    arrow(ax, 11.75, ym + h / 2, 12.15, ym + h / 2)
    # step numbers (Section 3.1)
    for n, (x, y) in enumerate([(0.45, yb + h), (0.45, yt + h), (2.85, yb + h), (7.65, ym + h), (12.15, ym + h)], 1):
        badge(ax, x, y, n)
    save(fig, "fig06_toolchain.png")


# ----------------------------------------------------------------------------------------------- figure 6
def fig_engram_split():
    fig, ax = canvas(12.2, 5.3)
    heads = ["Prepare Memory", "Compute Relevancy", "Retrieval", "Apply to Inference"]
    cw, x0s = 2.85, [0.1, 3.1, 6.1, 9.1]
    for t, x in zip(heads, x0s):
        box(ax, x, 0.75, cw, 4.45, "", "#ffffff", GRID, r=0.15)
        ax.text(x + cw / 2, 4.88, t, ha="center", va="center", fontsize=14.5, weight="bold")
    bw = 2.35
    bx = [x + (cw - bw) / 2 for x in x0s]
    box(ax, bx[0], 3.35, bw, 1.0, "table ▦\n384M rows", OTHER_L, OTHER)
    box(ax, bx[0], 1.9, bw, 1.0, "n-gram\nwindows", NKI_L, NKI)
    box(ax, bx[1], 1.9, bw, 2.45, "n-gram hash\n\n× ⊕ mod p", GP_L, GP, fs=14.5)
    box(ax, bx[2], 3.35, bw, 1.0, "hit compaction\n1.5% kept", GP_L, GP)
    box(ax, bx[2], 2.15, bw, 0.8, "gather hits", NKI_L, NKI)
    box(ax, bx[2], 1.0, bw, 0.75, "all-reduce", OTHER_L, OTHER)
    box(ax, bx[3], 3.35, bw, 1.0, "fp8 GEMM\n→ keys, value", NKI_L, NKI)
    box(ax, bx[3], 1.9, bw, 1.0, "gate", NKI_L, NKI)
    arrow(ax, bx[0] + bw, 2.4, bx[1], 2.4)
    arrow(ax, bx[1] + bw, 3.85, bx[2], 3.85)
    arrow(ax, bx[2] + bw / 2, 3.35, bx[2] + bw / 2, 2.95)
    arrow(ax, bx[2] + bw / 2, 2.15, bx[2] + bw / 2, 1.75)
    arrow(ax, bx[2] + bw, 1.375, bx[3], 3.6)
    arrow(ax, bx[3] + bw / 2, 3.35, bx[3] + bw / 2, 2.9)
    legend_row(ax, 0.2, 0.3, [(GP_L, GP, "GPSIMD custom op"), (NKI_L, NKI, "NKI"),
                              (OTHER_L, OTHER, "offline / collective")], fs=13)
    save(fig, "fig07_engram_split.png")


# ----------------------------------------------------------------------------------------------- figure 7
def fig_gather():
    import numpy as np
    rng = np.random.default_rng(3)
    nrow, ncol = 16, 24
    hits = np.zeros((nrow, ncol), bool)
    for r in range(nrow):        # rank 21 holds rows of hash columns 7 and 8 only (~0.375 hits a token)
        for c in (7, 8):
            if rng.random() < (0.19 if c == 7 else 0.185):
                hits[r, c] = True
    fig = plt.figure(figsize=(11.6, 5.0))
    panels = [("NKI only", "24,576 rows fetched  ·  192 DMAs", "#bcd6f6", NKI,
               [("#bcd6f6", "fetched, masked ×0"), (NKI, "hit")]),
              ("GPSIMD + NKI", "372 rows fetched  ·  3 gathers", "#f1f0ec", GP,
               [("#f1f0ec", "skipped"), (GP, "hit")])]
    for side, (title, stat, miss, hit, leg) in enumerate(panels):
        ax = fig.add_axes([0.01 + side * 0.5, 0.02, 0.47, 0.9])
        ax.set_xlim(-1.2, ncol + 0.2)
        ax.set_ylim(-5.0, nrow + 2.4)
        ax.axis("off")
        ax.text(-1.0, nrow + 1.5, title, fontsize=17, weight="bold", va="center")
        for r in range(nrow):
            for c in range(ncol):
                ax.add_patch(Rectangle((c + 0.07, nrow - 1 - r + 0.07), 0.86, 0.86,
                                       fc=hit if hits[r, c] else miss, ec="none"))
        ax.text(ncol / 2, -0.75, "24 hash columns  →", ha="center", va="center", fontsize=13, color=INK2)
        ax.text(-0.5, nrow / 2, "tokens  →", rotation=270, ha="center", va="center", fontsize=13, color=INK2)
        ax.text(-1.0, -2.35, stat, fontsize=15.5, weight="bold", va="center")
        x = -1.0
        for fc, lab in leg:
            ax.add_patch(Rectangle((x, -4.05), 0.8, 0.8, fc=fc, ec="none"))
            ax.text(x + 1.2, -3.65, lab, fontsize=13, va="center")
            x += 1.2 + 0.5 * len(lab) + 1.0
    save(fig, "fig08_gather.png")


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
    lanes = LANES + [("DMA", None, None)]
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
            ax.broken_barh(spans, (y - 0.36, 0.72), facecolors=col, edgecolor="none")
    ax.set_yticks(range(n))
    ax.set_yticklabels([l[0] for l in lanes][::-1], fontsize=13, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.7, n - 0.3)
    end = max(r["start_us"] + r["dur_us"] for r in rows)
    ax.axvline(end, color=INK, lw=1.4, ls=(0, (3, 2)))
    ax.text(end + xmax * 0.008, n - 1, f"{end:.0f} µs", fontsize=14, weight="bold", va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.15", fc=SURF, ec="none"))
    ax.set_title(title, loc="left", fontsize=15, weight="bold", color=INK, pad=12)
    ax.xaxis.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def fig_timeline(prefix, out, titles, xmax=None):
    b = load_tl(f"{prefix}_baseline.csv")
    g = load_tl(f"{prefix}_gpsimd.csv")
    if xmax is None:
        xmax = 1.1 * max(r["start_us"] + r["dur_us"] for r in b + g)
    fig, axs = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True)
    draw_tl(axs[0], b, titles[0], xmax)
    draw_tl(axs[1], g, titles[1], xmax)
    axs[1].set_xlabel("Time (µs)")
    fig.tight_layout(h_pad=1.2)
    save(fig, out)


# ----------------------------------------------------------------------------------------------- figure 11
def fig_amdahl():
    data = [  # label, retrieval stage measured alone, total
        ("NKI, T=1024", 506.9, 3362.9), ("GPSIMD + NKI, T=1024", 102.4, 2968.9),
        ("NKI, T=2048", 980.5, 6766.4), ("GPSIMD + NKI, T=2048", 200.7, 5926.4),
    ]
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    ys = [3.3, 2.5, 1.0, 0.2]
    for (lab, g, t), y in zip(data, ys):
        ax.barh(y, g, height=0.62, color=GP, edgecolor=SURF, linewidth=2)
        ax.barh(y, t - g, left=g, height=0.62, color=NKI, edgecolor=SURF, linewidth=2)
        ax.text(t + 80, y, f"{t:,.0f} µs ({100 * g / t:.0f}%)", va="center", fontsize=13.5, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([d[0] for d in data], fontsize=13.5, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 9000)
    ax.set_xlabel("Read-out time on one NeuronCore (µs)")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(color=GP, label="table retrieval"), Patch(color=NKI, label="attention + gate")],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=13, ncol=2)
    save(fig, "fig12_amdahl.png")


if __name__ == "__main__":
    fig_speedups()
    fig_fpga_split()
    fig_fpga_results()
    fig_toolchain()
    fig_engram_split()
    fig_gather()
    fig_amdahl()
    if os.path.exists(os.path.join(DATA, "hash_decode_bs10_gpsimd.csv")):
        fig_timeline("hash_decode_bs10", "fig09_timeline_hash.png", ["NKI only", "GPSIMD + NKI"])
    if os.path.exists(os.path.join(DATA, "gather_t1024_gpsimd.csv.gz")):
        fig_timeline("gather_t1024", "fig10_timeline_gather.png", ["NKI only", "GPSIMD + NKI"])
    if os.path.exists(os.path.join(DATA, "readout_t1024_gpsimd.csv.gz")):
        fig_timeline("readout_t1024", "fig11_timeline_readout.png", ["NKI only", "GPSIMD + NKI"])
