"""Figures of BLOG.md.  python figs/src/make_figs.py  (writes figs/*.png)

Figure 2 (fig02_pipeline.png) is figures/memory_manage_flow.pdf of arXiv:2603.29002, rendered at 300 dpi.
Figure 5 (fig05_trainium_arch.png) puts two AWS Neuron documentation diagrams side by side (src/aws/): the Trainium3
device (neuroncore-v4-overview.png, Trainium3 page) and a NeuronCore (nki-trn3-arch-2.png, Trainium3 NKI architecture guide).

Figures 3-4 use figs/data/engram_gpu_fpga.csv (GPU-FPGA Engram experiment of HeteroLLM, Engram reference configuration).
The other numbers come from nkiopt/README.md (median total_exec_time of 5 profiled executions) and the timeline CSVs
in figs/data/ (extract_profile.py on neuron-explorer JSON exports of one profiled execution).
Diagrams use few words, symbols and large fonts; details live in the blog text.
"""
import csv
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyBboxPatch, Patch, Polygon, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.dirname(HERE)
DATA = os.path.join(FIGS, "data")

# Inter for text, JetBrains Mono for code-like labels (figs/src/fonts/, SIL OFL), as in neuron-neff-doc;
# DejaVu covers the symbols these two fonts lack.
FONTS = os.path.join(HERE, "fonts")
for _f in sorted(os.listdir(FONTS)):
    if _f.endswith(".ttf"):
        font_manager.fontManager.addfont(os.path.join(FONTS, _f))

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
NKI, GP, DMA, OTHER = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
NKI_L, GP_L, DMA_L, OTHER_L = "#e3eefc", "#fde5da", "#d9f2e8", "#f0efec"
SBUF, SBUF_L, HBM, HBM_L = "#1baf7a", "#d9f2e8", "#7b5cd6", "#ece6fb"   # memories (figure 8)
GOOD, BAD = "#1baf7a", "#e34948"

plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"], "font.monospace": ["JetBrains Mono", "DejaVu Sans Mono"], "font.size": 15, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": SURF,
    "figure.facecolor": SURF, "savefig.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 15.5, "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15,
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


def badge(ax, x, y, n, r=0.19, fs=12.5):
    ax.add_patch(Circle((x, y), r, fc=INK, ec=SURF, lw=1.5, zorder=7))
    ax.text(x, y, str(n), ha="center", va="center", fontsize=fs, color=SURF, weight="bold", zorder=8)


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
    return x


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
    fig, ax = plt.subplots(figsize=(8.6, 4.85))
    fig.subplots_adjust(left=0.3, right=0.97, top=0.755, bottom=0.15)
    h = 0.29
    for k, (name, color, vals) in enumerate(series):
        for g, v in enumerate(vals):
            y = (len(groups) - 1 - g) + (1 - k) * h
            ax.barh(y, v, height=h, color=color, edgecolor=SURF, linewidth=1.5, label=name if g == 0 else None)
            ax.text(v + 0.07, y, f"{v:.2f}x", ha="left", va="center", fontsize=16, weight="bold", color=INK)
    ax.axvline(1.0, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=3)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups[::-1], fontsize=16, color=INK)
    ax.set_ylim(-0.55, len(groups) - 0.5)
    ax.set_xlim(0, 5.8)
    ax.set_xticks(range(0, 6))
    ax.set_xticklabels([f"{t}x" for t in range(0, 6)])
    ax.set_xlabel("Speedup over baseline without GPSIMD custom op")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.99), ncol=1, frameon=False, fontsize=15.5,
               handlelength=1.2, borderaxespad=0, labelspacing=0.35)
    save(fig, "fig01_speedups.png")


# ----------------------------------------------------------------------------------------------- figure 3
def fig_fpga_split():
    """GPU-FPGA split of the Engram module (Engram reference configuration, HeteroLLM kernels/deepseek_engram)."""
    fig, ax = canvas(15.75, 4.2)
    yb, hb = 0.95, 1.6
    ym = yb + hb / 2
    # devices
    box(ax, 1.4, 0.45, 7.0, 3.55, "", "#fff8f4", GP, r=0.22, lw=2.0)
    ax.text(1.65, 3.6, "FPGA", fontsize=19, weight="bold", va="center")
    box(ax, 9.5, 0.45, 5.2, 3.55, "", "#f5f9fe", NKI, r=0.22, lw=2.0)
    ax.text(9.75, 3.6, "GPU", fontsize=19, weight="bold", va="center")
    for x, lab, ic, sz in [(1.6, "hash", icon_hash, 0.68), (3.95, "table lookup", icon_table, 0.62),
                           (6.3, "K/V proj.", icon_matmul, 0.8)]:
        icon_box(ax, x, yb, 1.9, hb, lab, ic, sz, GP)
    icon_box(ax, 9.7, yb, 2.15, hb, "gate", icon_sigmoid, 0.72, NKI)
    icon_box(ax, 12.25, yb, 2.25, hb, "conv + residual", icon_window, 0.6, NKI)
    arrow(ax, 3.5, ym, 3.95, ym, lw=2.2)
    arrow(ax, 5.85, ym, 6.3, ym, lw=2.2)
    arrow(ax, 11.85, ym, 12.25, ym, lw=2.2)
    # PCIe hand-off
    arrow(ax, 8.2, ym, 9.7, ym, color=INK, lw=3.2)
    ax.text(8.95, ym + 0.22, "PCIe", ha="center", va="bottom", fontsize=19, weight="bold")
    # inputs and output
    ax.text(0.1, ym, "tokens", ha="left", va="center", fontsize=19)
    arrow(ax, 1.15, ym, 1.6, ym, lw=2.2)
    arrow(ax, 10.775, 0.0, 10.775, yb, lw=2.2)
    ax.text(10.6, 0.2, "h", ha="right", va="center", fontsize=20, style="italic")
    arrow(ax, 14.5, ym, 14.95, ym, lw=2.2)
    ax.text(15.02, ym, "attn", ha="left", va="center", fontsize=19)
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
                ha="center", va="bottom", fontsize=17, color=INK, weight="bold")
    ax.set_ylim(0, top * 1.35)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{int(r['tokens']):,}" for r in rows], color=INK, fontsize=16)
    ax.set_xlabel("Tokens")
    ax.set_ylabel("Engram time (ms)")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.legend(handles=[Patch(color=NKI, label="GPU only"), Patch(color=GP, label="GPU + 2 FPGAs")],
              loc="upper left", frameon=False, ncol=2, fontsize=16)
    save(fig, "fig04_fpga_results.png")


# ----------------------------------------------------------------------------------------------- figure 5
MONO = ["JetBrains Mono", "DejaVu Sans Mono"]


def icon_gear(ax, cx, cy, r, color, bg="#ffffff", teeth=8):
    """A gear: a compiler."""
    pts, step = [], 2 * np.pi / teeth
    for i in range(teeth):
        a0 = i * step
        for da, rr in [(-0.30, 0.74), (-0.16, 1.0), (0.16, 1.0), (0.30, 0.74)]:
            pts.append((cx + r * rr * np.cos(a0 + da * step), cy + r * rr * np.sin(a0 + da * step)))
    ax.add_patch(Polygon(pts, closed=True, fc=color, ec=color, lw=1, zorder=4))
    ax.add_patch(Circle((cx, cy), r * 0.32, fc=bg, ec=bg, zorder=5))


def icon_syringe(ax, cx, cy, s, color, bg="#ffffff"):
    """A syringe, needle down-left: injecting the operator into the NEFF."""
    from matplotlib.transforms import Affine2D
    tr = Affine2D().scale(s).rotate_deg(225).translate(cx, cy) + ax.transData
    kw = dict(transform=tr, zorder=4)
    ax.add_patch(Rectangle((-0.52, -0.17), 0.07, 0.34, fc=color, ec=color, **kw))            # plunger handle
    ax.add_patch(Rectangle((-0.46, -0.04), 0.24, 0.08, fc=color, ec=color, **kw))            # plunger rod
    ax.add_patch(Rectangle((-0.25, -0.21), 0.05, 0.42, fc=color, ec=color, **kw))            # finger flange
    ax.add_patch(Rectangle((-0.20, -0.12), 0.48, 0.24, fc=bg, ec=color, lw=2.2, **kw))       # barrel
    ax.add_patch(Rectangle((-0.02, -0.12), 0.30, 0.24, fc=color, ec=color, alpha=0.35, **kw))  # dose
    for x in (0.02, 0.10, 0.18):
        ax.plot([x, x], [0.12, 0.03], color=color, lw=1.6, transform=tr, zorder=5)            # graduations
    ax.add_patch(Polygon([(0.28, -0.12), (0.37, -0.04), (0.37, 0.04), (0.28, 0.12)], closed=True,
                         fc=color, ec=color, **kw))                                           # tip
    ax.plot([0.37, 0.62], [0, 0], color=color, lw=2.2, solid_capstyle="round", transform=tr, zorder=4)  # needle


def icon_chip(ax, cx, cy, s, color, bg="#ffffff"):
    """A chip with pins: the NeuronCore that runs the NEFF."""
    b = s * 0.62
    for k in (-1, 0, 1):
        o = k * b * 0.3
        for (x0, y0, x1, y1) in [(cx + o, cy + b / 2, cx + o, cy + s / 2), (cx + o, cy - b / 2, cx + o, cy - s / 2),
                                 (cx + b / 2, cy + o, cx + s / 2, cy + o), (cx - b / 2, cy + o, cx - s / 2, cy + o)]:
            ax.plot([x0, x1], [y0, y1], color=color, lw=2.6, solid_capstyle="round", zorder=4)
    ax.add_patch(FancyBboxPatch((cx - b / 2, cy - b / 2), b, b, boxstyle=f"round,pad=0,rounding_size={b * 0.12}",
                                fc=color, ec=color, zorder=5))
    ax.add_patch(Rectangle((cx - b * 0.22, cy - b * 0.22), b * 0.44, b * 0.44, fc=bg, ec=bg, zorder=6))


def icon_table(ax, cx, cy, s, color, bg="#ffffff"):
    """A table: header row and a grid of cells."""
    w, h = 1.35 * s, s
    x0, y0 = cx - w / 2, cy - h / 2
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={s * 0.08}",
                                fc=bg, ec=color, lw=2.2, zorder=4))
    ax.add_patch(Rectangle((x0, y0 + h * 0.74), w, h * 0.26, fc=color, ec=color, lw=0, zorder=5))
    for f in (0.25, 0.5):
        ax.plot([x0, x0 + w], [y0 + h * f, y0 + h * f], color=color, lw=1.5, zorder=5)
    for f in (1 / 3, 2 / 3):
        ax.plot([x0 + w * f, x0 + w * f], [y0, y0 + h * 0.74], color=color, lw=1.5, zorder=5)


def icon_hash(ax, cx, cy, s, color, bg=None):
    """A hash sign drawn with four strokes."""
    kw = dict(color=color, lw=4.6, solid_capstyle="round", zorder=4)
    ax.plot([cx - 0.22 * s, cx - 0.06 * s], [cy - 0.5 * s, cy + 0.5 * s], **kw)
    ax.plot([cx + 0.06 * s, cx + 0.22 * s], [cy - 0.5 * s, cy + 0.5 * s], **kw)
    for dy in (-0.17, 0.17):
        ax.plot([cx - 0.42 * s, cx + 0.42 * s], [cy + dy * s, cy + dy * s], **kw)


def icon_window(ax, cx, cy, s, color, bg=None):
    """A row of tokens with a sliding window around three of them."""
    q, g = 0.34 * s, 0.1 * s
    x0 = cx - (5 * q + 4 * g) / 2
    for i in range(5):
        inside = 1 <= i <= 3
        ax.add_patch(Rectangle((x0 + i * (q + g), cy - q / 2), q, q, fc=color, ec="none",
                               alpha=1.0 if inside else 0.3, zorder=4))
    xa, xb = x0 + (q + g) - 0.1 * s, x0 + 4 * (q + g) - g + 0.1 * s
    ax.add_patch(FancyBboxPatch((xa, cy - q / 2 - 0.13 * s), xb - xa, q + 0.26 * s,
                                boxstyle=f"round,pad=0,rounding_size={s * 0.08}", fc="none", ec=color, lw=2.4,
                                zorder=5))


def icon_list(ax, cx, cy, s, color, bg=None):
    """A list: three bullet rows."""
    for dy in (0.34, 0, -0.34):
        ax.add_patch(Circle((cx - 0.48 * s, cy + dy * s), 0.085 * s, fc=color, ec="none", zorder=4))
        ax.plot([cx - 0.3 * s, cx + 0.52 * s], [cy + dy * s, cy + dy * s], color=color, lw=3.6,
                solid_capstyle="round", zorder=4)


def icon_gather(ax, cx, cy, s, color, bg="#ffffff"):
    """Gather: two marked rows of a table move into one dense block."""
    bh, gp = 0.15 * s, 0.065 * s
    tot = 5 * bh + 4 * gp
    xl0, xl1 = cx - 0.78 * s, cx - 0.16 * s
    hit = (1, 3)
    for i in range(5):
        y = cy + tot / 2 - bh - i * (bh + gp)
        ax.add_patch(Rectangle((xl0, y), xl1 - xl0, bh, fc=color if i in hit else bg, ec=color, lw=1.4, zorder=4))
    xr0, xr1 = cx + 0.3 * s, cx + 0.8 * s
    for i in range(2):
        y = cy + gp / 2 - bh + (1 - i) * (bh + gp) - (bh + gp) / 2 + bh / 2 - bh / 2
        ax.add_patch(Rectangle((xr0, cy + (0.5 - i) * (bh + gp) - bh / 2), xr1 - xr0, bh, fc=color, ec=color,
                               lw=1.4, zorder=4))
    for i, k in enumerate(hit):
        ya = cy + tot / 2 - bh / 2 - k * (bh + gp)
        yb = cy + (0.5 - i) * (bh + gp)
        ax.annotate("", xy=(xr0 - 0.04 * s, yb), xytext=(xl1 + 0.04 * s, ya),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, shrinkA=0, shrinkB=0, mutation_scale=11),
                    zorder=5)


def icon_ring(ax, cx, cy, s, color, bg="#ffffff"):
    """All-reduce: four ranks on a ring, passing data around it."""
    R = 0.4 * s
    ax.add_patch(Circle((cx, cy), R, fc="none", ec=color, lw=2.4, zorder=4))
    for k in range(4):
        a = k * np.pi / 2
        ax.add_patch(Circle((cx + R * np.cos(a), cy + R * np.sin(a)), 0.12 * s, fc=color, ec=bg, lw=1.5, zorder=6))
        a += np.pi / 4                              # arrowhead between two ranks, counterclockwise
        p, t, n = np.array([cx + R * np.cos(a), cy + R * np.sin(a)]), np.array([-np.sin(a), np.cos(a)]), \
            np.array([np.cos(a), np.sin(a)])
        ax.add_patch(Polygon([p + t * 0.11 * s, p - t * 0.07 * s + n * 0.1 * s, p - t * 0.07 * s - n * 0.1 * s],
                             closed=True, fc=color, ec=color, lw=1, zorder=5))


def icon_matmul(ax, cx, cy, s, color, bg="#ffffff"):
    """A matrix multiplication: a weight matrix times a narrow input."""
    a = 0.66 * s
    xa, ya = cx - 0.62 * s, cy - a / 2
    ax.add_patch(Rectangle((xa, ya), a, a, fc=color, ec=color, lw=2.0, alpha=0.3, zorder=4))
    ax.add_patch(Rectangle((xa, ya), a, a, fc="none", ec=color, lw=2.2, zorder=5))
    for f in (1 / 3, 2 / 3):
        ax.plot([xa + a * f, xa + a * f], [ya, ya + a], color=color, lw=1.3, zorder=5)
        ax.plot([xa, xa + a], [ya + a * f, ya + a * f], color=color, lw=1.3, zorder=5)
    xm, d = cx + 0.18 * s, 0.07 * s
    for sgn in (1, -1):
        ax.plot([xm - d, xm + d], [cy - sgn * d, cy + sgn * d], color=color, lw=2.4, solid_capstyle="round", zorder=5)
    xb = cx + 0.36 * s
    ax.add_patch(Rectangle((xb, ya), 0.24 * s, a, fc=color, ec=color, lw=2.0, zorder=5))


def icon_sigmoid(ax, cx, cy, s, color, bg=None):
    """A gate: a sigmoid curve over its axis."""
    t = np.linspace(-1, 1, 80)
    ax.plot([cx - 0.62 * s, cx + 0.62 * s], [cy - 0.44 * s, cy - 0.44 * s], color=color, lw=1.6, alpha=0.5,
            solid_capstyle="round", zorder=4)
    ax.plot([cx, cx], [cy - 0.5 * s, cy + 0.5 * s], color=color, lw=1.6, alpha=0.5, solid_capstyle="round", zorder=4)
    ax.plot(cx + 0.6 * s * t, cy - 0.4 * s + 0.82 * s / (1 + np.exp(-6 * t)), color=color, lw=3.4,
            solid_capstyle="round", zorder=5)


def icon_attention(ax, cx, cy, s, color, bg="#ffffff"):
    """Attention: a causal score matrix, darker cells for higher scores."""
    n, q, g = 4, 0.24 * s, 0.045 * s
    x0, y0 = cx - (n * q + (n - 1) * g) / 2, cy + (n * q + (n - 1) * g) / 2
    score = [[1.0], [0.45, 0.9], [0.3, 0.85, 0.5], [0.8, 0.3, 0.55, 1.0]]
    for i in range(n):
        for j in range(n):
            a = score[i][j] if j <= i else 0.12
            ax.add_patch(Rectangle((x0 + j * (q + g), y0 - (i + 1) * q - i * g), q, q, fc=color, ec="none",
                                   alpha=a, zorder=4))


def custom_mark(ax, cx, cy, r=0.21):
    """The mark of a custom operator: a white star on a dark disc."""
    ax.add_patch(Circle((cx, cy), r, fc=INK, ec=SURF, lw=1.5, zorder=7))
    a = np.pi / 2 + np.arange(10) * np.pi / 5
    rr = np.where(np.arange(10) % 2 == 0, 0.7 * r, 0.3 * r)
    ax.add_patch(Polygon(np.c_[cx + rr * np.cos(a), cy + rr * np.sin(a)], closed=True, fc=SURF, ec="none", zorder=8))


def icon_box(ax, x, y, w, h, label, icon, s, color, fc="#ffffff", ec=None, fs=19, lw=2.0):
    """A step: an icon above its name (the style of the tool boxes of figure 6)."""
    box(ax, x, y, w, h, "", fc, ec or color, r=0.22, lw=lw)
    icon(ax, x + w / 2, y + h * 0.64, s, color, bg=fc)
    ax.text(x + w / 2, y + h * 0.23, label, ha="center", va="center", fontsize=fs, color=INK)


def tool_box(ax, x, y, w, h, label, icon, color, fs=19):
    """A tool: an icon above its name."""
    box(ax, x, y, w, h, "", "#ffffff", INK2, r=0.22, lw=1.8)
    icon(ax, x + w / 2, y + h * 0.67, {icon_gear: 0.3, icon_syringe: 0.72, icon_chip: 0.62}[icon], color)
    ax.text(x + w / 2, y + h * 0.24, label, ha="center", va="center", fontsize=fs, family=MONO, color=INK)


def file_box(ax, x, y, w, h, name, fc, ec, sub=None, fs=19, c=0.3):
    """A file (folded corner) with its name and an optional second line."""
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h - c), (x + w - c, y + h), (x, y + h)],
                         closed=True, fc=fc, ec=ec, lw=2.0))
    ax.plot([x + w - c, x + w - c, x + w], [y + h, y + h - c, y + h - c], color=ec, lw=1.6)
    if sub:
        ax.text(x + w / 2, y + h * 0.63, name, ha="center", va="center", fontsize=fs, family=MONO, color=INK)
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=fs - 3, family=MONO, color=INK2)
    else:
        ax.text(x + w / 2, y + h / 2, name, ha="center", va="center", fontsize=fs, family=MONO, color=INK)


def fig_toolchain():
    fig, ax = canvas(14.6, 6.0)
    h = 1.35
    yt, ym, yb = 4.45, 2.35, 0.25                 # NKI row, inject row, operator row
    xs, xc, xo, xi, xf = 0.3, 3.35, 6.0, 8.95, 11.75   # columns: source, compiler, output, inject, final
    ws, wc, wo, wi, wf = 2.55, 2.15, 2.25, 2.0, 2.55
    mid = lambda y: y + h / 2
    # NKI lane (blue)
    file_box(ax, xs, yt, ws, h, "nki_kernel.py", NKI_L, NKI, sub="gp.call(…)")
    tool_box(ax, xc, yt, wc, h, "neuronx-cc", icon_gear, NKI)
    file_box(ax, xo, yt, wo, h, "kernel.neff", NKI_L, NKI)
    arrow(ax, xs + ws, mid(yt), xc, mid(yt), lw=2.2)
    arrow(ax, xc + wc, mid(yt), xo, mid(yt), lw=2.2)
    # operator lane (orange)
    file_box(ax, xs, yb, ws, h, "op.cpp", GP_L, GP)
    tool_box(ax, xc, yb, wc, h, "gpsimd-cc", icon_gear, GP)
    file_box(ax, xo, yb, wo, h, "ops.so", GP_L, GP)
    arrow(ax, xs + ws, mid(yb), xc, mid(yb), lw=2.2)
    arrow(ax, xc + wc, mid(yb), xo, mid(yb), lw=2.2)
    # tile addresses on SBUF: read from the compiled NEFF, compiled into the operator
    arrow(ax, xo + 0.45, yt, xc + wc - 0.35, yb + h, lw=2.2)
    ax.text(xc + wc - 0.05, mid(ym) + 0.25, "Tile addresses\non SBUF", ha="right", va="center", fontsize=17,
            color=INK2, linespacing=1.25)
    # inject -> final NEFF -> run
    tool_box(ax, xi, ym, wi, h, "inject", icon_syringe, GP)
    arrow(ax, xo + wo, mid(yt) - 0.2, xi + wi * 0.35, ym + h, lw=2.2)
    arrow(ax, xo + wo, mid(yb) + 0.2, xi + wi * 0.35, ym, lw=2.2)
    file_box(ax, xf, ym, wf, h, "final.neff", NKI_L, NKI, sub="+ KERNEL_CALL")
    arrow(ax, xi + wi, mid(ym), xf, mid(ym), lw=2.2)
    xr, wr = xf + (wf - 2.0) / 2, 2.0
    tool_box(ax, xr, yb, wr, h, "run", icon_chip, INK2)
    arrow(ax, xf + wf / 2, ym, xf + wf / 2, yb + h, lw=2.2)
    # step numbers (the list in "Overall Development and Deployment Flow")
    for n, (x, y) in enumerate([(xs, yb + h), (xs, yt + h), (xc, yb + h), (xi, ym + h), (xr, yb + h)], 1):
        badge(ax, x, y, n, r=0.27, fs=17)
    save(fig, "fig06_toolchain.png")


# ----------------------------------------------------------------------------------------------- figure 7
EV, EV_L = "#7b5cd6", "#ece6fb"                   # events block


def fig_slot_layout():
    """Byte layout of the join marker that gp.call emits and of the KERNEL_CALL that inject writes, with the values of
    the n-gram hashing kernel (nkiopt/cases/ds41_ngram/build_dec_bs10/gpsimd: kernel.neff Pool0.bin slot 67,
    final.neff Pool0.bin slot 68). Field offsets: ts_operands() and inject() in toolchain/bin/gpsimd-neff."""
    W, H = 14.6, 7.75
    fig, ax = canvas(W, H)
    x0, cw, bh = 0.15, (W - 0.3) / 64, 1.42
    X = lambda b: x0 + b * cw
    WHITE = ("#ffffff", GRID)

    def field(y, a, b, name, val=None, col=(OTHER_L, OTHER), fs=16, tc=INK, w=None):
        x, wd = (X(a), (b - a + 1) * cw) if w is None else (a, w)
        box(ax, x, y, wd, bh, "", col[0], col[1], r=0.06, lw=1.6)
        cx = x + wd / 2
        if val:
            ax.text(cx, y + bh * 0.66, name, ha="center", va="center", fontsize=fs, color=tc, linespacing=1.15)
            ax.text(cx, y + bh * 0.24, val, ha="center", va="center", fontsize=fs - 1.5, family=MONO, color=INK2,
                    linespacing=1.15)
        elif name:
            ax.text(cx, y + bh / 2, name, ha="center", va="center", fontsize=fs, color=tc, linespacing=1.15)

    def ruler(y, marks, xf=X):
        for b in marks:
            ax.text(xf(b), y - 0.12, str(b), ha="center", va="top", fontsize=14.5, color=MUTED)

    # join marker emitted by gp.call (kernel.neff)
    ya = H - 0.5 - bh
    ax.text(x0, ya + bh + 0.3, "Example marker instruction", fontsize=18.5, weight="bold",
            va="center")
    for f in [(0, 3, "header", "0x43", (OTHER_L, OTHER), 15), (4, 11, "events", None, (EV_L, EV)), (12, 15, "unused", None, WHITE, 15.5, MUTED),
              (16, 19, "src\naddr", None, (NKI_L, NKI)), (20, 31, "src pattern\nstep[3] num[3]", None, (NKI_L, NKI)),
              (32, 39, "dtype,\npartitions,\nALU ops", "int32, 128", (OTHER_L, OTHER), 14.5),
              (40, 43, "imm 0"), (44, 47, "imm 1\n= tag", "join", (GP_L, GP)),
              (48, 51, "dst\naddr", "0x4600", (DMA_L, DMA)),
              (52, 63, "dst pattern\nstep[3] num[3]", "num = 48, 1, 1", (DMA_L, DMA))]:
        field(ya, *f)
    ruler(ya, range(0, 65, 8))
    # zoom into the events block
    zc, yz = 0.82, ya - 1.3 - bh
    Z = lambda b: x0 + (b - 4) * zc
    yk = ya - 0.5                                  # bracket under bytes 4-11, below the byte numbers
    ax.plot([X(4), X(4), X(12), X(12)], [yk + 0.1, yk, yk, yk + 0.1], color=EV, lw=1.4)
    for xa, xb in [(X(4), Z(4)), (X(12), Z(12))]:
        ax.plot([xa, xb], [yk, yz + bh], color=EV, lw=1.3, ls=(0, (4, 3)))
    for f in [(4, "wait\nmode", "≥"), (5, "wait\nsem.", "S[3]"), (6, "update\nmode", "+1"), (7, "update\nsem.", "S[3]")]:
        field(yz, Z(f[0]), None, f[1], f[2], (EV_L, EV), 14.5, w=zc)
    field(yz, Z(8), None, "wait threshold /\nupdate amount", "28", (EV_L, EV), 15.5, w=4 * zc)
    ruler(yz, [4, 5, 6, 7, 8, 12], Z)
    # KERNEL_CALL written by inject (final.neff)
    yb = yz - 1.15 - bh
    ax.text(x0, yb + bh + 0.3, "KERNEL_CALL", fontsize=18.5, weight="bold", va="center")
    for f in [(0, 3, "header", "0x85", (OTHER_L, OTHER), 15), (4, 11, "events", "S[3] += 9", (EV_L, EV)), (12, 12, "", None, (GP_L, GP)),
              (13, 63, "unused (zero)", None, WHITE, 16, MUTED)]:
        field(yb, *f)
    ruler(yb, range(0, 65, 8))
    ax.plot([X(12.5), X(12.5)], [yb, yb - 0.55], color=GP, lw=1.6)
    ax.text(X(12.5), yb - 0.62, "operator index = 0", ha="center", va="top", fontsize=16, color=GP)
    save(fig, "fig07_slot_layout.png")


# ----------------------------------------------------------------------------------------------- figure 6
def fig_engram_split():
    fig, ax = canvas(14.6, 7.0)
    heads = ["Prepare Memory", "Compute Relevancy", "Retrieval", "Apply to Inference"]
    cw = 3.45
    x0s = [0.1 + i * (cw + 0.2) for i in range(4)]
    for t, x in zip(heads, x0s):
        box(ax, x, 0.7, cw, 6.1, "", "#ffffff", GRID, r=0.18, lw=1.8)
        ax.text(x + cw / 2, 6.38, t, ha="center", va="center", fontsize=19, weight="bold")
    bw, h = 2.85, 1.35
    bx = [x + (cw - bw) / 2 for x in x0s]
    yt, yc, yl = 4.55, 2.75, 0.95                      # rows: top, middle, bottom
    # color = where the step runs (GPSIMD or the other engines) or where its data lives (SBUF or HBM)
    icon_box(ax, bx[0], yt, bw, h, "n-gram windows", icon_window, 0.56, SBUF, SBUF_L)
    icon_box(ax, bx[0], yc, bw, h, "table (384M rows)", icon_table, 0.56, HBM, HBM_L)
    icon_box(ax, bx[1], yt, bw, h, "n-gram hash", icon_hash, 0.62, GP, GP_L)
    icon_box(ax, bx[2], yt, bw, h, "build gather list", icon_list, 0.62, GP, GP_L)
    icon_box(ax, bx[2], yc, bw, h, "gather listed rows", icon_gather, 0.64, GP, GP_L)   # NKI's indirect DMAs
    icon_box(ax, bx[2], yl, bw, h, "all-reduce", icon_ring, 0.64, NKI, NKI_L)
    icon_box(ax, bx[3], yl, bw, h, "Attention", icon_attention, 0.62, NKI, NKI_L)
    icon_box(ax, bx[3], yc, bw, h, "gate", icon_sigmoid, 0.62, NKI, NKI_L)
    for b in (bx[1], bx[2]):                           # the two custom operators
        custom_mark(ax, b + bw - 0.32, yt + h - 0.32)
    mid = lambda y: y + h / 2
    arrow(ax, bx[0] + bw, mid(yt), bx[1], mid(yt), lw=2.2)          # windows -> hash
    arrow(ax, bx[1] + bw, mid(yt), bx[2], mid(yt), lw=2.2)          # hash -> gather list
    arrow(ax, bx[2] + bw / 2, yt, bx[2] + bw / 2, yc + h, lw=2.2)   # gather list -> gather
    arrow(ax, bx[0] + bw, mid(yc), bx[2], mid(yc), lw=2.2)          # table rows -> gather
    arrow(ax, bx[2] + bw / 2, yc, bx[2] + bw / 2, yl + h, lw=2.2)   # gather -> all-reduce
    arrow(ax, bx[2] + bw, mid(yl), bx[3], mid(yl), lw=2.2)          # all-reduce -> attention
    arrow(ax, bx[3] + bw / 2, yl + h, bx[3] + bw / 2, yc, lw=2.2)   # attention -> gate
    x = legend_row(ax, 0.2, 0.25, [(SBUF_L, SBUF, "SBUF"), (HBM_L, HBM, "HBM"), (GP_L, GP, "GPSIMD"),
                                   (NKI_L, NKI, "other engines")], fs=17, gap=0.5)
    custom_mark(ax, x - 0.1, 0.25, r=0.17)
    ax.text(x + 0.19, 0.25, "custom operator", va="center", fontsize=17, color=INK)
    save(fig, "fig08_engram_split.png")


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
               [("#bcd6f6", "fetched, masked out"), (NKI, "hit")]),
              ("GPSIMD + NKI", "372 rows fetched  ·  3 gathers", "#f1f0ec", GP,
               [("#f1f0ec", "skipped"), (GP, "hit")])]
    for side, (title, stat, miss, hit, leg) in enumerate(panels):
        ax = fig.add_axes([0.01 + side * 0.5, 0.02, 0.47, 0.9])
        ax.set_xlim(-1.2, ncol + 0.2)
        ax.set_ylim(-5.0, nrow + 2.4)
        ax.axis("off")
        ax.text(-1.0, nrow + 1.5, title, fontsize=18.5, weight="bold", va="center")
        for r in range(nrow):
            for c in range(ncol):
                ax.add_patch(Rectangle((c + 0.07, nrow - 1 - r + 0.07), 0.86, 0.86,
                                       fc=hit if hits[r, c] else miss, ec="none"))
        ax.text(ncol / 2 - 1.2, -0.8, "24 hash columns", ha="center", va="center", fontsize=15, color=INK2)
        arrow(ax, ncol / 2 + 3.5, -0.8, ncol / 2 + 5.4, -0.8, lw=1.8)
        ax.text(-0.55, nrow / 2 + 1.6, "tokens", rotation=270, ha="center", va="center", fontsize=15, color=INK2)
        arrow(ax, -0.55, nrow / 2 - 0.4, -0.55, nrow / 2 - 2.4, lw=1.8)
        ax.text(-1.0, -2.35, stat, fontsize=17, weight="bold", va="center")
        x = -1.0
        for fc, lab in leg:
            ax.add_patch(Rectangle((x, -4.05), 0.8, 0.8, fc=fc, ec="none"))
            ax.text(x + 1.2, -3.65, lab, fontsize=15, va="center")
            x += 1.2 + 0.58 * len(lab) + 1.0
    save(fig, "fig10_gather.png")


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
    ax.set_yticklabels([l[0] for l in lanes][::-1], fontsize=16.5, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.7, n - 0.3)
    end = max(r["start_us"] + r["dur_us"] for r in rows)
    ax.axvline(end, color=INK, lw=1.4, ls=(0, (3, 2)))
    ax.text(end + xmax * 0.008, n - 1, f"{end:.0f} µs", fontsize=18, weight="bold", va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.15", fc=SURF, ec="none"))
    ax.set_title(title, loc="left", fontsize=19, weight="bold", color=INK, pad=12)
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
    axs[1].set_xlabel("Time (µs)", fontsize=17)
    axs[1].tick_params(axis="x", labelsize=16.5)
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
        ax.text(t + 80, y, f"{t:,.0f} µs ({100 * g / t:.0f}%)", va="center", fontsize=15.5, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([d[0] for d in data], fontsize=15.5, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 9000)
    ax.set_xlabel("Read-out time on one NeuronCore (µs)")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(color=GP, label="table retrieval"), Patch(color=NKI, label="attention + gate")],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=15.5, ncol=2)
    save(fig, "fig13_amdahl.png")



# ----------------------------------------------------------------------------- figure 5
def fig_trainium_arch():
    """the two Neuron documentation diagrams side by side on white, at a common height (no redrawing)"""
    from PIL import Image
    H, gap, pad = 760, 70, 20
    parts = []
    for name in ("trainium3_device.png", "trainium3_neuroncore.png"):
        im = Image.open(os.path.join(HERE, "aws", name)).convert("RGBA")
        bg = Image.new("RGBA", im.size, "white")
        im = Image.alpha_composite(bg, im).convert("RGB")
        parts.append(im.resize((round(im.width * H / im.height), H), Image.LANCZOS))
    out = Image.new("RGB", (sum(p.width for p in parts) + gap + 2 * pad, H + 2 * pad), "white")
    x = pad
    for p in parts:
        out.paste(p, (x, pad))
        x += p.width + gap
    out.save(os.path.join(FIGS, "fig05_trainium_arch.png"))
    print("wrote fig05_trainium_arch.png")

if __name__ == "__main__":
    fig_speedups()
    fig_fpga_split()
    fig_fpga_results()
    fig_trainium_arch()
    fig_toolchain()
    fig_slot_layout()
    fig_engram_split()
    fig_gather()
    fig_amdahl()
    if os.path.exists(os.path.join(DATA, "hash_decode_bs10_gpsimd.csv")):
        fig_timeline("hash_decode_bs10", "fig09_timeline_hash.png", ["Baseline", "GPSIMD + NKI"])
    if os.path.exists(os.path.join(DATA, "gather_t1024_gpsimd.csv.gz")):
        fig_timeline("gather_t1024", "fig11_timeline_gather.png", ["Baseline", "GPSIMD + NKI"])
    if os.path.exists(os.path.join(DATA, "readout_t1024_gpsimd.csv.gz")):
        fig_timeline("readout_t1024", "fig12_timeline_readout.png", ["Baseline", "GPSIMD + NKI"])
