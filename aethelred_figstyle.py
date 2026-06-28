# -*- coding: utf-8 -*-
"""
Aethelred figure style system — one palette / typography for ALL paper figures.

Design goals (AAAI/NeurIPS-grade, venue-agnostic):
  * Aethelred is ALWAYS the same bold colour (#1A4E8A) in every figure -> instant
    recognizability across the paper.
  * Colourblind-safe contrastive palette (Okabe-Ito derived).
  * One look: same linewidth, font, grid, legend, margins everywhere.
  * Every figure saved as BOTH .png (300 dpi) and .pdf (vector) into figures/.

Import this in every plotting script:
    import aethelred_figstyle as fs
    fs.apply()
    fig, ax = fs.new_fig()
    ax.plot(xs, ys, **fs.style('Aethelred'))
    fs.save(fig, 'f1_cert_pred')
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(ROOT, "figures")

# --- palette: WARM, contrastive, high-saturation; black edges add definition.
# Aethelred = the most vivid crimson (hero, stands out); baselines warm-toned.
C = {
    "Aethelred":   "#E8112D",  # vivid crimson — OURS, always (most saturated)
    "Aethelred-2": "#FF6F59",  # warm coral — Aethelred variant
    "PGNNCert":    "#F77F00",  # orange
    "PGNNCert-E":  "#F77F00",  # orange
    "PGNNCert-N":  "#B5651D",  # burnt orange
    "XGNNCert":    "#6A040F",  # deep wine — distinct from crimson
    "Bagging":     "#FCBF49",  # gold / amber
    "RS":          "#E76F51",  # coral
    "Bi-RS":       "#9C6644",  # brown
    "Bi-RS-Exclude": "#9C6644", # brown (strong RS variant)
    "Bi-RS-Include": "#C99A6A", # light tan-brown
    "V-InfoR":     "#C9379D",  # warm magenta
    "PGExplainer": "#BC8A5F",  # tan  — base explainers
    "ReFine":      "#8B5E34",  # mid-brown
    "GSAT":        "#6F4518",  # dark brown
    "DIR":         "#E07A5F",  # terracotta — causal baseline
    "GNNExplainer-spm": "#B08968",  # warm taupe — non-causal explainer
    "Undefended":  "#ADADAD",  # neutral grey
    "GNNExplainer": "#A8A8A8",
    # DIR-paper Table 2 interpretation baselines (secondary; muted, distinct tones)
    "Attention":   "#9AA0A6",  # cool grey
    "ASAP":        "#B5838D",  # dusty rose
    "Top-k Pool":  "#8E9B6C",  # olive
    "SAG Pool":    "#C08552",  # caramel
}
# linestyle + marker per method (ours solid+filled; baselines dashed/dotted)
_LS = {
    "Aethelred": ("-", "o"), "Aethelred-2": ("-", "s"),
    "PGNNCert": ("--", "^"), "PGNNCert-E": ("--", "^"), "PGNNCert-N": ("--", "v"),
    "XGNNCert": ("--", "D"), "Bagging": (":", "s"), "RS": (":", "P"),
    "Bi-RS": (":", "X"), "Bi-RS-Exclude": (":", "X"), "Bi-RS-Include": (":", "P"),
    "V-InfoR": ("-.", "*"),
    "PGExplainer": (":", "."), "ReFine": (":", "."), "GSAT": (":", "."),
    "DIR": ("--", "s"), "GNNExplainer-spm": (":", "v"),
    "Attention": (":", "v"), "ASAP": ((0, (4, 2)), "^"),
    "Top-k Pool": ((0, (2, 1)), "D"), "SAG Pool": ((0, (5, 1, 1, 1)), "s"),
    "Undefended": ((0, (1, 1)), None),
}


def apply():
    """Publication rcParams. Clean, tight, consistent."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",       # always available; swap to Times for AAAI camera-ready
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.30,
        "grid.linewidth": 0.6,
        "legend.fontsize": 9.5,
        "legend.frameon": False,
        "lines.linewidth": 2.2,
        "lines.markersize": 5.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.constrained_layout.use": True,
    })


def style(method, **over):
    """kwargs for ax.plot for a given method. Markers get BLACK edges."""
    ls, mk = _LS.get(method, ("-", "o"))
    d = dict(color=C.get(method, "#333333"), linestyle=ls, marker=mk,
             label=method, markeredgecolor="black", markeredgewidth=0.7,
             markersize=6.5)
    # ours: thicker + on top
    if method.startswith("Aethelred"):
        d.update(linewidth=2.9, zorder=5, markersize=7.0)
    d.update(over)
    return d


def bar(ax, x, height, width, method, **over):
    """Bar with BLACK edge in the method's colour."""
    kw = dict(color=C.get(method, "#333333"), edgecolor="black",
              linewidth=0.9, label=method)
    kw.update(over)
    return ax.bar(x, height, width, **kw)


def new_fig(w=5.0, h=3.6):
    return plt.subplots(figsize=(w, h))


def grid_fig(n, ncols=2, w=4.2, h=3.2):
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows),
                             squeeze=False)
    return fig, axes


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    return os.path.join(FIGDIR, f"{name}.png")
