# -*- coding: utf-8 -*-
"""F0 — Aethelred architecture schematic with intuitive per-stage glyphs.

Each pipeline stage is an icon tile whose mini node-link glyph *shows* the
operation; the stage name sits beneath the tile.  Pure matplotlib (no raster
assets), so it is deterministic and re-runs cheaply.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon
import aethelred_figstyle as fs

fs.apply()
C = fs.C
RED  = C.get("Aethelred", "#E8112D")
SALM = C.get("Aethelred-2", "#F4978E")
GOLD = C.get("Bagging", "#FCBF49")
PEACH = "#E8956B"
GREEN = "#1B7837"
GREY = "#9A9A9A"
DK = "#333333"

# ----------------------------------------------------------------------------
# A single small graph (a "house" causal motif + spurious context) reused so the
# viewer sees the SAME graph transformed at every stage.
# ----------------------------------------------------------------------------
GN = {0: (-0.55, -0.50), 3: (0.55, -0.50), 2: (0.55, 0.15), 1: (-0.55, 0.15),
      4: (0.0, 0.72), 5: (-1.05, -0.05), 6: (1.05, -0.18)}
CAUSAL = [(0, 3), (3, 2), (2, 1), (1, 0), (1, 4), (2, 4)]   # the house = motif
CONTEXT = [(0, 5), (5, 1), (3, 6)]                           # spurious context


def _p(cx, cy, s, i):
    return (cx + GN[i][0] * s, cy + GN[i][1] * s)


def edge(ax, cx, cy, s, e, color, lw, ls="-", alpha=1.0, z=5):
    a, b = _p(cx, cy, s, e[0]), _p(cx, cy, s, e[1])
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls, alpha=alpha,
            zorder=z, solid_capstyle="round")


def node(ax, cx, cy, s, i, fc=DK, r=0.052, ec="black", lw=0.8, z=8):
    ax.add_patch(Circle(_p(cx, cy, s, i), r, facecolor=fc, edgecolor=ec, lw=lw, zorder=z))


def msg_arrow(ax, cx, cy, s, e, color):
    a, b = _p(cx, cy, s, e[0]), _p(cx, cy, s, e[1])
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    for p, q in ((a, mid), (b, mid)):
        ax.add_patch(FancyArrowPatch(p, ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2),
                     arrowstyle="-|>", mutation_scale=8, lw=1.6, color=color, zorder=6))


# ----------------------------------------------------------------------------
# Per-stage glyphs
# ----------------------------------------------------------------------------
def g_input(ax, cx, cy, s):
    for e in CONTEXT + CAUSAL:
        edge(ax, cx, cy, s, e, GREY, 1.7)
    for i in GN:
        node(ax, cx, cy, s, i, fc="#888888")


def g_core(ax, cx, cy, s):                      # score ONE edge from its 2 endpoints
    for e in CONTEXT + CAUSAL:
        edge(ax, cx, cy, s, e, "#D9D9D9", 1.5)
    for i in GN:
        node(ax, cx, cy, s, i, fc="#D0D0D0", r=0.04, lw=0.5)
    tgt = (2, 4)
    edge(ax, cx, cy, s, tgt, RED, 3.2, z=9)
    for i in tgt:                                # only the two endpoints "light up"
        node(ax, cx, cy, s, i, fc=RED, r=0.062, ec="black", lw=1.1, z=10)
    a, b = _p(cx, cy, s, tgt[0]), _p(cx, cy, s, tgt[1])
    ax.text((a[0] + b[0]) / 2 + 0.16 * s, (a[1] + b[1]) / 2 + 0.12,
            "$s(u,v)$", color=RED, fontsize=8.2, ha="left", va="center", weight="bold")


def g_mask(ax, cx, cy, s):                      # weighted causal mask
    for e in CONTEXT:
        edge(ax, cx, cy, s, e, "#C8C8C8", 1.2, alpha=0.7)
    for e in CAUSAL:
        edge(ax, cx, cy, s, e, RED, 3.0)
    for i in (5, 6):
        node(ax, cx, cy, s, i, fc="#CFCFCF", r=0.042, lw=0.5)
    for i in (0, 1, 2, 3, 4):
        node(ax, cx, cy, s, i, fc=DK)


def g_focal(ax, cx, cy, s):                     # message passing gated to the motif
    for e in CONTEXT:
        edge(ax, cx, cy, s, e, "#D2D2D2", 1.2, alpha=0.6)
    for e in CAUSAL:
        edge(ax, cx, cy, s, e, RED, 2.0, alpha=0.55)
        msg_arrow(ax, cx, cy, s, e, RED)
    for i in (5, 6):
        node(ax, cx, cy, s, i, fc="#CFCFCF", r=0.042, lw=0.5)
    for i in (0, 1, 2, 3, 4):
        node(ax, cx, cy, s, i, fc=DK)


def g_pred(ax, cx, cy, s):                      # class output
    xs = [cx - 0.28, cx, cx + 0.28]
    hs = [0.18, 0.52, 0.12]
    cols = ["#C9C9C9", RED, "#C9C9C9"]
    for x, h, c in zip(xs, hs, cols):
        ax.add_patch(plt.Rectangle((x - 0.075, cy - 0.34), 0.15, h, facecolor=c,
                                   edgecolor="black", lw=0.8, zorder=6))
    ax.text(cx, cy + 0.42, "$\\checkmark$", color=GREEN, fontsize=14, ha="center",
            va="center", weight="bold")


def g_cert(ax, cx, cy, s):                      # ranked scores with a margin at top-k
    ys = [cy + 0.40, cy + 0.22, cy + 0.04, cy - 0.20, cy - 0.38]
    ws = [0.60, 0.52, 0.45, 0.23, 0.16]
    cols = [RED, RED, RED, "#C7C7C7", "#C7C7C7"]
    x0 = cx - 0.74
    for y, w, c in zip(ys, ws, cols):
        ax.add_patch(plt.Rectangle((x0, y - 0.05), w, 0.10, facecolor=c,
                                   edgecolor="black", lw=0.7, zorder=6))
    yc = (ys[2] + ys[3]) / 2                     # top-k cutoff line (k = 3)
    ax.plot([x0 - 0.05, cx + 0.04], [yc, yc], color=DK, lw=1.0, ls=(0, (3, 2)), zorder=7)
    ax.text(x0, ys[0] + 0.16, "top-$k$", color=DK, fontsize=7.2, ha="left", va="center")
    mx = cx + 0.24                               # margin marker, clear of the bars
    ax.add_patch(FancyArrowPatch((mx, ys[2]), (mx, ys[3]), arrowstyle="<|-|>",
                 mutation_scale=7, lw=1.3, color=RED, zorder=8))
    ax.text(mx + 0.08, yc, "margin", color=RED, fontsize=7.2, ha="left", va="center",
            weight="bold")


def g_certexpl(ax, cx, cy, s):                  # certified subgraph + guarantee badge
    for e in CAUSAL:
        edge(ax, cx, cy, s, e, RED, 3.0)
    for i in (0, 1, 2, 3, 4):
        node(ax, cx, cy, s, i, fc=DK)
    sh = 0.20                                    # shield with a check = "guaranteed"
    bx, by = cx + 0.46, cy + 0.34
    pts = [(bx + dx * sh, by + dy * sh) for dx, dy in
           [(-0.5, 0.55), (0.5, 0.55), (0.5, -0.1), (0, -0.62), (-0.5, -0.1)]]
    ax.add_patch(Polygon(pts, closed=True, facecolor=GREEN, edgecolor="black", lw=0.9, zorder=11))
    ax.plot([bx - 0.10, bx - 0.02, bx + 0.12], [by + 0.04, by - 0.06, by + 0.14],
            color="white", lw=1.8, zorder=12, solid_capstyle="round")


# ----------------------------------------------------------------------------
# Tiles + labels
# ----------------------------------------------------------------------------
def tile(ax, x, y, w, h, glyph, fc, ec, elw=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.03",
                                linewidth=elw, edgecolor=ec, facecolor=fc, zorder=2))
    glyph(ax, x + w / 2, y + h / 2, min(w, h) * 0.40)
    return (x + w / 2, y, y + h, w)             # (cx, ybottom, ytop, w)


def label(ax, cx, yref, title, sub=None, tc="black", above=False):
    if above:
        ax.text(cx, yref + 0.06, title, ha="center", va="bottom", fontsize=9.3,
                weight="bold", color=tc)
        if sub:
            ax.text(cx, yref + 0.31, sub, ha="center", va="bottom", fontsize=7.3, color="#555555")
    else:
        ax.text(cx, yref - 0.04, title, ha="center", va="top", fontsize=9.3,
                weight="bold", color=tc)
        if sub:
            ax.text(cx, yref - 0.30, sub, ha="center", va="top", fontsize=7.3, color="#555555")


def arrow(ax, x1, y1, x2, y2, color="black", lw=1.9):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=15, lw=lw, color=color, zorder=4))


fig, ax = plt.subplots(figsize=(12.6, 5.6))
ax.set_xlim(0, 12.6); ax.set_ylim(0, 5.6); ax.axis("off")

H = 1.40
# top linear row: input -> core -> mask  (labels below)
ti = tile(ax, 0.2, 2.62, 1.7, H, g_input, "#F3F3F3", GREY)
tc = tile(ax, 2.45, 2.62, 2.15, H, g_core, "#FFF0EC", SALM, elw=2.2)
tm = tile(ax, 5.15, 2.62, 1.7, H, g_mask, "#FFF7E3", GOLD, elw=2.2)
# upper branch: focal -> prediction  (labels ABOVE so the fork stays tight)
tf = tile(ax, 7.55, 3.45, 2.0, H, g_focal, "#FFF1E8", PEACH, elw=2.0)
tp = tile(ax, 10.15, 3.55, 1.55, 1.20, g_pred, "#F3F3F3", GREY)
# lower branch (the contribution): certificate -> certified explanation  (labels below)
tk = tile(ax, 7.55, 1.79, 2.0, H, g_cert, "#FFEDED", RED, elw=2.6)
tx = tile(ax, 10.15, 1.89, 1.55, 1.20, g_certexpl, "#FFF1F1", RED, elw=2.0)

label(ax, ti[0], ti[1], "Input graph $G$")
label(ax, tc[0], tc[1], "Causal Discovery Core", "propagation-free edge scorer", tc=RED)
label(ax, tm[0], tm[1], "Causal mask $G_c$")
label(ax, tf[0], tf[2], "Focal Engine", "GCN gated by $G_c$", above=True)
label(ax, tp[0], tp[2], "Prediction $\\hat{y}$", above=True)
label(ax, tk[0], tk[1], "Deterministic certificate", "rank margin $k-r$", tc=RED)
label(ax, tx[0], tx[1], "Certified explanation")

# flow arrows — a tight, symmetric fork at the mask (+/-0.83 in y)
arrow(ax, 1.9, 3.32, 2.45, 3.32)
arrow(ax, 4.6, 3.32, 5.15, 3.32)
arrow(ax, 6.95, 3.52, 7.55, 4.15)               # mask -> focal (up)
arrow(ax, 6.95, 3.12, 7.55, 2.49, color=RED)    # mask -> certificate (down)
arrow(ax, 9.55, 4.15, 10.15, 4.15)
arrow(ax, 9.55, 2.49, 10.15, 2.49, color=RED)

ax.text(8.55, 1.12, "no voting · no smoothing · zero variance", ha="center",
        fontsize=8.3, color=RED, weight="bold")

import os
os.makedirs(fs.FIGDIR, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(fs.FIGDIR, f"F0_architecture.{ext}"), bbox_inches="tight", dpi=300)
print("saved", os.path.join(fs.FIGDIR, "F0_architecture.png"))
