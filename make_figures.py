# -*- coding: utf-8 -*-
"""
Aethelred — reproduce the baseline papers' certified-robustness figures with our
model overlaid. Reads result JSONs and produces shared-axis comparison plots.

Sources (all optional; whatever exists is plotted):
  PREDICTION cert (vs PGNNCert):
    _ref_pgnncert/results/{graph,node}_<ds>_GCN_T*.json -> ['PGNNCert-E']['certified']
    results/phase2_aethelred_predcert_<ds>.json         -> ['curve']  (certified acc vs B)
  EXPLANATION cert (vs XGNNCert):
    results/phase2_xgnncert_<ds>.json   -> ['expl_stability_frac'] / ['certified_faithfulness_frac']
    results/phase2_aethelred_<ds>.json  -> ['curve']  (certified overlap frac vs B)

Outputs PNG+PDF into figures/.
"""
import os, json, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(ROOT, "figures")
os.makedirs(FIGDIR, exist_ok=True)

GRAPH_DS = ["MUTAG", "AIDS", "PROTEINS", "DD"]
NODE_DS = ["Cora-ML", "CiteSeer", "PubMed", "Amazon-C"]
XGNN_DS = ["Benzene", "FC", "Mutagenicity", "SG+House", "BAHouse", "BA3Motif"]


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _curve_to_xy(d):
    """dict {budget_str: val} -> sorted (xs, ys)."""
    items = sorted(((int(k), float(v)) for k, v in d.items()), key=lambda t: t[0])
    return [x for x, _ in items], [y for _, y in items]


# --------------------------------------------------------------------------
def fig_prediction_cert():
    """PGNNCert vs Aethelred: certified accuracy vs edge budget, per dataset."""
    datasets = []
    for kind, dss in (("graph", GRAPH_DS), ("node", NODE_DS)):
        for ds in dss:
            pg = (glob.glob(f"{ROOT}/_ref_pgnncert/results/{kind}_{ds}_GCN_T*.json") or [None])[0]
            ae = f"{ROOT}/results/phase2_aethelred_predcert_{ds}.json"
            if pg or os.path.exists(ae):
                datasets.append((ds, kind, pg, ae))
    if not datasets:
        print("[fig] prediction-cert: no data yet")
        return
    n = len(datasets)
    cols = min(4, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), squeeze=False)
    for i, (ds, kind, pg, ae) in enumerate(datasets):
        ax = axes[i // cols][i % cols]
        if pg:
            d = _load(pg)
            cert = d.get("PGNNCert-E", {}).get("certified") if d else None
            if cert:
                xs, ys = _curve_to_xy(cert)
                ax.plot(xs, ys, "o-", color="#888", label="PGNNCert-E (reproduced)")
        ad = _load(ae)
        if ad and ad.get("curve"):
            xs, ys = _curve_to_xy(ad["curve"])
            ax.plot(xs, ys, "s-", color="#1f77b4", label="Aethelred (ours)")
        ax.set_title(f"{ds} ({kind})"); ax.set_xlabel("edge budget B")
        ax.set_ylabel("certified accuracy"); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("Certified prediction accuracy vs edge-perturbation budget", y=1.0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIGDIR}/cert_prediction_vs_pgnncert.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[fig] saved cert_prediction_vs_pgnncert ({n} datasets)")


def fig_explanation_cert():
    """XGNNCert vs Aethelred: certified explanation overlap vs budget, per dataset."""
    datasets = []
    for ds in XGNN_DS:
        xg = f"{ROOT}/results/phase2_xgnncert_{ds}.json"
        ae = f"{ROOT}/results/phase2_aethelred_{ds}.json"
        if os.path.exists(xg) or os.path.exists(ae):
            datasets.append((ds, xg, ae))
    if not datasets:
        print("[fig] explanation-cert: no data yet")
        return
    n = len(datasets); cols = min(4, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), squeeze=False)
    for i, (ds, xg, ae) in enumerate(datasets):
        ax = axes[i // cols][i % cols]
        xd = _load(xg)
        if xd and xd.get("expl_stability_frac"):
            xs, ys = _curve_to_xy(xd["expl_stability_frac"])
            ax.plot(xs, ys, "o-", color="#888", label="XGNNCert (reproduced)")
        ad = _load(ae)
        if ad and ad.get("curve"):
            xs, ys = _curve_to_xy(ad["curve"])
            ax.plot(xs, ys, "s-", color="#d62728", label="Aethelred (ours)")
        ax.set_title(ds); ax.set_xlabel("edge budget B")
        ax.set_ylabel("certified expl. overlap"); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("Certified explanation overlap vs edge-perturbation budget", y=1.0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIGDIR}/cert_explanation_vs_xgnncert.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[fig] saved cert_explanation_vs_xgnncert ({n} datasets)")


if __name__ == "__main__":
    fig_prediction_cert()
    fig_explanation_cert()
    print("[fig] done ->", FIGDIR)
