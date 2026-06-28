# -*- coding: utf-8 -*-
"""
Phase-2 paper figures, built on aethelred_figstyle. Reads result JSONs (reproduced
+ published SOTA) and emits the head-to-head comparison plots into figures/.

F1  Certified PREDICTION accuracy vs edge budget — Aethelred vs PGNNCert-E/-N
    (reproduced) vs Bagging (reported). Small-multiples over graph datasets.
"""
import os, json, glob
import numpy as np
import aethelred_figstyle as fs

ROOT = os.path.dirname(os.path.abspath(__file__))


def _load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def _xy(curve_dict, maxb=None):
    pts = sorted((int(k), float(v)) for k, v in curve_dict.items())
    if maxb is not None:
        pts = [(x, y) for x, y in pts if x <= maxb]
    return [x for x, _ in pts], [y for _, y in pts]


def fig1_cert_prediction():
    """SECONDARY: certified prediction (test-time edge perturbation). Baselines
    from PGNNCert's authoritative Table 2; restricted to PROTEINS/DD where the
    edge-cert protocol is clean (AIDS/MUTAG omitted — sparse-subgraph vote
    degenerates on tiny graphs, inflating margins for both methods)."""
    fs.apply()
    sota = _load(os.path.join(ROOT, "results/sota_published.json"))
    t2 = sota["PGNNCert"]["table2_certified_graph_accuracy_vs_Bagging"]
    xb = t2["_columns_edges"]                      # [0,5,10,15] edges
    datasets = ["PROTEINS", "DD"]
    fig, axes = fs.grid_fig(len(datasets), ncols=2)
    hl = {}
    for idx, ds in enumerate(datasets):
        ax = axes[0][idx]
        ln, = ax.plot(xb, t2[ds]["PGNNCert-E"], **fs.style("PGNNCert-E")); hl["PGNNCert-E"] = ln
        ln, = ax.plot(xb, t2[ds]["PGNNCert-N"], **fs.style("PGNNCert-N")); hl["PGNNCert-N"] = ln
        ln, = ax.plot(xb, t2[ds]["Bagging"], **fs.style("Bagging")); hl["Bagging"] = ln
        ap = _load(os.path.join(ROOT, f"_ref_pgnncert/results/phase2_aethelred_predcert_{ds}.json")) \
            or _load(os.path.join(ROOT, f"results/phase2_aethelred_predcert_{ds}.json"))
        if ap and "curve" in ap:
            xs, ys = _xy(ap["curve"], maxb=15)
            ln, = ax.plot(xs, ys, **fs.style("Aethelred")); hl["Aethelred"] = ln
        ax.text(0.96, 0.94, ds, transform=ax.transAxes, ha="right", va="top",
                fontsize=10, fontweight="bold")
        ax.set_xlabel("perturbed edges  B")
        ax.set_ylabel("certified accuracy"); ax.set_ylim(-0.02, 1.0)
    fig.legend(hl.values(), hl.keys(), loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.05))
    print("saved", fs.save(fig, "F1_certified_prediction_vs_budget"))


def fig1b_node_injection_sota():
    """Node-injection certified accuracy — full SOTA field across all 4 node
    datasets (PGNNCert-E/-N, Bi-RS-Include/-Exclude, RS, Bagging). Precise,
    validated digitization from PGNNCert Fig 7 (figures/pgnncert_fig_digitized)."""
    fs.apply()
    d = _load(os.path.join(ROOT, "figures/pgnncert_fig_digitized/fig7_sota_comparison.json"))
    if not d:
        print("F1b: no digitized data"); return
    panels = ["Fig7_Cora-ML", "Fig7_Citeseer", "Fig7_Pubmed", "Fig7_Amazon-C"]
    order = ["PGNNCert-N", "Bi-RS-Exclude", "PGNNCert-E", "Bi-RS-Include", "Bagging", "RS"]
    fig, axes = fs.grid_fig(len(panels), ncols=2)
    hl = {}
    for i, pk in enumerate(panels):
        ax = axes[i // 2][i % 2]
        series = d[pk]["series"]
        for m in order:
            if m in series and series[m]:
                xs, ys = _xy(series[m])
                xs, ys = [x for x in xs if x <= 30], [y for x, y in zip(xs, ys) if x <= 30]
                ln, = ax.plot(xs, ys, **fs.style(m, markevery=4))
                hl[m] = ln
        ax.text(0.96, 0.94, d[pk]["dataset"], transform=ax.transAxes, ha="right",
                va="top", fontsize=10, fontweight="bold")
        ax.set_xlabel("injected nodes")
        ax.set_ylabel("certified accuracy"); ax.set_ylim(-0.02, 0.95)
    fig.legend(hl.values(), hl.keys(), loc="upper center", ncol=6,
               bbox_to_anchor=(0.5, 1.04), fontsize=8.5)
    print("saved", fs.save(fig, "F1b_node_injection_sota"))


def fig2b_under_attack_vs_vinfor():
    """XGNNCert vs V-InfoR explanation robustness under attack (XGNNCert Table 4)."""
    fs.apply()
    sota = _load(os.path.join(ROOT, "results/sota_published.json"))
    t4 = sota["XGNNCert"]["table4_empirical_robustness_under_attack"]
    ds = t4["_datasets_order"]
    # Aethelred under-attack (Benzene/FC) — aligns to those columns
    aeth_acc = [np.nan] * len(ds); aeth_diff = [np.nan] * len(ds)
    for j, dname in enumerate(ds):
        a = _load(os.path.join(ROOT, f"results/phase2_aethelred_attack_{dname}.json"))
        if a:
            aeth_acc[j] = a["explanation_accuracy"]; aeth_diff[j] = a["difference_fraction_pct"]
    import matplotlib.pyplot as plt
    x = np.arange(len(ds)); w = 0.27
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.8, 3.6))
    fs.bar(a1, x - w, t4["explanation_accuracy"]["V-InfoR"], w, "V-InfoR")
    fs.bar(a1, x, t4["explanation_accuracy"]["XGNNCert"], w, "XGNNCert")
    fs.bar(a1, x + w, aeth_acc, w, "Aethelred")
    a1.set_ylabel("explanation accuracy  ($\\uparrow$)")
    a1.set_xticks(x); a1.set_xticklabels(ds, rotation=20, ha="right"); a1.legend()
    fs.bar(a2, x - w, t4["difference_fraction_pct"]["V-InfoR"], w, "V-InfoR")
    fs.bar(a2, x, t4["difference_fraction_pct"]["XGNNCert"], w, "XGNNCert")
    fs.bar(a2, x + w, aeth_diff, w, "Aethelred")
    a2.set_ylabel("explanation change %  ($\\downarrow$)")
    a2.set_xticks(x); a2.set_xticklabels(ds, rotation=20, ha="right"); a2.legend()
    print("saved", fs.save(fig, "F2b_explanation_robustness_vs_VInfoR"))


def fig_unified_faithfulness():
    """UNIFIED faithfulness: Aethelred vs ALL explanation SOTA in one figure —
    XGNNCert, PGExplainer, ReFine, GSAT (precision vs ground-truth motif).
    Aethelred/XGNNCert reproduced; base explainers from XGNNCert Table 1 (orig)."""
    fs.apply()
    sota = _load(os.path.join(ROOT, "results/sota_published.json"))
    t1 = sota["XGNNCert"]["table1_explanation_accuracy"]
    SG = {"BAHouse": "SG+House", "BADiamond": "SG+Diamond", "BAWheel": "SG+Wheel",
          "Benzene": "Benzene", "FC": "FC"}
    datasets = ["BAHouse", "BADiamond", "BAWheel", "Benzene", "FC"]
    methods = ["Aethelred", "XGNNCert", "PGExplainer", "ReFine", "GSAT"]
    vals = {m: [] for m in methods}
    for ds in datasets:
        sg = SG[ds]
        a = _load(os.path.join(ROOT, f"results/phase2_aethelred_faith_{ds}.json"))
        x = _load(os.path.join(ROOT, f"results/phase2_xgnncert_{ds}.json"))
        vals["Aethelred"].append(a["faithfulness_precision_at_k"] if a else 0)
        vals["XGNNCert"].append(t1["PGExplainer"][sg][3])  # authoritative published T70
        vals["PGExplainer"].append(t1["PGExplainer"][sg][0])   # orig
        vals["ReFine"].append(t1["ReFine"][sg][0])
        vals["GSAT"].append(t1["GSAT"][sg][0])

    import matplotlib.pyplot as plt
    x = np.arange(len(datasets)); w = 0.16
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    for i, m in enumerate(methods):
        fs.bar(ax, x + (i - 2) * w, vals[m], w, m)
    ax.set_xticks(x); ax.set_xticklabels(datasets)
    ax.set_ylabel("explanation faithfulness  (precision@k)")
    ax.set_ylim(0, 0.85); ax.legend(ncol=5, fontsize=8.5, loc="upper center",
                                    bbox_to_anchor=(0.5, 1.13))
    print("saved", fs.save(fig, "F2c_unified_faithfulness"))


def fig2_cert_faithfulness():
    """Certified faithfulness (ground-truth recall) vs edge budget — Aethelred vs
    XGNNCert on the same graphxai datasets. Small-multiples."""
    fs.apply()
    datasets = ["BAHouse", "BADiamond", "BAWheel", "Benzene", "FC"]
    avail = []
    for ds in datasets:
        a = _load(os.path.join(ROOT, f"results/phase2_aethelred_faith_{ds}.json"))
        x = _load(os.path.join(ROOT, f"results/phase2_xgnncert_{ds}.json"))
        if a or x:
            avail.append((ds, a, x))
    if not avail:
        print("F2: no faithfulness data yet"); return
    fig, axes = fs.grid_fig(len(avail), ncols=3)
    hl = {}
    for i, (ds, a, x) in enumerate(avail):
        ax = axes[i // 3][i % 3]
        if a and "certified_faithfulness_recall" in a:
            xs, ys = _xy(a["certified_faithfulness_recall"])
            ln, = ax.plot(xs, ys, **fs.style("Aethelred")); hl["Aethelred"] = ln
        if x and x.get("base_explainer") == "PGExplainer" and \
                "certified_faithfulness_recall" in x:
            xs, ys = _xy(x["certified_faithfulness_recall"])
            ln, = ax.plot(xs, ys, **fs.style("XGNNCert")); hl["XGNNCert"] = ln
        ax.set_title(ds); ax.set_xlabel("perturbed edges  B")
        ax.set_ylabel("certified faithfulness"); ax.set_ylim(-0.02, 1.0)
    for j in range(len(avail), axes.size):
        axes[j // 3][j % 3].axis("off")
    fig.legend(hl.values(), hl.keys(), loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.07))
    print("saved", fs.save(fig, "F2_certified_faithfulness_vs_budget"))


def fig3_frontier():
    """Faithfulness–stability frontier: x = certified stability (fraction of
    ground-truth edges guaranteed at budget B=2), y = clean faithfulness
    (precision@k). XGNNCert from AUTHORITATIVE published values (Table 1 +
    M_λ frontier). Aethelred should sit top-right (stable AND faithful)."""
    fs.apply()
    sota = _load(os.path.join(ROOT, "results/sota_published.json"))
    t1 = sota["XGNNCert"]["table1_explanation_accuracy"]
    ref = _load(os.path.join(ROOT, "figures/xgnncert_fig_digitized/xgnncert_default_T70_reference.json"))
    KGT = {"BAHouse": 6, "BADiamond": 5, "BAWheel": 8, "Benzene": 6, "FC": 5}
    SG = {"BAHouse": "SG+House", "BADiamond": "SG+Diamond", "BAWheel": "SG+Wheel",
          "Benzene": "Benzene", "FC": "FC"}
    from adjustText import adjust_text
    fig, ax = fs.new_fig(5.8, 4.8)
    texts = []; seen = set()
    for ds in ["BAHouse", "BADiamond", "BAWheel", "Benzene", "FC"]:
        k_gt = KGT[ds]; sg = SG[ds]
        a = _load(os.path.join(ROOT, f"results/phase2_aethelred_faith_{ds}.json"))
        pts = []
        if a:
            pts.append(("Aethelred", a.get("certified_faithfulness_recall", {}).get("2", 0),
                        a["faithfulness_precision_at_k"]))
        xref = (ref or {}).get("PGExplainer_" + SG[ds], {})
        if xref:
            xstab = max([j for j in range(1, k_gt + 1) if xref.get(str(j), 0) >= 2], default=0) / k_gt
            pts.append(("XGNNCert", xstab, t1["PGExplainer"][sg][3]))
        scatters = []
        for meth, stab, faith in pts:
            kw = dict(color=fs.C[meth], edgecolor="black", linewidth=0.8, s=95,
                      marker="o" if meth == "Aethelred" else "D", zorder=5)
            scatters.append(ax.scatter(stab, faith,
                                       label=(meth if meth not in seen else None), **kw))
            texts.append(ax.text(stab, faith, ds, fontsize=7.5, zorder=6,
                                 bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                           ec="none", alpha=0.75)))
            seen.add(meth)
    # push labels off every marker, with leader lines
    adjust_text(texts, ax=ax, force_text=(0.8, 1.4), force_static=(0.5, 0.9),
                expand=(1.8, 2.2), max_move=40,
                arrowprops=dict(arrowstyle="-", color="#9A9A9A", lw=0.6))
    ax.set_xlabel("certified stability  (g.t. edges guaranteed @ B=2)")
    ax.set_ylabel("faithfulness  (precision@k)")
    ax.set_xlim(-0.02, 1.0); ax.set_ylim(-0.02, 0.85)
    ax.legend(loc="lower left")
    print("saved", fs.save(fig, "F3_faithfulness_stability_frontier"))


def fig9_spmotif_thesis():
    """THE thesis figure: as spurious-correlation bias rises, the non-causal
    explainer (GNNExplainer) drifts onto the spurious feature (faithfulness
    collapses), DIR partly resists, and Aethelred stays bias-invariant on the
    CAUSAL motif. Aethelred = computed (table6); DIR/GNNExplainer = published
    DIR-paper Table 6 (precision@K vs ground-truth motif)."""
    fs.apply()
    import matplotlib.pyplot as plt
    biasx = [0.33, 0.50, 0.70, 0.90]
    a = _load(os.path.join(ROOT, "results/table6.json"))["agg"]["Aethelred"]
    keys = ["Balance", "b=0.50", "b=0.70", "b=0.90"]
    aeth = [a[k][0] for k in keys]; aeth_s = [a[k][1] for k in keys]
    # published DIR-paper Table 6 (mean, std)
    DIR = {"m": [0.257, 0.255, 0.247, 0.192], "s": [0.014, 0.016, 0.012, 0.044]}
    GNX = {"m": [0.249, 0.203, 0.167, 0.066], "s": [0.011, 0.019, 0.039, 0.007]}
    fig, ax = fs.new_fig(5.6, 4.2)

    def band(xs, m, s, style):
        m = np.array(m); s = np.array(s)
        ln, = ax.plot(xs, m, **style)
        ax.fill_between(xs, m - s, m + s, color=style["color"], alpha=0.15, lw=0)
        return ln
    band(biasx, GNX["m"], GNX["s"], fs.style("GNNExplainer-spm", label="GNNExplainer (non-causal)"))
    band(biasx, DIR["m"], DIR["s"], fs.style("DIR", label="DIR (causal)"))
    band(biasx, aeth, aeth_s, fs.style("Aethelred"))
    ax.set_xlabel("spurious-correlation bias")
    ax.set_ylabel("explanation faithfulness  (precision@K vs causal motif)")
    ax.set_xticks(biasx); ax.set_ylim(0, 0.40)
    ax.legend(loc="lower left")
    print("saved", fs.save(fig, "F9_spmotif_causal_vs_spurious"))


def fig7_mstar_vs_lambda():
    """M_λ (certified perturbation size) vs λ (# ground-truth edges guaranteed) —
    XGNNCert's headline metric/convention. Aethelred (computed from our certified
    faithfulness) vs XGNNCert's AUTHORITATIVE PUBLISHED frontier (default T=70,
    PGExplainer; figures/xgnncert_fig_digitized). Higher M_λ = stronger."""
    fs.apply()
    ref = _load(os.path.join(ROOT, "figures/xgnncert_fig_digitized/xgnncert_default_T70_reference.json"))
    KGT = {"BAHouse": 6, "BADiamond": 5, "BAWheel": 8, "Benzene": 6, "FC": 5}
    SG = {"BAHouse": "SG+House", "BADiamond": "SG+Diamond", "BAWheel": "SG+Wheel",
          "Benzene": "Benzene", "FC": "FC"}
    datasets = ["Benzene", "FC", "BAHouse", "BADiamond", "BAWheel"]
    fig, axes = fs.grid_fig(len(datasets), ncols=3)
    hl = {}
    for i, ds in enumerate(datasets):
        ax = axes[i // 3][i % 3]
        a = _load(os.path.join(ROOT, f"results/phase2_aethelred_faith_{ds}.json"))
        k_gt = KGT[ds]
        if a and "certified_faithfulness_recall" in a:
            rec = {int(k): v for k, v in a["certified_faithfulness_recall"].items()}
            lam = list(range(1, k_gt + 1))
            ms = [max([B for B, r in rec.items() if r >= j / k_gt], default=0) for j in lam]
            ln, = ax.plot(lam, ms, **fs.style("Aethelred")); hl["Aethelred"] = ln
        xref = (ref or {}).get("PGExplainer_" + SG[ds], {})
        if xref:
            lam = sorted(int(k) for k in xref)
            ln, = ax.plot(lam, [xref[str(j)] for j in lam], **fs.style("XGNNCert"))
            hl["XGNNCert"] = ln
        tag = "" if ds in ("Benzene", "FC") else "  (BA≈SG)"
        ax.text(0.96, 0.94, ds + tag, transform=ax.transAxes, ha="right", va="top",
                fontsize=9.5, fontweight="bold")
        ax.set_xlabel("λ  (g.t. edges guaranteed)")
        ax.set_ylabel("M_λ  (cert. perturb. size)")
    for j in range(len(datasets), axes.size):
        axes[j // 3][j % 3].axis("off")
    fig.legend(hl.values(), hl.keys(), loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.06))
    print("saved", fs.save(fig, "F7_Mlambda_vs_lambda"))


def fig_efficiency_composite():
    """Composite: (a) certification cost O(1) vs O(T); (b) determinism — zero
    rerun variance. Two single-panel plots that pair cleanly."""
    fs.apply()
    import matplotlib.pyplot as plt
    det = _load(os.path.join(ROOT, "results/determinism.json"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.8))
    # (a) cost
    T = np.arange(10, 101, 5)
    a1.plot(T, 2 * T, **fs.style("XGNNCert", label="XGNNCert"))
    a1.plot(T, T, **fs.style("PGNNCert", label="PGNNCert"))
    a1.plot(T, np.ones_like(T), **fs.style("Aethelred", label="Aethelred"))
    a1.set_yscale("log"); a1.set_xlabel("number of subgraphs  T")
    a1.set_ylabel("cert. cost (passes / graph)")
    a1.text(0.04, 0.96, "(a)", transform=a1.transAxes, va="top", fontweight="bold")
    a1.legend()
    # (b) determinism
    if det:
        bp = a2.boxplot([det["aethelred"], det["voting"]], patch_artist=True,
                        widths=0.55, medianprops=dict(color="black", linewidth=1.4))
        for patch, col in zip(bp["boxes"], [fs.C["Aethelred"], fs.C["XGNNCert"]]):
            patch.set_facecolor(col); patch.set_edgecolor("black"); patch.set_linewidth(1.0)
        a2.set_xticklabels(["Aethelred", "subgraph-voting"])
        a2.set_ylabel("explanation self-agreement\n(Jaccard across reruns)")
        a2.text(0.04, 0.96, "(b)", transform=a2.transAxes, va="top", fontweight="bold")
        a2.set_ylim(0, 1.05)
    print("saved", fs.save(fig, "F56_efficiency_determinism"))


def fig6_cert_cost():
    """Certification cost (explainer/forward calls per certified graph) vs T."""
    fs.apply()
    T = np.arange(10, 101, 5)
    fig, ax = fs.new_fig(5.2, 3.6)
    ax.plot(T, 2 * T, **fs.style("XGNNCert", label="XGNNCert"))   # T explanations + T votes
    ax.plot(T, T, **fs.style("PGNNCert", label="PGNNCert"))        # T classifier votes
    ax.plot(T, np.ones_like(T), **fs.style("Aethelred", label="Aethelred"))  # 1 deterministic pass
    ax.set_yscale("log")
    ax.set_xlabel("number of subgraphs  T  (security parameter)")
    ax.set_ylabel("cert. cost  (passes / graph)")
    ax.set_title("Certification cost: deterministic vs voting")
    ax.legend()
    print("saved", fs.save(fig, "F6_certification_cost"))


if __name__ == "__main__":
    fig1_cert_prediction()
    fig1b_node_injection_sota()
    fig_unified_faithfulness()
    fig2b_under_attack_vs_vinfor()
    fig3_frontier()
    fig_efficiency_composite()   # F56 = cost (a) + determinism (b); supersedes standalone F5/F6
    fig9_spmotif_thesis()
    fig7_mstar_vs_lambda()       # F7 = M_λ vs λ: Aethelred vs XGNNCert AUTHORITATIVE published frontier
