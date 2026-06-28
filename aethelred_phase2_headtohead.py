# -*- coding: utf-8 -*-
"""
Project Aethelred — Phase 2: Certified-Explanation Head-to-Head (vs XGNNCert)
============================================================================

Produces the "certified explanation vs edge-perturbation budget B" curve — the
core comparison figure of the paper — on graph-classification datasets
(MUTAG / AIDS / PROTEINS / DD), the shared benchmark with XGNNCert.

Metric (common to both methods): CERTIFIED EXPLANATION SIZE @ B, normalized.
For a test graph with clean top-k explanation E_k, it is the number of edges
guaranteed to remain in the top-k explanation under ANY edge perturbation of
budget <= B, divided by k. Averaged over the test set.

  * Aethelred (this file, fully implemented): deterministic. By Theorem 2 in
    aethelred_edge_certify, the rank-r edge has certified radius k - r, so the
    certified size @ B is exactly #{r : k - r >= B} = max(0, k - B). Averaging
    max(0, k_g - B) / k_g over test graphs gives the curve. No smoothing, no
    voting, no variance.

  * XGNNCert baseline: hash-subgraph voting certificate (NOT in _ref_pgnncert,
    which is prediction-only). Implemented separately — see the XGNNCert section
    below once the baseline strategy is fixed.

Usage:
    python aethelred_phase2_headtohead.py --dataset MUTAG --epochs 100 \
        --top_k_frac 0.25 --budgets 0 1 2 3 4 6 8 10
"""

import os
import json
import argparse

import torch

from datasets.dataset_loader import load_graph_data
from run_aethelred_comparison import train_aethelred_graph, FULL_HPARAMS
import aethelred_edge_certify as ec


def aethelred_cert_curve(model, test_graphs, budgets, top_k_frac=0.25,
                         max_graphs=None, run_soundness=True):
    """Aethelred certified explanation-size curve over a list of test graphs.

    Returns
    -------
    curve : dict[int, float]     budget B -> mean certified-overlap fraction
    detail: dict                 per-graph k, plus soundness summary
    """
    model.eval()
    device = next(model.parameters()).device
    if max_graphs is not None:
        test_graphs = test_graphs[:max_graphs]

    # Per-graph certified overlap fraction at each budget: max(0, k-B)/k.
    sums = {int(B): 0.0 for B in budgets}
    ks = []
    soundness_fail = 0
    n = 0
    for g in test_graphs:
        g = g.to(device)
        if not hasattr(g, "batch") or g.batch is None:
            g.batch = torch.zeros(g.x.size(0), dtype=torch.long, device=device)
        rep = ec.certify_edge_explanation(model, g, top_k_frac=top_k_frac,
                                          verbose=False)
        k = rep["k"]
        if k <= 0:
            continue
        ks.append(k)
        for B in budgets:
            sums[int(B)] += max(0, k - int(B)) / k
        if run_soundness:
            # Spot-check soundness at a mid budget on a subsample.
            if n < 25:
                s = ec.soundness_check(model, g, budget=max(1, k // 4),
                                       top_k_frac=top_k_frac, seed=n,
                                       verbose=False)
                if not s["passed"]:
                    soundness_fail += 1
        n += 1

    curve = {int(B): (sums[int(B)] / n if n > 0 else 0.0) for B in budgets}
    detail = {
        "n_graphs": n,
        "mean_k": (sum(ks) / len(ks) if ks else 0.0),
        "soundness_checked": min(n, 25),
        "soundness_failures": soundness_fail,
    }
    return curve, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MUTAG")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--top_k_frac", type=float, default=0.25)
    ap.add_argument("--budgets", type=int, nargs="+",
                    default=[0, 1, 2, 3, 4, 6, 8, 10])
    ap.add_argument("--max_graphs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    graphs, nf, nc, masks, labels = load_graph_data(args.dataset)
    train_mask, val_mask, test_mask = masks
    test_graphs = [graphs[i] for i in range(len(graphs)) if test_mask[i]]

    train_args = {
        "epochs": args.epochs, "lr": args.lr, "num_envs": 5,
        "hparams": dict(FULL_HPARAMS), "arch": "GCN", "task": "graph",
        "dataset": args.dataset, "force_retrain": True, "seed": args.seed,
        "gate_lambda": 1.0,
    }
    print(f"[phase2] training Aethelred(GCN) on {args.dataset} "
          f"({args.epochs} epochs, lr={args.lr}) ...")
    model, test_acc = train_aethelred_graph(graphs, nf, nc, masks, labels, train_args)
    print(f"[phase2] {args.dataset} test_acc = {test_acc:.4f}")

    curve, detail = aethelred_cert_curve(
        model, test_graphs, args.budgets, top_k_frac=args.top_k_frac)

    print(f"[phase2] {args.dataset}  mean_k={detail['mean_k']:.1f}  "
          f"n_test={detail['n_graphs']}  "
          f"soundness_failures={detail['soundness_failures']}/"
          f"{detail['soundness_checked']}")
    print("  B   certified-overlap-fraction")
    for B in args.budgets:
        print(f"  {B:<3} {curve[int(B)]:.4f}")

    out = {
        "method": "Aethelred", "dataset": args.dataset, "test_acc": test_acc,
        "top_k_frac": args.top_k_frac, "budgets": list(args.budgets),
        "curve": {str(B): curve[int(B)] for B in args.budgets},
        "detail": detail,
    }
    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", f"phase2_aethelred_{args.dataset}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[phase2] saved -> {path}")


if __name__ == "__main__":
    main()
