# -*- coding: utf-8 -*-
"""
Generate Aethelred's certified curves for the head-to-head overlays (torch 2.4).

Per GRAPH dataset, trains Aethelred (robust mode = spanning-tree environments, the
analogue of PGNNCert's sub-graph training) and emits:
  * PREDICTION cert curve (aethelred_pred_certify): certified accuracy vs edge
    budget B at PGNNCert's budgets -> results/phase2_aethelred_predcert_<ds>.json
  * EXPLANATION cert curve (aethelred_edge_certify): certified overlap vs B
    -> results/phase2_aethelred_<ds>.json

Usage:
  python run_aethelred_certcurves.py --dataset MUTAG --epochs 150 --T 50
"""
import os, json, argparse
import torch

from datasets.dataset_loader import load_graph_data
from run_aethelred_comparison import train_aethelred_graph, FULL_HPARAMS
import aethelred_pred_certify as pc
import aethelred_edge_certify as ec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MUTAG")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--robust", type=int, default=1, help="1=spanning-tree envs (match PGNNCert subgraph training)")
    ap.add_argument("--pred_budgets", type=int, nargs="+", default=[0,1,2,3,5,10,15,20,25,30])
    ap.add_argument("--expl_budgets", type=int, nargs="+", default=[0,1,2,3,4,6,8,10])
    ap.add_argument("--top_k_frac", type=float, default=0.25)
    ap.add_argument("--max_graphs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    graphs, nf, nc, masks, labels = load_graph_data(args.dataset)
    _, _, test_mask = masks
    test_graphs = [graphs[i] for i in range(len(graphs)) if test_mask[i]]
    if args.max_graphs:
        test_graphs = test_graphs[:args.max_graphs]

    train_args = {
        "epochs": args.epochs, "lr": args.lr,
        "num_envs": 8 if args.robust else 5,
        "use_spanning_tree": bool(args.robust),
        "robust": bool(args.robust),
        "hparams": dict(FULL_HPARAMS), "arch": "GCN", "task": "graph",
        "dataset": args.dataset, "force_retrain": True, "seed": args.seed,
        "gate_lambda": 1.0,
    }
    print(f"[certcurves] training Aethelred(GCN, robust={args.robust}) on {args.dataset} ...")
    model, test_acc = train_aethelred_graph(graphs, nf, nc, masks, labels, train_args)
    print(f"[certcurves] {args.dataset} clean test_acc={test_acc:.4f}")

    # --- PREDICTION cert (vs PGNNCert) ---
    pred_curve, npred = pc.certified_accuracy_curve(
        model, test_graphs, args.pred_budgets, T=args.T, verbose=False)
    out_pred = {"method": "Aethelred", "dataset": args.dataset, "task": "graph",
                "test_acc": test_acc, "T": args.T, "n_test": npred,
                "budgets": list(args.pred_budgets),
                "curve": {str(b): pred_curve[int(b)] for b in args.pred_budgets}}
    with open(f"results/phase2_aethelred_predcert_{args.dataset}.json", "w") as f:
        json.dump(out_pred, f, indent=2)
    print(f"[certcurves] pred-cert: {out_pred['curve']}")

    # --- EXPLANATION cert (vs XGNNCert; self-stability on this dataset) ---
    sums = {int(b): 0.0 for b in args.expl_budgets}; ks = []; n = 0
    device = next(model.parameters()).device
    for g in test_graphs:
        g = g.to(device)
        rep = ec.certify_edge_explanation(model, g, top_k_frac=args.top_k_frac, verbose=False)
        k = rep["k"]
        if k <= 0:
            continue
        ks.append(k)
        for b in args.expl_budgets:
            sums[int(b)] += max(0, k - int(b)) / k
        n += 1
    expl_curve = {str(b): (sums[int(b)] / n if n else 0.0) for b in args.expl_budgets}
    out_expl = {"method": "Aethelred", "dataset": args.dataset,
                "top_k_frac": args.top_k_frac, "mean_k": (sum(ks)/len(ks) if ks else 0),
                "n_test": n, "budgets": list(args.expl_budgets), "curve": expl_curve}
    with open(f"results/phase2_aethelred_{args.dataset}.json", "w") as f:
        json.dump(out_expl, f, indent=2)
    print(f"[certcurves] expl-cert: {expl_curve}")
    print(f"[certcurves] DONE {args.dataset}")


if __name__ == "__main__":
    main()
