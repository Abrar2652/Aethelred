# -*- coding: utf-8 -*-
"""
Aethelred certified FAITHFULNESS on graphxai ground-truth datasets — the head-to-
head against XGNNCert on the SAME data/metric (Benzene/FC + ShapeGGen synthetic).

Trains Aethelred (graph classification) on a graphxai dataset, then on each
POSITIVE test graph computes the certified explanation (aethelred_edge_certify,
deterministic) and its overlap with the ground-truth motif edges:

  * precision@k / recall    : top-k explanation vs ground-truth (faithfulness)
  * certified stability @B  : fraction of top-k edges certified (= (k-B)/k, Thm 2)
  * certified faithfulness@B: ground-truth edges among the certified-stable ones
                              (the edges guaranteed to stay under <=B perturbations)

Output mirrors run_xgnncert_baseline so make_figures can overlay them directly:
  results/phase2_aethelred_faith_<ds>.json

Usage:
  python run_aethelred_graphxai_faith.py --dataset BAHouse --epochs 80 --k 6
"""
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "baselines", "XGNNCert"))
from run_aethelred_comparison import train_aethelred_graph, FULL_HPARAMS
from aethelred_edge_certify import _undirected_edge_scores, _top_k


def _load_graphxai(name, seed):
    from graphxai.datasets import (Benzene, BAHouse, BADiamond, BAWheel, BACycle)
    from graphxai.datasets import FluorideCarbonyl as FC, AlkaneCarbonyl as AC
    table = dict(Benzene=Benzene, BAHouse=BAHouse, BADiamond=BADiamond,
                 BAWheel=BAWheel, BACycle=BACycle, FC=FC, AC=AC)
    return table[name](split_sizes=(0.7, 0.2, 0.1), seed=seed)


def _gt_edges(ge):
    gt = set()
    lst = ge if isinstance(ge, (list, tuple)) else [ge]
    for e in lst:
        ei = e.graph.edge_index
        pos = (e.edge_imp == 1).nonzero(as_tuple=True)[0].cpu().tolist()
        for i in pos:
            a, b = int(ei[0, i]), int(ei[1, i])
            gt.add((min(a, b), max(a, b)))
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="BAHouse")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--budgets", type=int, nargs="+", default=[0, 1, 2, 3, 4, 6, 8, 10])
    ap.add_argument("--max_test", type=int, default=80)
    ap.add_argument("--seed", type=int, default=1200)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    ds = _load_graphxai(args.dataset, args.seed)
    graphs = list(ds.graphs)
    for g in graphs:
        g.y = g.y.long().view(-1) if torch.is_tensor(g.y) else torch.tensor([int(g.y)])
    N = len(graphs)
    nf = graphs[0].x.shape[1]
    labels = [int(g.y) for g in graphs]
    nc = int(max(labels)) + 1

    def _mask(idx):
        m = torch.zeros(N, dtype=torch.bool)
        if idx is not None:
            m[torch.as_tensor(idx)] = True
        return m
    masks = (_mask(ds.train_index), _mask(ds.val_index), _mask(ds.test_index))
    print(f"[aeth-faith] {args.dataset}: N={N} dim={nf} classes={nc} "
          f"train={int(masks[0].sum())} test={int(masks[2].sum())}")

    train_args = {"epochs": args.epochs, "lr": args.lr, "num_envs": 5,
                  "hparams": dict(FULL_HPARAMS), "arch": "GCN", "task": "graph",
                  "dataset": f"gx_{args.dataset}", "force_retrain": True,
                  "seed": args.seed, "gate_lambda": 1.0}
    model, test_acc = train_aethelred_graph(graphs, nf, nc, masks, labels, train_args)
    print(f"[aeth-faith] {args.dataset} test_acc={test_acc:.4f}")
    device = next(model.parameters()).device
    model.eval()

    test_pos, gt_pos = ds.get_test_w_labels(label=1)
    budgets = list(args.budgets)
    stab = {B: 0.0 for B in budgets}
    faith = {B: 0.0 for B in budgets}
    prec = rec = 0.0
    n = 0
    for gi in range(min(args.max_test, len(test_pos))):
        g = test_pos[gi].to(device)
        gt = _gt_edges(gt_pos[gi])
        if not gt:
            continue
        k_real = len(gt)
        with torch.no_grad():
            out = model(g)
            mask = out[1] if isinstance(out, (tuple, list)) else out
        keys, scores = _undirected_edge_scores(g.edge_index, mask)
        k, topi, topv = _top_k(scores, k=args.k)
        # top edges, strongest first
        order = torch.argsort(topv, descending=True)
        top_edges = [(int(keys[topi[o]][0]), int(keys[topi[o]][1])) for o in order]
        top_edges = [(min(a, b), max(a, b)) for a, b in top_edges]
        inter = sum(1 for e in top_edges if e in gt)
        prec += inter / max(1, k)
        rec += inter / k_real
        for B in budgets:
            # Thm2: rank-r edge (1-indexed) certified for B<=k-r -> top (k-B) edges certified
            ncert = max(0, k - int(B))
            cert_edges = top_edges[:ncert]
            stab[B] += ncert / max(1, k)
            faith[B] += sum(1 for e in cert_edges if e in gt) / k_real
        n += 1

    out = {"method": "Aethelred", "dataset": args.dataset, "task": "graph",
           "test_acc": test_acc, "k": args.k, "n_test": n, "budgets": budgets,
           "faithfulness_precision_at_k": prec / max(1, n),
           "faithfulness_recall_at_k": rec / max(1, n),
           "expl_stability_frac": {str(B): stab[B] / max(1, n) for B in budgets},
           "certified_faithfulness_recall": {str(B): faith[B] / max(1, n) for B in budgets}}
    print("[aeth-faith] RESULT", json.dumps(out, indent=2))
    os.makedirs("results", exist_ok=True)
    with open(f"results/phase2_aethelred_faith_{args.dataset}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[aeth-faith] saved -> results/phase2_aethelred_faith_{args.dataset}.json")


if __name__ == "__main__":
    main()
