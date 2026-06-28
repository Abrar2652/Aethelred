# -*- coding: utf-8 -*-
"""F8 — sensitivity master (mirrors PGNNCert Figs 3-6 S-sweep + XGNNCert p/gamma).
Train ONCE, certify at many T / many k (cheap, no retrain)."""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "baselines", "XGNNCert"))
from datasets.dataset_loader import load_graph_data
from run_aethelred_comparison import train_aethelred_graph, FULL_HPARAMS
import aethelred_pred_certify as pc
from aethelred_edge_certify import _undirected_edge_scores, _top_k
import aethelred_figstyle as fs
import matplotlib.pyplot as plt

fs.apply()
torch.manual_seed(0); np.random.seed(0)
WARM = ["#F77F00", "#E8112D", "#9C1A4F", "#5A0A2C"]   # warm sequential for T

# ---- Panel A: pred-cert certified accuracy vs budget, T-sweep (on PROTEINS) ----
graphs, nf, nc, masks, labels = load_graph_data("PROTEINS")
_, _, test_mask = masks
test_graphs = [graphs[i] for i in range(len(graphs)) if test_mask[i]]
model, acc = train_aethelred_graph(graphs, nf, nc, masks, labels,
    {"epochs": 120, "lr": 0.002, "num_envs": 8, "use_spanning_tree": True,
     "robust": True, "hparams": dict(FULL_HPARAMS), "arch": "GCN", "task": "graph",
     "dataset": "PROTEINS_sens", "force_retrain": True, "seed": 42, "gate_lambda": 1.0})
print("PROTEINS acc", acc)
budgets = [0, 1, 2, 3, 5, 10, 15, 20]
Tvals = [30, 50, 70, 90]
curvesA = {}
for T in Tvals:
    c, n = pc.certified_accuracy_curve(model, test_graphs, budgets, T=T, verbose=False)
    curvesA[T] = c

# ---- Panel B: faithfulness vs top-k fraction tau (on BAHouse, graphxai) ----
from graphxai.datasets import BAHouse
ds = BAHouse(split_sizes=(0.7, 0.2, 0.1), seed=1200)
g_all = list(ds.graphs)
for g in g_all:
    g.y = g.y.long().view(-1)
N = len(g_all); bnf = g_all[0].x.shape[1]; blab = [int(g.y) for g in g_all]; bnc = max(blab) + 1
def _m(idx):
    m = torch.zeros(N, dtype=torch.bool); m[torch.as_tensor(idx)] = True; return m
bmasks = (_m(ds.train_index), _m(ds.val_index), _m(ds.test_index))
test_pos, gt_pos = ds.get_test_w_labels(label=1)
bmodel, bacc = train_aethelred_graph(g_all, bnf, bnc, bmasks, blab,
    {"epochs": 60, "lr": 0.002, "num_envs": 5, "hparams": dict(FULL_HPARAMS),
     "arch": "GCN", "task": "graph", "dataset": "BAHouse_sens",
     "force_retrain": True, "seed": 42, "gate_lambda": 1.0})
print("BAHouse acc", bacc)
device = next(bmodel.parameters()).device
from torch_geometric.data import Data
def gt_set(ge):
    s = set(); lst = ge if isinstance(ge, (list, tuple)) else [ge]
    for e in lst:
        ei = e.graph.edge_index
        for i in (e.edge_imp == 1).nonzero(as_tuple=True)[0].cpu().tolist():
            a, b = int(ei[0, i]), int(ei[1, i]); s.add((min(a, b), max(a, b)))
    return s
kvals = [3, 4, 6, 8, 10]
precB = []
for k in kvals:
    pr = []
    for i in range(min(80, len(test_pos))):
        g = test_pos[i]; gt = gt_set(gt_pos[i])
        if not gt: continue
        x = g.x.to(device); ei = g.edge_index.to(device)
        gg = Data(x=x, edge_index=ei, y=g.y, batch=torch.zeros(x.size(0), dtype=torch.long, device=device))
        with torch.no_grad(): mask = bmodel(gg)[1]
        keys, scores = _undirected_edge_scores(ei, mask)
        kk, topi, _ = _top_k(scores, k=k)
        edges = [(min(int(keys[topi[j]][0]), int(keys[topi[j]][1])),
                  max(int(keys[topi[j]][0]), int(keys[topi[j]][1]))) for j in range(kk)]
        pr.append(sum(1 for e in edges if e in gt) / kk)
    precB.append(np.mean(pr))

# ---- plot ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.8))
for col, T in zip(WARM, Tvals):
    xs = budgets; ys = [curvesA[T][b] for b in budgets]
    a1.plot(xs, ys, marker="o", color=col, markeredgecolor="black",
            markeredgewidth=0.7, label=f"T={T}", linewidth=2.2)
a1.set_xlabel("perturbed edges  B"); a1.set_ylabel("certified accuracy")
a1.text(0.04, 0.05, "(a) PROTEINS", transform=a1.transAxes, fontweight="bold")
a1.legend(); a1.set_ylim(-0.02, 1.0)
a2.plot(kvals, precB, marker="o", color=fs.C["Aethelred"], markeredgecolor="black",
        markeredgewidth=0.7, linewidth=2.6)
a2.set_xlabel("explanation size  k  (τ)"); a2.set_ylabel("faithfulness (precision@k)")
a2.text(0.04, 0.05, "(b) BAHouse", transform=a2.transAxes, fontweight="bold")
a2.set_ylim(0, 1.0)
fs.save(fig, "F8_sensitivity"); print("saved F8")
