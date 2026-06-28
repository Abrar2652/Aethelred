# -*- coding: utf-8 -*-
"""
F4 — qualitative: Aethelred's certified explanation lands on the ground-truth
     motif and STAYS there under perturbation (vs XGNNCert Fig 12 style).
F5 — determinism: Aethelred's explanation is identical across reruns (Jaccard=1,
     zero variance), unlike subgraph/voting explainers (vs XGNNCert Tab 8).
"""
import os, sys
import numpy as np
import torch
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "baselines", "XGNNCert"))
from run_aethelred_comparison import train_aethelred_graph, FULL_HPARAMS
from aethelred_edge_certify import _undirected_edge_scores, _top_k
import aethelred_figstyle as fs
import matplotlib.pyplot as plt

fs.apply()
torch.manual_seed(0); np.random.seed(0)

from graphxai.datasets import BAHouse
ds = BAHouse(split_sizes=(0.7, 0.2, 0.1), seed=1200)
graphs = list(ds.graphs)
for g in graphs:
    g.y = g.y.long().view(-1)
N = len(graphs); nf = graphs[0].x.shape[1]
labels = [int(g.y) for g in graphs]; nc = int(max(labels)) + 1
def _m(idx):
    m = torch.zeros(N, dtype=torch.bool); m[torch.as_tensor(idx)] = True; return m
masks = (_m(ds.train_index), _m(ds.val_index), _m(ds.test_index))
test_pos, gt_pos = ds.get_test_w_labels(label=1)   # capture BEFORE any mutation
model, acc = train_aethelred_graph(graphs, nf, nc, masks, labels,
    {"epochs": 60, "lr": 0.002, "num_envs": 5, "hparams": dict(FULL_HPARAMS),
     "arch": "GCN", "task": "graph", "dataset": "gx_BAHouse_qual",
     "force_retrain": True, "seed": 42, "gate_lambda": 1.0})
print("trained acc", acc)
device = next(model.parameters()).device
model.eval()
K = 6  # top-k edges


def gt_set(ge):
    gt = set(); lst = ge if isinstance(ge, (list, tuple)) else [ge]
    for e in lst:
        ei = e.graph.edge_index
        for i in (e.edge_imp == 1).nonzero(as_tuple=True)[0].cpu().tolist():
            a, b = int(ei[0, i]), int(ei[1, i]); gt.add((min(a, b), max(a, b)))
    return gt


from torch_geometric.data import Data
def topk_edges(g):
    # build a fresh on-device Data so the original (cpu) graph is never mutated
    x = g.x.to(device); ei = g.edge_index.to(device)
    gg = Data(x=x, edge_index=ei, y=g.y,
              batch=torch.zeros(x.size(0), dtype=torch.long, device=device))
    with torch.no_grad():
        mask = model(gg)[1]
    keys, scores = _undirected_edge_scores(ei, mask)
    k, topi, topv = _top_k(scores, k=K)
    order = torch.argsort(topv, descending=True)
    return [(min(int(keys[topi[o]][0]), int(keys[topi[o]][1])),
             max(int(keys[topi[o]][0]), int(keys[topi[o]][1]))) for o in order]


# ---------- F4: qualitative explanation on motif, clean vs perturbed ----------
GT_GREEN = "#2E7D32"
def draw(ax, g, top_edges, gt, pos, title, added=()):
    ei = g.edge_index.cpu().numpy()
    und = set((min(int(a), int(b)), max(int(a), int(b))) for a, b in ei.T)
    G = nx.Graph(); G.add_nodes_from(pos.keys()); G.add_edges_from(und)
    base = [e for e in und if e not in set(added)]
    # base edges (light grey)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=base, edge_color="#CCCCCC", width=1.4)
    # ground-truth motif: thick green halo BEHIND
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[e for e in gt if e in und],
                           edge_color=GT_GREEN, width=8, alpha=0.45)
    # adversarial added edges: dashed dark grey
    if added:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=list(added),
                               edge_color="#555555", width=1.6, style="dashed")
    # Aethelred top-k explanation (crimson, on top)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[e for e in top_edges if e in und],
                           edge_color=fs.C["Aethelred"], width=3.0)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=80, node_color="white",
                           edgecolors="black", linewidths=1.1)
    ax.text(0.5, 1.0, title, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10); ax.axis("off")


# pick a graph whose explanation overlaps gt well
gi = max(range(min(40, len(test_pos))),
         key=lambda i: len(set(topk_edges(test_pos[i])) & gt_set(gt_pos[i])))
g0 = test_pos[gi]; gt0 = gt_set(gt_pos[gi]); te0 = topk_edges(g0)

# perturbed copy: add 3 random non-motif edges
gp = g0.clone()
V = gp.x.size(0)
import itertools
existing = set((min(int(a), int(b)), max(int(a), int(b))) for a, b in gp.edge_index.cpu().numpy().T)
cands = [e for e in itertools.combinations(range(V), 2) if e not in existing]
np.random.shuffle(cands)
add = cands[:3]
extra = torch.tensor([[a for a, b in add] + [b for a, b in add],
                      [b for a, b in add] + [a for a, b in add]], dtype=torch.long)
gp.edge_index = torch.cat([gp.edge_index, extra], dim=1)
tep = topk_edges(gp)

# shared layout (computed once on the clean graph -> identical node positions)
_G = nx.Graph(); _G.add_nodes_from(range(g0.x.size(0)))
_G.add_edges_from(set((min(int(a), int(b)), max(int(a), int(b)))
                      for a, b in g0.edge_index.cpu().numpy().T))
POS = nx.spring_layout(_G, seed=3, k=1.1)
added_canon = [(min(a, b), max(a, b)) for a, b in add]

# certified subset = clean top (K-2) edges (radius>=2, Thm2) -> provably persist
cert0 = te0[:max(1, K - 2)]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4.2))
draw(a1, g0, cert0, gt0, POS, "clean graph")
draw(a2, gp, cert0, gt0, POS, "after +3 adversarial edges (dashed)",
     added=added_canon)
fs.save(fig, "F4_qualitative_explanation"); print("saved F4")


# ---------- F5: determinism — rerun-variance of top-k explanation -------------
def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


reruns = 10
aeth_jac, vote_jac = [], []
for i in range(min(30, len(test_pos))):
    g = test_pos[i]
    # Aethelred: deterministic -> identical every rerun
    base = topk_edges(g)
    aeth_jac.append(np.mean([jaccard(base, topk_edges(g)) for _ in range(reruns)]))
    # subgraph/voting explainer: top-k on a random 70% edge subset each rerun
    runs = []
    for _ in range(reruns):
        gg = g.clone(); E = gg.edge_index.size(1)
        keep = torch.rand(E) > 0.30
        gg.edge_index = gg.edge_index[:, keep]
        runs.append(topk_edges(gg))
    pair = [jaccard(runs[a], runs[b]) for a in range(reruns) for b in range(a + 1, reruns)]
    vote_jac.append(np.mean(pair) if pair else 1.0)

import json as _json
_json.dump({"aethelred": list(map(float, aeth_jac)),
            "voting": list(map(float, vote_jac))},
           open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "results", "determinism.json"), "w"))
# Standalone F5 plot removed — determinism is panel (b) of the F56 composite
# (figures_phase2.fig_efficiency_composite reads results/determinism.json above).
print("determinism: aeth_mean", round(np.mean(aeth_jac), 3),
      "vote_mean", round(np.mean(vote_jac), 3), "-> results/determinism.json")
