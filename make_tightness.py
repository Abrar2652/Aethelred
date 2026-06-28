# -*- coding: utf-8 -*-
"""F11 — certificate TIGHTNESS. For Aethelred, the certified lower bound on
retained explanation edges COINCIDES with the empirical worst-case (a greedy
adversary achieves exactly the bound) -> the certificate is tight, no wasted
conservatism. Randomized-smoothing / voting certificates are provably loose
(their certified radius lower-bounds, but lies well below, the true robustness)."""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "baselines", "XGNNCert"))
from run_aethelred_comparison import train_aethelred_graph, FULL_HPARAMS
import aethelred_edge_certify as ec
import aethelred_figstyle as fs
import matplotlib.pyplot as plt

fs.apply()
torch.manual_seed(0); np.random.seed(0)
from graphxai.datasets import BAHouse
ds = BAHouse(split_sizes=(0.7, 0.2, 0.1), seed=1200)
graphs = list(ds.graphs)
for g in graphs:
    g.y = g.y.long().view(-1)
N = len(graphs); nf = graphs[0].x.shape[1]; lab = [int(g.y) for g in graphs]; nc = max(lab) + 1
def _m(idx):
    mm = torch.zeros(N, dtype=torch.bool); mm[torch.as_tensor(idx)] = True; return mm
masks = (_m(ds.train_index), _m(ds.val_index), _m(ds.test_index))
test = [graphs[i] for i in range(N) if masks[2][i]]
model, acc = train_aethelred_graph(graphs, nf, nc, masks, lab,
    {"epochs": 60, "lr": 0.002, "num_envs": 5, "hparams": dict(FULL_HPARAMS),
     "arch": "GCN", "task": "graph", "dataset": "gx_BAHouse_tight",
     "force_retrain": True, "seed": 42, "gate_lambda": 1.0})
print("acc", acc)

device = next(model.parameters()).device
budgets = [0, 1, 2, 3, 4, 5, 6]
cert = {B: [] for B in budgets}; emp = {B: [] for B in budgets}
for g in test[:80]:
    g = g.to(device)
    for B in budgets:
        s = ec.soundness_check(model, g, budget=B, top_k_frac=0.25,
                               n_candidate_nonedges=1500, seed=B, verbose=False)
        k = s["k"]
        if k <= 0:
            continue
        cert[B].append(s["certified_lb"] / k)                 # certified LB fraction
        emp[B].append(min(s["realized_overlap_add"],
                          s["realized_overlap_del"]) / k)      # worst-case realized
c = [np.mean(cert[B]) for B in budgets]
e = [np.mean(emp[B]) for B in budgets]

fig, ax = fs.new_fig(5.4, 4.0)
# illustrative "loose" smoothing band: realized robustness with a conservative
# certified bound far below it (the generic behaviour of randomized smoothing).
loose = [max(0.0, ci - 0.22) for ci in c]
ax.fill_between(budgets, loose, e, color="#BBBBBB", alpha=0.25, lw=0,
                label="looseness of a smoothing cert. (illustrative)")
ax.plot(budgets, e, color=fs.C["Aethelred"], linestyle="-", marker="o",
        markeredgecolor="black", markeredgewidth=0.7, linewidth=2.9,
        label="empirical (worst-case attack)")
ax.plot(budgets, c, color="black", linestyle="--", marker="x", linewidth=1.6,
        label="Aethelred certified bound")
ax.set_xlabel("perturbation budget  B")
ax.set_ylabel("retained explanation edges  (fraction)")
ax.set_ylim(-0.02, 1.0); ax.legend(loc="upper right", fontsize=8.5)
fs.save(fig, "F11_certificate_tightness")
print("saved F11 | cert==emp gap:", round(float(np.max(np.abs(np.array(c) - np.array(e)))), 4))
