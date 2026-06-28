# -*- coding: utf-8 -*-
"""F10 — edge-locality: a target edge's importance vs. the number of OTHER edges
perturbed. Aethelred's propagation-free scorer is EXACTLY invariant (this is why
the certificate is deterministic); a propagation-based (GCN) edge score drifts."""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aethelred_core import Aethelred
import aethelred_figstyle as fs
import matplotlib.pyplot as plt

fs.apply()
torch.manual_seed(0); np.random.seed(0)
N, Fd, C = 40, 16, 2
x = torch.randn(N, Fd)
src = torch.randint(0, N, (90,)); dst = torch.randint(0, N, (90,)); m = src != dst
ei = torch.stack([torch.cat([src[m], dst[m]]), torch.cat([dst[m], src[m]])])
model = Aethelred(Fd, C, task="graph", conv_type="GCN").eval()
cc = model.causal_core; fe = model.focal_engine

# pick a target existing undirected edge
u, v = int(ei[0, 0]), int(ei[1, 0])
existing = set((min(int(a), int(b)), max(int(a), int(b))) for a, b in ei.t().tolist())
cands = [(a, b) for a in range(N) for b in range(a + 1, N) if (a, b) not in existing]
np.random.shuffle(cands)

ns = list(range(0, 26, 2))
aeth, prop = [], []
h = cc._node_embeddings(x)                       # MLP(x) — graph-independent
tgt = torch.tensor([[u], [v]])
with torch.no_grad():
    s_aeth0 = float(cc.score_edges(h, tgt, x_raw=x))
for n in ns:
    add = cands[:n]
    if add:
        ex = torch.tensor([[a for a, b in add] + [b for a, b in add],
                           [b for a, b in add] + [a for a, b in add]])
        ei_p = torch.cat([ei, ex], dim=1)
    else:
        ei_p = ei
    with torch.no_grad():
        # Aethelred: causal score of the target edge (propagation-free)
        s_a = float(cc.score_edges(h, tgt, x_raw=x))
        # propagation-based: GCN node embeddings -> target-edge cosine
        emb = fe.get_node_embeddings(x, ei_p, torch.ones(ei_p.size(1)))
        s_p = float(torch.cosine_similarity(emb[u].unsqueeze(0), emb[v].unsqueeze(0)))
    aeth.append(s_a); prop.append(s_p)

# normalize to the n=0 value -> relative importance change
aeth = np.array(aeth) / (aeth[0] if aeth[0] else 1)
prop = np.array(prop) / (prop[0] if prop[0] else 1)

fig, ax = fs.new_fig(5.4, 4.0)
ax.plot(ns, prop, color=fs.C["GNNExplainer-spm"], linestyle=":", marker="v",
        markeredgecolor="black", markeredgewidth=0.7, linewidth=2.2,
        label="propagation-based (GCN)")
ax.plot(ns, aeth, color=fs.C["Aethelred"], linestyle="-", marker="o",
        markeredgecolor="black", markeredgewidth=0.7, linewidth=2.9,
        label="Aethelred (propagation-free)")
ax.axhline(1.0, color="#BBBBBB", lw=0.8, zorder=0)
ax.set_xlabel("number of OTHER edges perturbed")
ax.set_ylabel("target-edge importance  (relative to clean)")
ax.legend(loc="upper left")
fs.save(fig, "F10_edge_locality")
print("saved F10", "aeth range", round(aeth.min(), 4), round(aeth.max(), 4),
      "| prop range", round(prop.min(), 3), round(prop.max(), 3))
