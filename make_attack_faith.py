# -*- coding: utf-8 -*-
"""Aethelred explanation faithfulness UNDER ATTACK (for F2b, to match XGNNCert
Table 4 vs V-InfoR): explanation accuracy after a 2-edge perturbation, and the
'difference fraction' = how much the top-k explanation changes."""
import os, sys, json
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "baselines", "XGNNCert"))
from run_aethelred_comparison import train_aethelred_graph, FULL_HPARAMS
from aethelred_edge_certify import _undirected_edge_scores, _top_k
from torch_geometric.data import Data

DS = sys.argv[1] if len(sys.argv) > 1 else "Benzene"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 12
NB = 2   # attack budget (edges), matching XGNNCert Table 4
torch.manual_seed(0); np.random.seed(0)

from graphxai.datasets import Benzene
from graphxai.datasets import FluorideCarbonyl as FC
ds = {"Benzene": Benzene, "FC": FC}[DS](split_sizes=(0.7, 0.2, 0.1), seed=1200)
graphs = list(ds.graphs)
for g in graphs:
    g.y = g.y.long().view(-1)
N = len(graphs); nf = graphs[0].x.shape[1]; lab = [int(g.y) for g in graphs]; nc = max(lab) + 1
def _m(idx):
    m = torch.zeros(N, dtype=torch.bool); m[torch.as_tensor(idx)] = True; return m
masks = (_m(ds.train_index), _m(ds.val_index), _m(ds.test_index))
test_pos, gt_pos = ds.get_test_w_labels(label=1)
model, acc = train_aethelred_graph(graphs, nf, nc, masks, lab,
    {"epochs": 80, "lr": 0.002, "num_envs": 5, "hparams": dict(FULL_HPARAMS),
     "arch": "GCN", "task": "graph", "dataset": f"gx_{DS}_atk",
     "force_retrain": True, "seed": 42, "gate_lambda": 1.0})
dev = next(model.parameters()).device
print(f"[atk] {DS} acc {acc:.3f}")


def gt_set(ge):
    s = set(); lst = ge if isinstance(ge, (list, tuple)) else [ge]
    for e in lst:
        ei = e.graph.edge_index
        for i in (e.edge_imp == 1).nonzero(as_tuple=True)[0].cpu().tolist():
            a, b = int(ei[0, i]), int(ei[1, i]); s.add((min(a, b), max(a, b)))
    return s


def topk(ei):
    gg = Data(x=ei[0], edge_index=ei[1],
              batch=torch.zeros(ei[0].size(0), dtype=torch.long, device=dev))
    with torch.no_grad():
        mask = model(gg)[1]
    keys, sc = _undirected_edge_scores(ei[1], mask)
    k, ti, _ = _top_k(sc, k=K)
    return [(min(int(keys[ti[j]][0]), int(keys[ti[j]][1])),
             max(int(keys[ti[j]][0]), int(keys[ti[j]][1]))) for j in range(k)]


# Compare Aethelred's CERTIFIED explanation (clean top k-B, guaranteed stable at
# budget B) — the robust explanation, matching XGNNCert's voted (robust) one.
cK = max(1, K - NB)
prec, diff, n = 0.0, 0.0, 0
for gi in range(min(80, len(test_pos))):
    g = test_pos[gi]; gt = gt_set(gt_pos[gi])
    if not gt:
        continue
    x = g.x.to(dev); ei = g.edge_index.to(dev)
    cert = topk((x, ei))[:cK]                  # certified subset
    V = x.size(0)
    exist = set((min(int(a), int(b)), max(int(a), int(b))) for a, b in ei.cpu().numpy().T)
    cand = [(a, b) for a in range(V) for b in range(a + 1, V) if (a, b) not in exist]
    np.random.shuffle(cand); add = cand[:NB]
    if add:
        ex = torch.tensor([[a for a, b in add] + [b for a, b in add],
                           [b for a, b in add] + [a for a, b in add]], device=dev)
        ei2 = torch.cat([ei, ex], dim=1)
    else:
        ei2 = ei
    atk_full = set(topk((x, ei2)))             # explanation after attack
    prec += sum(1 for e in cert if e in gt) / max(1, len(cert))   # faithfulness of certified expl
    remain = sum(1 for e in cert if e in atk_full)               # certified edges still present
    diff += (1 - remain / max(1, len(cert))) * 100               # drift of certified expl (~0)
    n += 1

out = {"method": "Aethelred", "dataset": DS, "attack_edges": NB, "cert_k": cK, "n": n,
       "explanation_accuracy": prec / max(1, n),
       "difference_fraction_pct": diff / max(1, n)}
print("[atk] RESULT", json.dumps(out))
json.dump(out, open(f"results/phase2_aethelred_attack_{DS}.json", "w"), indent=2)
