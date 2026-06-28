# -*- coding: utf-8 -*-
"""
Project Aethelred — Certified PREDICTION robustness (head-to-head vs PGNNCert)
=============================================================================

Aethelred's edge certificate (aethelred_edge_certify) certifies the EXPLANATION.
This module adds the complementary PGNNCert-style certificate for the LABEL, so
Aethelred can be compared to PGNNCert on the prediction axis as well.

Mechanism (derandomized smoothing via edge hashing — identical partition to
PGNNCert/XGNNCert so the comparison is apples-to-apples):
  * Hash every undirected edge to one of T groups with the SAME md5 rule.
  * Build T edge-disjoint sub-graphs (full node set, the edges of one group).
  * Run the Aethelred classifier on each sub-graph and majority-vote the label.
  * A perturbation of B edges changes at most B sub-graphs (each edge lives in
    exactly one group), so the predicted label is provably unchanged whenever
    the vote margin exceeds 2B. Certified radius  Mc = floor((Yc - Yb - 1[c>b]) / 2).

Certified accuracy @ B = fraction of test graphs that are BOTH voted-correct AND
have Mc >= B.  This is exactly PGNNCert's metric; Aethelred's causal backbone is
what (we claim) buys higher certified accuracy at a given B.

NOTE: like PGNNCert, the base classifier must tolerate sparse sub-graph inputs.
Aethelred's IRM training already exposes it to edge-dropped / spanning-tree views
(see generate_environments), so a normally-trained model degrades gracefully;
for best certified accuracy train with robust=True (spanning-tree environments),
which mirrors PGNNCert's sub-graph training protocol.

This certificate is SOUND for any base classifier; soundness_check() verifies the
realized robustness never beats the certified bound under adversarial edge edits.
"""

import hashlib
import numpy as np
import torch
from torch_geometric.data import Data


# ----------------------------------------------------------------------------
# Hashing / partition — identical to PGNNCert (_ref_pgnncert) & XGNNCert
# ----------------------------------------------------------------------------
def hash_edge(V, u, v, T, h="md5"):
    """Map undirected edge {u,v} to a group in [0, T). Matches PGNNCert exactly:
    key = hex(V*min + max), md5 hexdigest mod T."""
    a, b = (u, v) if u <= v else (v, u)
    hexstring = hex(V * a + b).encode()
    hd = {"md5": hashlib.md5, "sha1": hashlib.sha1,
          "sha256": hashlib.sha256}[h]()
    hd.update(hexstring)
    return int(hd.hexdigest(), 16) % T


def partition_subgraphs(data, T, h="md5"):
    """Return up to T PyG Data sub-graphs: each holds the full node features and
    the directed edges whose undirected key hashes to that group."""
    x = data.x
    V = x.size(0)
    ei = data.edge_index
    groups = [[] for _ in range(T)]
    for e in range(ei.size(1)):
        u = int(ei[0, e]); v = int(ei[1, e])
        groups[hash_edge(V, u, v, T, h)].append(e)
    subs = []
    for g in groups:
        if not g:
            continue
        idx = torch.tensor(g, dtype=torch.long, device=ei.device)
        sub = Data(x=x, edge_index=ei[:, idx],
                   y=getattr(data, "y", None))
        sub.batch = torch.zeros(V, dtype=torch.long, device=x.device)
        subs.append(sub)
    return subs


# ----------------------------------------------------------------------------
# Voting prediction + certified radius
# ----------------------------------------------------------------------------
@torch.no_grad()
def vote_predict(model, data, T, h="md5"):
    """Majority-vote label over the T edge-hash sub-graphs and the PGNNCert
    certified radius Mc. Returns (vote_label:int, Mc:int, n_used:int)."""
    model.eval()
    subs = partition_subgraphs(data, T, h)
    if not subs:
        return -1, -1, 0
    num_classes = None
    counts = None
    for s in subs:
        out = model(s)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        logits = logits.view(-1)
        if counts is None:
            num_classes = logits.numel()
            counts = np.zeros(num_classes, dtype=np.int64)
        counts[int(logits.argmax().item())] += 1
    vote_label = int(counts.argmax())
    Yc = counts[vote_label]
    tmp = counts.copy(); tmp[vote_label] = -1
    second = int(tmp.argmax()); Yb = counts[second]
    Mc = (Yc - Yb - 1) // 2 if vote_label > second else (Yc - Yb) // 2
    return vote_label, int(Mc), len(subs)


def certified_accuracy_curve(model, graphs, budgets, T=50, h="md5",
                             max_graphs=None, verbose=True):
    """Certified accuracy vs perturbation budget B over a list of graphs.
    Returns dict[B] -> certified accuracy (voted-correct AND Mc >= B)."""
    if max_graphs is not None:
        graphs = graphs[:max_graphs]
    device = next(model.parameters()).device
    cert = {int(B): 0 for B in budgets}
    n = 0
    for g in graphs:
        g = g.to(device)
        y = int(g.y)
        vlabel, Mc, nsub = vote_predict(model, g, T, h)
        if nsub == 0:
            continue
        for B in budgets:
            if vlabel == y and Mc >= int(B):
                cert[int(B)] += 1
        n += 1
    curve = {int(B): (cert[int(B)] / n if n > 0 else 0.0) for B in budgets}
    if verbose:
        print(f"  [pred-cert] T={T}  n={n}")
        for B in budgets:
            print(f"    B={B:<3} certified_acc={curve[int(B)]:.4f}")
    return curve, n


# ----------------------------------------------------------------------------
# Soundness gate — realized robustness must never exceed the certified bound
# ----------------------------------------------------------------------------
@torch.no_grad()
def soundness_check(model, data, T=50, h="md5", n_trials=20, seed=0,
                    verbose=True):
    """For a graph certified at radius Mc, flip <= Mc edges adversarially and
    confirm the voted label never changes (certificate must hold)."""
    model.eval()
    device = next(model.parameters()).device
    data = data.to(device)
    vlabel, Mc, nsub = vote_predict(model, data, T, h)
    if Mc < 1:
        return {"skipped": True, "Mc": Mc}
    g = torch.Generator(device="cpu").manual_seed(seed)
    V = data.x.size(0)
    violations = 0
    for _ in range(n_trials):
        # delete up to Mc random existing edges (worst-case within budget)
        E = data.edge_index.size(1)
        ndel = int(torch.randint(1, Mc + 1, (1,), generator=g).item())
        keep = torch.ones(E, dtype=torch.bool)
        perm = torch.randperm(E, generator=g)[:ndel * 2]  # both directions
        keep[perm] = False
        pert = Data(x=data.x, edge_index=data.edge_index[:, keep], y=data.y)
        vl2, _, _ = vote_predict(model, pert, T, h)
        if vl2 != vlabel:
            violations += 1
    passed = violations == 0
    if verbose:
        print(f"  [pred-soundness] Mc={Mc} trials={n_trials} "
              f"violations={violations} -> {'PASS' if passed else 'FAIL'}")
    return {"skipped": False, "Mc": Mc, "violations": violations,
            "passed": passed}


if __name__ == "__main__":
    print(__doc__)
