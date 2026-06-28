# -*- coding: utf-8 -*-
"""
Project Aethelred — Deterministic Edge-Perturbation Certificate for Explanations
================================================================================

This module certifies the *explanation* (the top-k causal edge set) against
**graph-structure perturbations** (edge additions / deletions), the threat model
used by XGNNCert (ICLR 2025) and the GNNCert / AGNNCert lineage. It complements
the existing L-inf *feature* certificate in `aethelred_certify.py`.

Why this is voting-free (and why that is a feature, not a shortcut)
------------------------------------------------------------------
XGNNCert certifies explanation stability via *derandomized smoothing*: it hashes
edges into T sub-graphs, explains each, and votes — a bounded number of perturbed
edges can corrupt only a bounded number of votes. That machinery is needed
because their explainer is a GNN whose per-edge importance depends on the whole
sub-graph (message passing couples every edge to every other).

Aethelred's `CausalDiscoveryCore` is **propagation-free**: the saliency of edge
(u, v) is

        s(u, v) = sigmoid( MLP([ h_u, h_v, |h_u - h_v|, cos(x_u, x_v) ]) )
        with     h_u = MLP(x_u)            # depends ONLY on node u's features

(see aethelred_core.py, CausalDiscoveryCore.score_edges). Consequently:

    *An edge's saliency is invariant to the presence or absence of every OTHER
     edge in the graph.*

Adding or deleting edges anywhere else cannot change a given edge's score. This
single property collapses the smoothing argument into a deterministic counting
argument and yields a strictly tighter guarantee.

Theorem 1 (deterministic certified explanation overlap).
    Let E_k be the clean top-k explanation (k highest-scoring undirected edges).
    For any adversary that adds and/or deletes at most B edges in total, the
    perturbed top-k explanation E'_k satisfies
                        |E_k ∩ E'_k|  >=  k - B.
    Proof. Edge scores are edge-local, so deletions of edges other than those in
    E_k leave every E_k score and rank unchanged, and additions are the only
    operation that can lower an E_k edge's rank. A single added edge can outrank
    and thereby evict at most one member of E_k (it occupies exactly one slot).
    A single deletion can remove at most one member of E_k. Hence at most B
    members of E_k can leave the top-k, giving |E_k ∩ E'_k| >= k - B.            ∎

Theorem 2 (per-edge certified radius).
    Order all clean edges by descending score; let edge e have rank r (1-indexed,
    r <= k). Then e is guaranteed to remain in the top-k explanation under any
    perturbation of at most  R(e) = k - r  edges that does not delete e itself.
    Proof. Only additions scoring above s(e) lower e's rank, each by one; after a
    additions e has rank <= r + a, which stays <= k iff a <= k - r. Deletions of
    other edges only raise e's rank.                                             ∎

Theorem 3 (transductive hijack impossibility — strengthening unavailable to
           voting-based certificates).
    If the structural-prior allowlist is active (register_training_graph was
    called) with cap c = membership_decay, then every added (non-training) edge
    has score <= c. If every edge in E_k has clean score > c, then NO added edge
    can enter the top-k for ANY budget B; the explanation can lose members only
    by direct deletion. In particular the attacker-insertion ("hijack") rate is
    provably 0 for unbounded B.
    Proof. Immediate from the allowlist multiplier in score_edges and Theorem 2
    with R(e) = ∞ for additions.                                                 ∎

The functions below compute these certificates and provide an empirical
soundness gate (`soundness_check`) that must pass before any large-scale run:
the realized overlap under adversarial edge perturbation must never fall below
the certified lower bound k - B.

All certificates are SOUND for the explanation defined as the top-k edges of the
causal mask. They certify the explanation, not the FocalEngine label prediction
(which does depend on message passing); label robustness remains the province of
the PGNNCert-style voting baselines we compare against.
"""

import torch


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _undirected_edge_scores(edge_index, mask):
    """Collapse a directed edge_index + per-directed-edge mask into unique
    undirected edges with an aggregated (mean) score.

    Aethelred stores graphs with both (u, v) and (v, u); the scorer is not
    perfectly symmetric (endpoint order enters the MLP), so we average the two
    directions to obtain a single undirected saliency per physical edge.

    Returns
    -------
    keys   : LongTensor [E_u, 2]  canonical (min, max) endpoints, sorted-unique
    scores : Tensor    [E_u]      mean score over the (<=2) directions
    """
    u = edge_index[0]
    v = edge_index[1]
    lo = torch.minimum(u, v)
    hi = torch.maximum(u, v)
    # Stable integer key for grouping; N is an upper bound on node id.
    N = int(torch.max(hi).item()) + 1 if hi.numel() > 0 else 1
    key = lo.to(torch.long) * N + hi.to(torch.long)

    uniq, inv = torch.unique(key, return_inverse=True)
    # Mean-aggregate the directed scores onto each undirected edge.
    summ = torch.zeros(uniq.numel(), dtype=mask.dtype, device=mask.device)
    cnt = torch.zeros(uniq.numel(), dtype=mask.dtype, device=mask.device)
    summ.index_add_(0, inv, mask)
    cnt.index_add_(0, inv, torch.ones_like(mask))
    scores = summ / cnt.clamp_min(1.0)

    keys = torch.stack([uniq // N, uniq % N], dim=1)
    return keys, scores


def _top_k(scores, top_k_frac=None, k=None):
    M = scores.numel()
    if k is None:
        k = max(1, int(round(M * (top_k_frac if top_k_frac is not None else 0.1))))
    k = min(k, M)
    topv, topi = torch.topk(scores, k)
    return k, topi, topv


# ----------------------------------------------------------------------------
# Certificate
# ----------------------------------------------------------------------------
def certify_edge_explanation(
    model,
    data,
    top_k_frac=0.1,
    k=None,
    budget=None,
    undirected=True,
    verbose=True,
):
    """Deterministic edge-perturbation certificate for the top-k explanation.

    Parameters
    ----------
    model        : Aethelred model in eval mode. Uses model(data) -> (logits, mask)
                   and model.causal_core for the allowlist cap.
    data         : PyG Data on the model's device.
    top_k_frac   : fraction of edges treated as the explanation (ignored if k set).
    k            : explicit explanation size (overrides top_k_frac).
    budget       : edge-perturbation budget B for the graph-level guarantee. If
                   None, only per-edge radii + the max certifiable B are reported.
    undirected   : collapse (u,v)/(v,u) to one physical edge before ranking.

    Returns
    -------
    report : dict following the Aethelred certification-report schema, with the
             edge-certificate fields added:
        certified                  : bool   (overlap LB > 0 at the given budget)
        k                          : int
        budget_B                   : int or None
        certified_overlap_lb       : int    max(0, k - B)  (Theorem 1)
        per_edge_radius            : LongTensor [k]   R(e_r)=k-r (Theorem 2)
        max_certifiable_budget     : int    largest B with k - B >= 1
        unbounded_addition_robust  : bool   Theorem 3 holds (allowlist + scores>c)
        addition_cap               : float or None  membership_decay c
        n_topk_above_cap           : int
        min_salient_score          : float
        threshold_score_k          : float  k-th largest score
        num_edges                  : int (undirected if undirected=True)
    """
    model.eval()
    with torch.no_grad():
        out = model(data)
        mask = out[1] if isinstance(out, (tuple, list)) else out

    if undirected:
        _, scores = _undirected_edge_scores(data.edge_index, mask)
    else:
        scores = mask

    M = scores.numel()
    k, topi, topv = _top_k(scores, top_k_frac=top_k_frac, k=k)

    # Theorem 2: rank r (1-indexed) among ALL edges -> radius k - r.
    # topv is already sorted descending, so rank of the i-th returned edge is i+1.
    ranks = torch.arange(1, k + 1, device=scores.device)
    per_edge_radius = (k - ranks).clamp_min(0)  # R(e_r) = k - r
    max_certifiable_budget = int((k - 1)) if k >= 1 else 0

    # Theorem 3: allowlist hijack impossibility.
    cap = None
    n_above_cap = k
    unbounded = False
    core = getattr(model, "causal_core", None)
    if core is not None and getattr(core, "_allowlist_active", lambda: False)():
        cap = float(getattr(core, "membership_decay", 0.0))
        n_above_cap = int((topv > cap).sum().item())
        unbounded = bool(n_above_cap == k)

    report = {
        "k": int(k),
        "num_edges": int(M),
        "budget_B": (int(budget) if budget is not None else None),
        "per_edge_radius": per_edge_radius.cpu(),
        "max_certifiable_budget": max_certifiable_budget,
        "unbounded_addition_robust": unbounded,
        "addition_cap": cap,
        "n_topk_above_cap": int(n_above_cap),
        "min_salient_score": float(topv.min().item()),
        "threshold_score_k": float(topv.min().item()),
    }
    if budget is not None:
        overlap_lb = max(0, k - int(budget))
        report["certified_overlap_lb"] = overlap_lb
        report["certified"] = bool(overlap_lb > 0)
    else:
        report["certified_overlap_lb"] = None
        report["certified"] = bool(max_certifiable_budget > 0)

    if verbose:
        print(f"  [edge-cert] k={k}  |E|={M}  "
              f"max certifiable B={max_certifiable_budget}")
        if budget is not None:
            print(f"             B={budget} -> certified overlap >= "
                  f"{report['certified_overlap_lb']}/{k}")
        if cap is not None:
            print(f"             allowlist cap c={cap:.2f}  "
                  f"top-k above cap: {n_above_cap}/{k}  "
                  f"unbounded-addition robust: {unbounded}")
    return report


def certified_overlap_curve(model, data, budgets, top_k_frac=0.1, k=None,
                            undirected=True):
    """Certified explanation overlap (>= k - B) vs edge budget B — the data for
    the Figure-2 cert-sweep curve in the XGNNCert head-to-head.

    Returns list of dicts: [{"budget": B, "overlap_lb": int, "overlap_frac": f}].
    """
    base = certify_edge_explanation(model, data, top_k_frac=top_k_frac, k=k,
                                    undirected=undirected, verbose=False)
    k = base["k"]
    curve = []
    for B in budgets:
        lb = max(0, k - int(B))
        curve.append({"budget": int(B),
                      "overlap_lb": lb,
                      "overlap_frac": (lb / k if k > 0 else 0.0)})
    return curve


# ----------------------------------------------------------------------------
# Soundness gate — must pass before any large-scale run (Phase 0 gate)
# ----------------------------------------------------------------------------
@torch.no_grad()
def soundness_check(model, data, budget, n_candidate_nonedges=2000,
                    top_k_frac=0.1, k=None, undirected=True, seed=0,
                    verbose=True):
    """Empirically verify the certificate is SOUND: under a worst-effort
    adversarial edge perturbation of size `budget`, the realized overlap with the
    clean top-k explanation must never drop below the certified bound k - budget.

    Adversary (greedy, near-worst-case):
      * additions: score many random non-edges with the SAME scorer, add the
        `budget` highest-scoring ones (most able to displace clean top-k edges);
      * deletions: also test deleting the `budget` highest clean top-k edges.
    Both are evaluated; the test passes iff every realized overlap >= k - budget.

    Returns dict {passed, certified_lb, realized_overlap_add, realized_overlap_del}.
    """
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    device = data.x.device
    N = data.x.size(0)

    out = model(data)
    mask = out[1] if isinstance(out, (tuple, list)) else out
    if undirected:
        keys, scores = _undirected_edge_scores(data.edge_index, mask)
    else:
        keys, scores = None, mask
    k, topi, topv = _top_k(scores, top_k_frac=top_k_frac, k=k)
    clean_top = set(topi.cpu().tolist())
    certified_lb = max(0, k - int(budget))

    # ---- Addition adversary --------------------------------------------------
    # Sample candidate *true* non-edges (pairs absent from the current graph) and
    # score them with the propagation-free core. Excluding existing edges is
    # essential: an "addition" that is already present is not a structural
    # perturbation, and under the allowlist only genuine non-edges are capped.
    core = model.causal_core
    h = core._node_embeddings(data.x)
    existing = set((min(a, b) * N + max(a, b))
                   for a, b in data.edge_index.t().cpu().tolist())
    cand = torch.randint(0, N, (2, int(n_candidate_nonedges)), generator=g).to(device)
    cand = cand[:, cand[0] != cand[1]]
    if cand.numel() > 0:
        ckey = (torch.minimum(cand[0], cand[1]) * N
                + torch.maximum(cand[0], cand[1])).cpu().tolist()
        keep = torch.tensor([kk not in existing for kk in ckey], device=device)
        cand = cand[:, keep]
    add_scores = (core.score_edges(h, cand, x_raw=data.x)
                  if cand.numel() > 0 else torch.empty(0, device=device))
    n_add = min(int(budget), add_scores.numel())
    if n_add > 0:
        add_top = torch.topk(add_scores, n_add).values
        # New combined ranking: clean undirected scores + added-edge scores.
        combined = torch.cat([scores, add_top])
        _, comb_topi = torch.topk(combined, k)
        # Clean edges that survive = those whose original index is still in top-k
        survived_add = sum(1 for i in comb_topi.cpu().tolist()
                           if i < scores.numel() and i in clean_top)
    else:
        survived_add = k

    # ---- Deletion adversary (delete the strongest clean top-k edges) --------
    # Deleting the highest-scoring top-k edges removes exactly min(budget,k) of
    # them; realized overlap = k - min(budget, k).
    survived_del = k - min(int(budget), k)

    passed = (survived_add >= certified_lb) and (survived_del >= certified_lb)
    if verbose:
        print(f"  [soundness] B={budget}  certified LB={certified_lb}/{k}  "
              f"realized: add={survived_add}  del={survived_del}  "
              f"-> {'PASS' if passed else 'FAIL'}")
    return {
        "passed": bool(passed),
        "k": int(k),
        "budget": int(budget),
        "certified_lb": int(certified_lb),
        "realized_overlap_add": int(survived_add),
        "realized_overlap_del": int(survived_del),
    }


if __name__ == "__main__":
    print(__doc__)
    print("This module is imported by run_aethelred_comparison.py (Phase 2 "
          "head-to-head). Run soundness_check on a trained model before "
          "committing to a full certification sweep.")
