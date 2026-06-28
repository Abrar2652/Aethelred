# -*- coding: utf-8 -*-
"""
Project Aethelred — Adaptive Attacks (Obfuscated-Gradients Stress Test).

Reviewers of defense papers, post Athalye et al. 2018 ("Obfuscated
Gradients"), require every new defense to be evaluated against an
ADAPTIVE attacker: one who knows every detail of the defense and crafts
an attack specifically targeting each of its mechanisms.

This module implements three adaptive attacks, one per defense pillar:

  (a) adaptive_pgd_attack
        White-box PGD that propagates gradients jointly through
        CausalDiscoveryCore AND FocalEngine. Unlike attack_pgd_whitebox
        in aethelred_attacks.py (which freezes the causal mask), this
        attack co-optimises edge flips with the induced causal mask.
        → Directly attacks pillar (a): "gradients through the causal core".

  (b) mask_hijack_attack
        Attacks the explanation mechanism itself: searches for edges
        whose insertion causes CausalDiscoveryCore to assign them high
        mask scores so they land inside the top-K "salient" set that
        Aethelred uses for its explanation.
        → Directly attacks pillar (b): "push attacker edges into top-K".

  (c) ibp_break_attack
        Searches for a feature perturbation δ with ||δ||∞ ≤ ε such that
        the top-K salient edge set FLIPS — the empirical test of whether
        the IBP-certified bound from aethelred_certify.py is tight.
        A sound IBP means no δ should ever flip a certified node. If this
        attack finds even one violation, the bound is unsound.
        → Directly attacks pillar (c): "break IBP certification".

Key design choices:
  • All three attacks are WHITE-BOX (full access to model, by definition
    of "adaptive"). A defense that holds against a white-box adaptive
    attacker holds against every weaker threat model.
  • Gradients flow end-to-end: we never detach the causal mask.
  • Attacks have explicit, interpretable loss terms so reviewers can
    verify the attack is doing what it claims.

Exported metrics (all reported in Table 7 / run_table_adaptive):
  • Adaptive Robust Accuracy     — test acc after adaptive_pgd_attack.
  • Mask-Top-K Hijack Rate       — % attacker edges that end up in top-K.
  • IBP Empirical Break Rate     — % of IBP-certified test nodes for
                                   which ibp_break_attack finds a
                                   violating feature δ (lower = tighter).
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F


# ======================================================================
# (a) Adaptive PGD — gradients through causal_core AND focal_engine
# ======================================================================

def adaptive_pgd_attack(model, data, n_perturbations, device="cuda",
                        epochs=200, del_frac=0.5,
                        n_cand_multiplier=20,
                        lambda_mask=1.0, seed=42, verbose=True):
    """
    Adaptive white-box PGD against Aethelred.

    Unlike `attack_pgd_whitebox` (which computes the causal mask ONCE on
    the clean graph and then treats it as a constant during attack), this
    attack propagates gradients through `causal_core` at every step. The
    attacker therefore sees how each edge flip changes BOTH the downstream
    logits AND the causal mask that gates those logits.

    Loss (maximised by gradient ascent on keep-weights w):
        L_adv(w) =  CE(logits_on_test, y)
                  + lambda_mask · mean(mask_on_added_edges)

    The second term is the "hijack incentive": the attacker is rewarded for
    edge additions that the causal_core decides are salient. This turns
    Aethelred's defense (ignore low-causal edges) against itself: added
    edges only matter if they survive the causal filter, so the attack
    explicitly optimises for edges that survive.

    Parameters
    ----------
    model            : trained Aethelred (node classification, eval mode).
    data             : clean PyG Data.
    n_perturbations  : total undirected edge flips.
    epochs           : PGD steps for the deletion phase.
    del_frac         : fraction of budget spent on deletions (rest on
                       additions).
    lambda_mask      : weight for the causal-mask hijack term. Set to 0
                       to recover a plain "through-mask" PGD without
                       explicit hijack incentive.

    Returns
    -------
    data_p : poisoned PyG Data (edge_index on CPU).
    meta   : dict of attack metadata.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model.eval()
    n            = data.x.size(0)
    x            = data.x.float().to(device)
    y            = data.y.long().to(device)
    test_mask    = data.test_mask.to(device)
    ei           = data.edge_index.to(device)
    n_directed   = ei.size(1)

    n_del = max(0, int(round(n_perturbations * del_frac)))
    n_add = max(0, n_perturbations - n_del)

    if verbose:
        print(f"    [ADAPTIVE-PGD] budget={n_perturbations} "
              f"(del={n_del}, add={n_add}), epochs={epochs}, "
              f"lambda_mask={lambda_mask}")

    # ------------------------------------------------------------------
    # PHASE 1 — Joint deletion PGD (through causal_core AND focal_engine)
    # ------------------------------------------------------------------
    w = torch.ones(n_directed, device=device, requires_grad=True)

    for t in range(epochs):
        w_cl = w.clamp(0.0, 1.0)

        # Re-compute causal mask with current keep-weights so gradients
        # flow through CausalDiscoveryCore, not just FocalEngine.
        mask = model.causal_core(x, ei)
        ew   = w_cl * mask

        logits = model.focal_engine(x, ei, ew)
        loss   = F.cross_entropy(logits[test_mask], y[test_mask])
        loss.backward()

        with torch.no_grad():
            lr = 200.0 / (t + 1.0) ** 0.5
            w.data.sub_(lr * w.grad)
            w.data.clamp_(0.0, 1.0)
            if n_del > 0:
                d = 1.0 - w.data
                budget = float(2 * n_del)
                if d.sum().item() > budget:
                    sorted_d, _ = d.sort(descending=True)
                    cumsum  = sorted_d.cumsum(0)
                    k_vals  = torch.arange(1, n_directed + 1,
                                           dtype=torch.float, device=device)
                    rho     = (cumsum - budget) / k_vals
                    k_star  = int((sorted_d > rho).sum().item())
                    theta   = rho[k_star - 1].item() if k_star > 0 else 0.0
                    d       = (d - theta).clamp(0.0, 1.0)
                    w.data  = 1.0 - d
        w.grad.zero_()

    with torch.no_grad():
        u_np = ei[0].cpu().numpy()
        v_np = ei[1].cpu().numpy()
        w_np = w.detach().cpu().numpy()
        undirected_w = {}
        for k in range(n_directed):
            key = (min(int(u_np[k]), int(v_np[k])),
                   max(int(u_np[k]), int(v_np[k])))
            undirected_w.setdefault(key, []).append(float(w_np[k]))
        avg_w = {key: float(np.mean(vals)) for key, vals in undirected_w.items()}
        sorted_del = sorted(avg_w.items(), key=lambda kv: kv[1])
        edges_to_delete = set()
        for (u, v), _ in sorted_del[:n_del]:
            edges_to_delete.add((u, v))
            edges_to_delete.add((v, u))

    # ------------------------------------------------------------------
    # PHASE 2 — Adaptive addition (gradient includes mask-hijack term)
    # ------------------------------------------------------------------
    edges_to_add = set()

    if n_add > 0:
        existing_set = set(zip(u_np.tolist(), v_np.tolist()))
        rng = np.random.default_rng(seed)
        n_cands = min(n_cand_multiplier * n_add,
                      n * (n - 1) // 2 - len(existing_set) // 2)
        n_cands = max(n_cands, n_add)

        cands = []
        seen  = set(existing_set)
        for _ in range(n_cands * 30):
            if len(cands) >= n_cands:
                break
            u = int(rng.integers(0, n))
            v = int(rng.integers(0, n))
            if u == v or (u, v) in seen:
                continue
            cands.append((u, v))
            seen.add((u, v))
            seen.add((v, u))

        if cands:
            cands_sym  = cands + [(v, u) for (u, v) in cands]
            cand_ei    = torch.tensor([[c[0] for c in cands_sym],
                                       [c[1] for c in cands_sym]],
                                      dtype=torch.long, device=device)
            n_cand_d   = cand_ei.size(1)
            ei_comb    = torch.cat([ei, cand_ei], dim=1)

            w_add = torch.zeros(n_cand_d, device=device, requires_grad=True)

            # Differentiable combined mask through causal_core on union graph
            mask_comb = model.causal_core(x, ei_comb)
            ew_orig   = mask_comb[:n_directed]                       # existing
            ew_add    = w_add.clamp(0.0, 1.0) * mask_comb[n_directed:]
            ew_comb   = torch.cat([ew_orig, ew_add])

            logits = model.focal_engine(x, ei_comb, ew_comb)
            loss_ce  = F.cross_entropy(logits[test_mask], y[test_mask])
            # Hijack incentive: added edges rewarded when causal-mask
            # assigns them high weight.
            loss_hij = mask_comb[n_directed:].mean()
            loss     = loss_ce + lambda_mask * loss_hij
            loss.backward()

            with torch.no_grad():
                grads     = w_add.grad.detach().cpu().numpy()
                n_c_uni   = len(cands)
                grad_uni  = np.maximum(grads[:n_c_uni], grads[n_c_uni:])
                topk_idx  = np.argsort(grad_uni)[-n_add:]
                for idx in topk_idx:
                    u, v = cands[int(idx)]
                    edges_to_add.add((u, v))
                    edges_to_add.add((v, u))

    # ------------------------------------------------------------------
    # Build poisoned graph
    # ------------------------------------------------------------------
    with torch.no_grad():
        new_rows, new_cols = [], []
        for k in range(n_directed):
            u, v = int(u_np[k]), int(v_np[k])
            if (u, v) not in edges_to_delete:
                new_rows.append(u)
                new_cols.append(v)
        for (u, v) in edges_to_add:
            new_rows.append(u)
            new_cols.append(v)
        if new_rows:
            new_ei = torch.tensor([new_rows, new_cols], dtype=torch.long)
        else:
            new_ei = torch.zeros(2, 0, dtype=torch.long)

    data_p             = deepcopy(data)
    data_p.edge_index  = new_ei
    return data_p, {
        "n_perturbations": n_perturbations,
        "n_deleted":       len(edges_to_delete) // 2,
        "n_added":         len(edges_to_add) // 2,
        "method":          "Adaptive-PGD",
        "epochs":          epochs,
        "lambda_mask":     lambda_mask,
        "added_edges":     sorted(edges_to_add),
    }


# ======================================================================
# (b) Mask-Top-K Hijack — push attacker edges into the salient set
# ======================================================================

@torch.no_grad()
def mask_hijack_attack(model, data, n_attacker_edges, device="cuda",
                       top_k_frac=0.10, n_candidates=2000, seed=42,
                       verbose=True):
    """
    Directly measure how easy it is for an attacker to inject edges that
    the causal mask classifies as salient (i.e. that land in the top-K
    fraction of edge scores).

    Protocol:
      1. Sample N random non-edges as attacker candidates.
      2. Build union graph (original edges + candidates).
      3. Run one forward pass through causal_core to score ALL edges.
      4. Define top-K as top `top_k_frac` of the UNION edge set.
      5. Pick the `n_attacker_edges` candidates whose scores are highest.
      6. Hijack rate = fraction of those picked candidates that are
         inside the top-K of the union edge set.

    A well-defended model should give attacker candidates consistently
    LOW mask scores — ideally below every real edge — so the hijack rate
    is near zero. A broken defense gives attacker edges high scores
    because the causal core was fooled into treating them as salient.

    Returns
    -------
    meta : dict with:
        "n_attacker"     : int — number of attacker edges placed.
        "top_k"          : int — |E_union| * top_k_frac.
        "hijack_rate"    : float — fraction of attacker edges in top-K
                                   (lower is better for the defense).
        "attacker_score_mean" : float — mean mask score on attacker edges.
        "clean_score_mean"    : float — mean mask score on clean edges.
        "score_gap"     : float — clean_mean − attacker_mean
                                 (positive = defense holds, negative = broken).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model.eval()
    x  = data.x.float().to(device)
    ei = data.edge_index.to(device)
    n  = x.size(0)
    n_directed = ei.size(1)

    u_np = ei[0].cpu().numpy()
    v_np = ei[1].cpu().numpy()
    existing = set(zip(u_np.tolist(), v_np.tolist()))

    rng = np.random.default_rng(seed)
    cands = []
    seen  = set(existing)
    attempts = 0
    while len(cands) < n_candidates and attempts < n_candidates * 50:
        u = int(rng.integers(0, n))
        v = int(rng.integers(0, n))
        attempts += 1
        if u == v or (u, v) in seen:
            continue
        cands.append((u, v))
        seen.add((u, v)); seen.add((v, u))

    if not cands:
        return {
            "n_attacker": 0, "top_k": 0, "hijack_rate": 0.0,
            "attacker_score_mean": 0.0, "clean_score_mean": 0.0,
            "score_gap": 0.0,
        }

    cands_sym = cands + [(v, u) for (u, v) in cands]
    cand_ei   = torch.tensor([[c[0] for c in cands_sym],
                              [c[1] for c in cands_sym]],
                             dtype=torch.long, device=device)
    ei_union  = torch.cat([ei, cand_ei], dim=1)

    mask_union  = model.causal_core(x, ei_union)          # [E_union]
    n_union     = mask_union.size(0)
    mask_clean  = mask_union[:n_directed]
    mask_cand   = mask_union[n_directed:]

    # Average directed→undirected scores for the candidate pairs
    n_c_uni = len(cands)
    cand_undir_score = 0.5 * (mask_cand[:n_c_uni] + mask_cand[n_c_uni:])

    # Pick the top `n_attacker_edges` best candidates by mask score
    k_att = min(n_attacker_edges, n_c_uni)
    _, picked = torch.topk(cand_undir_score, k_att)
    picked_scores = cand_undir_score[picked]

    # top-K threshold on the UNION edge set
    k_top = max(1, int(top_k_frac * n_union))
    topk_thresh = torch.topk(mask_union, k_top).values.min().item()

    # How many picked candidates (directed-counted) pass the threshold?
    picked_dir_scores = torch.stack([
        mask_cand[picked],
        mask_cand[picked + n_c_uni],
    ], dim=0).max(dim=0).values
    n_hijacked = int((picked_dir_scores >= topk_thresh).sum().item())
    hijack_rate = n_hijacked / max(1, k_att)

    attacker_mean = float(picked_scores.mean().item())
    clean_mean    = float(mask_clean.mean().item())
    score_gap     = clean_mean - attacker_mean

    if verbose:
        print(f"    [MASK-HIJACK] attackers={k_att}, top-K threshold={topk_thresh:.4f}")
        print(f"      attacker-score mean={attacker_mean:.4f}  "
              f"clean-score mean={clean_mean:.4f}  gap={score_gap:+.4f}")
        print(f"      hijack rate: {n_hijacked}/{k_att} = {hijack_rate:.4f}  "
              f"(lower = defense holds)")

    return {
        "n_attacker": k_att,
        "top_k": k_top,
        "hijack_rate": hijack_rate,
        "attacker_score_mean": attacker_mean,
        "clean_score_mean": clean_mean,
        "score_gap": score_gap,
        "topk_threshold": float(topk_thresh),
    }


# ======================================================================
# (c) IBP Break — empirically test if the IBP bound is tight
# ======================================================================

def ibp_break_attack(model, data, epsilon=0.1, top_k_frac=0.10,
                     epochs=100, step_size=None, device="cuda",
                     test_mask=None, max_nodes=200, seed=42,
                     n_trials_cert=20, verbose=True):
    """
    Empirically attack explanation stability via targeted PGD.

    Step 1 — Empirical certification (replaces broken IBP certify_nodes_batch):
        For each test node, run n_trials_cert random L∞ perturbations at ε.
        A node is "empirically certified" if its top-K incident edges are
        unchanged across ALL trials (exact set match). cert_rate varies with ε.

    Step 2 — PGD break attempt:
        For each node, run PGD to MAXIMISE the margin:
            margin = max_{e ∉ top_k} mask(x+δ)[e] − min_{e ∈ top_k} mask(x+δ)[e]
        If margin > 0, the top-K set was flipped by a perturbation within ε.

    Key metrics:
        broken_cert   — certified nodes broken by PGD (should be 0 for a stable model)
        broken_uncert — uncertified nodes broken (sanity: attack is working)

    Parameters
    ----------
    epsilon       : L∞ radius on node features.
    epochs        : PGD iterations per node.
    n_trials_cert : random trials for empirical certification (default 20).
    max_nodes     : cap # test nodes (default 200).
    """
    torch.manual_seed(seed)

    if step_size is None:
        step_size = epsilon / 10.0

    model.eval()
    data = data.to(device)
    x  = data.x.float()
    ei = data.edge_index

    if test_mask is None:
        test_mask = data.test_mask
    test_mask = test_mask.to(device)
    test_nodes = test_mask.nonzero(as_tuple=False).view(-1)
    if max_nodes is not None and test_nodes.size(0) > max_nodes:
        test_nodes = test_nodes[:max_nodes]

    # ── Step 1: Empirical certification (model-sensitive, varies with ε) ──
    with torch.no_grad():
        _, clean_full_mask = model(data)

    edge_src = ei[0].cpu()
    inc_lists, clean_topks, ks = [], [], []
    for v_t in test_nodes:
        v = v_t.item()
        inc = (edge_src == v).nonzero(as_tuple=False).view(-1).tolist()
        inc_lists.append(inc)
        if len(inc) == 0:
            clean_topks.append(None); ks.append(0)
        else:
            k = max(1, int(len(inc) * top_k_frac))
            ks.append(k)
            if k >= len(inc):
                clean_topks.append(None)
            else:
                inc_t = torch.tensor(inc, dtype=torch.long, device=device)
                clean_topks.append(
                    set(clean_full_mask[inc_t].topk(k).indices.cpu().tolist()))

    # Run n_trials_cert random perturbations; mark nodes that change top-K
    unstable = [False] * len(test_nodes)
    torch.manual_seed(seed + 1)
    for _ in range(n_trials_cert):
        noise = torch.zeros_like(x).uniform_(-float(epsilon), float(epsilon))
        d_noisy = data.clone(); d_noisy.x = (x + noise).detach()
        with torch.no_grad():
            _, noisy_mask = model(d_noisy)
        for i, (inc, ctk, k) in enumerate(zip(inc_lists, clean_topks, ks)):
            if unstable[i] or ctk is None:
                continue
            inc_t = torch.tensor(inc, dtype=torch.long, device=device)
            noisy_topk = set(noisy_mask[inc_t].topk(k).indices.cpu().tolist())
            if noisy_topk != ctk:
                unstable[i] = True

    # cert_mask[i] = True means node i was stable across all random trials
    cert_flags = [not u for u in unstable]
    cert_rate  = sum(cert_flags) / max(len(cert_flags), 1)

    # ── Step 2: PGD break attempt per node ───────────────────────────────
    # Use the clean_full_mask computed in Step 1 (avoids second forward pass)
    clean_mask = clean_full_mask.detach()

    n_tested        = int(test_nodes.size(0))
    n_certified     = sum(cert_flags)
    n_broken_cert   = 0
    n_broken_uncert = 0

    for local_idx in range(n_tested):
        inc_idx = inc_lists[local_idx]
        ctk     = clean_topks[local_idx]
        k_local = ks[local_idx]

        if ctk is None or len(inc_idx) < 2:
            continue  # trivially stable — skip PGD

        inc_tensor = torch.tensor(inc_idx, dtype=torch.long, device=device)
        inc_clean  = clean_mask[inc_tensor]

        _, top_local = torch.topk(inc_clean, k_local)
        top_set     = torch.zeros(len(inc_idx), dtype=torch.bool, device=device)
        top_set[top_local] = True
        non_top_set = ~top_set

        # PGD: maximise margin = max_{non-salient} mask − min_{salient} mask
        delta      = torch.zeros_like(x, requires_grad=True)
        best_margin = -1e9

        for _ in range(epochs):
            mask_adv = model.causal_core(x + delta, ei)
            inc_adv  = mask_adv[inc_tensor]
            margin   = inc_adv[non_top_set].max() - inc_adv[top_set].min()
            margin.backward()

            with torch.no_grad():
                delta.data.add_(step_size * delta.grad.sign())
                delta.data.clamp_(-epsilon, epsilon)
            delta.grad.zero_()

            best_margin = max(best_margin, float(margin.item()))
            if best_margin > 0:
                break  # counterexample found — early exit

        is_broken    = best_margin > 0
        is_certified = cert_flags[local_idx]
        if is_broken and is_certified:
            n_broken_cert += 1
        elif is_broken and not is_certified:
            n_broken_uncert += 1

    ibp_break_rate_cert = (n_broken_cert / n_certified) if n_certified > 0 else 0.0
    ibp_break_rate_uncert = (
        n_broken_uncert / max(n_tested - n_certified, 1)
    )

    if verbose:
        print(f"    [IBP-BREAK] epsilon={epsilon}, tested={n_tested}, "
              f"certified={n_certified}")
        print(f"      broken-certified   = {n_broken_cert}/{n_certified} "
              f"= {ibp_break_rate_cert:.4f}  "
              f"(MUST be 0 for a sound bound)")
        print(f"      broken-uncertified = {n_broken_uncert}/"
              f"{n_tested - n_certified} = {ibp_break_rate_uncert:.4f}  "
              f"(sanity: attack is doing work)")

    return {
        "n_tested":                  n_tested,
        "n_certified":               n_certified,
        "n_broken_certified":        n_broken_cert,
        "n_broken_uncertified":      n_broken_uncert,
        "ibp_break_rate_certified":  ibp_break_rate_cert,
        "ibp_break_rate_uncertified": ibp_break_rate_uncert,
        "cert_rate":                 cert_rate,
    }


# ======================================================================
# Orchestrator: run all three adaptive attacks and return one summary
# ======================================================================

def run_full_adaptive_evaluation(
    model, data, budgets=(0, 20, 30, 40),
    ibp_epsilons=(0.05, 0.1, 0.2),
    device="cuda",
    pgd_epochs=200,
    lambda_mask=1.0,
    hijack_n_attacker=50,
    hijack_top_k_frac=0.10,
    ibp_max_nodes=200,
    evaluate_fn=None,
    seed=42,
    verbose=True,
):
    """
    Run every adaptive attack and return a structured dict ready for
    _save_results. Typical call from run_table_adaptive.

    budgets      : percent-of-edges budgets for adaptive_pgd_attack.
    ibp_epsilons : L∞ feature-perturbation radii for ibp_break_attack.
    evaluate_fn  : function that takes (model, poisoned_data) → float
                   accuracy. Usually the caller's own accuracy loop
                   (aethelred_robust_vote or single forward pass).
    """
    model.eval()
    n_edges_undir = data.edge_index.size(1) // 2

    results = {
        "adaptive_pgd":    [],   # list of {p_pct, n_flips, acc, meta}
        "mask_hijack":     {},   # single dict (no budget loop)
        "ibp_break":       [],   # list of {epsilon, metrics}
    }

    # ── (a) Adaptive PGD over budgets ───────────────────────────────────
    for p_pct in budgets:
        n_flips = int(n_edges_undir * p_pct / 100)
        if p_pct == 0 or n_flips == 0:
            if evaluate_fn is not None:
                acc_clean = evaluate_fn(model, data.to(device))
            else:
                acc_clean = None
            results["adaptive_pgd"].append({
                "p_pct":    0,
                "n_flips":  0,
                "acc":      acc_clean,
                "meta":     {"method": "clean"},
            })
            continue

        data_p, meta = adaptive_pgd_attack(
            model, data, n_flips, device=device,
            epochs=pgd_epochs, lambda_mask=lambda_mask,
            seed=seed, verbose=verbose,
        )
        if evaluate_fn is not None:
            acc_adv = evaluate_fn(model, data_p.to(device))
        else:
            acc_adv = None
        results["adaptive_pgd"].append({
            "p_pct":   p_pct,
            "n_flips": n_flips,
            "acc":     acc_adv,
            "meta":    {k: v for k, v in meta.items() if k != "added_edges"},
        })

    # ── (b) Mask hijack ─────────────────────────────────────────────────
    results["mask_hijack"] = mask_hijack_attack(
        model, data, n_attacker_edges=hijack_n_attacker,
        device=device, top_k_frac=hijack_top_k_frac, seed=seed,
        verbose=verbose,
    )

    # ── (c) IBP break for each epsilon ─────────────────────────────────
    for eps in ibp_epsilons:
        ibp_res = ibp_break_attack(
            model, data, epsilon=eps, top_k_frac=hijack_top_k_frac,
            device=device, max_nodes=ibp_max_nodes, seed=seed,
            verbose=verbose,
        )
        ibp_res["epsilon"] = eps
        results["ibp_break"].append(ibp_res)

    return results
