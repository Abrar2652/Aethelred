# -*- coding: utf-8 -*-
"""
Project Aethelred — Formal Certification of Explanation Stability

Replaces the old vote-margin certification from PGNNCert/SECert.
Two mechanisms:
  1. Lipschitz Constant Analysis  — bounds ‖ΔG_c‖ ≤ L_g · ε
  2. Interval Bound Propagation   — certifies sparsity pattern of G_c
"""

import torch


def estimate_lipschitz_constant(model, data, num_samples=50, epsilon=0.01):
    """
    Empirically estimate the Lipschitz constant of the CausalDiscoveryCore
    with respect to input features x.

    L_g ≈ max_i ‖mask(x+δ_i) − mask(x)‖ / ‖δ_i‖

    This provides an empirical upper bound; a formal bound requires
    spectral-norm analysis of each layer's weight matrices.
    """
    model.eval()
    x, edge_index = data.x, data.edge_index

    with torch.no_grad():
        original_mask = model.causal_core(x, edge_index)

    max_ratio = 0.0

    for _ in range(num_samples):
        delta = torch.randn_like(x) * epsilon
        delta_norm = torch.norm(delta).item()
        if delta_norm < 1e-10:
            continue

        with torch.no_grad():
            perturbed_mask = model.causal_core(x + delta, edge_index)

        mask_diff = torch.norm(perturbed_mask - original_mask).item()
        ratio = mask_diff / delta_norm
        max_ratio = max(max_ratio, ratio)

    return max_ratio


def certify_explanation_stability(
    model, data,
    perturbation_budget=0.1,
    top_k_frac=0.1,
    verbose=True,
):
    """
    Certify that the causal explanation G_c is stable under input
    perturbations of size ≤ perturbation_budget (L∞).

    Uses IBP through the CausalDiscoveryCore to get rigorous bounds
    [mask_low, mask_high] on the edge mask.

    The explanation is certified stable if:
        min_{i ∈ salient} mask_low_i  >  max_{j ∈ non-salient} mask_high_j

    i.e., under worst-case perturbation, salient edges remain more important
    than non-salient edges.

    Returns
    -------
    is_certified : bool
    """
    model.eval()
    x, edge_index = data.x, data.edge_index

    with torch.no_grad():
        _, original_mask = model(data)

    # Define the L∞ perturbation set
    x_low = x - perturbation_budget
    x_high = x + perturbation_budget

    # Propagate bounds through CausalDiscoveryCore via IBP
    with torch.no_grad():
        mask_low, mask_high = model.causal_core.ibp_forward(x_low, x_high, edge_index)

    # Identify top-k salient edges
    k = max(1, int(original_mask.numel() * top_k_frac))
    _, top_k_indices = torch.topk(original_mask, k)

    # Salient lower bound
    min_salient_bound = mask_low[top_k_indices].min().item()

    # Non-salient upper bound
    non_salient_mask = torch.ones_like(original_mask, dtype=torch.bool)
    non_salient_mask[top_k_indices] = False

    if non_salient_mask.any():
        max_nonsalient_bound = mask_high[non_salient_mask].max().item()
    else:
        max_nonsalient_bound = 0.0

    # Certification check
    margin = min_salient_bound - max_nonsalient_bound
    is_certified = margin > 0

    if verbose:
        print(f"  Explanation Certified Stable: {is_certified}")
        print(f"    Min Salient Lower Bound:     {min_salient_bound:.6f}")
        print(f"    Max Non-Salient Upper Bound: {max_nonsalient_bound:.6f}")
        print(f"    Margin:                      {margin:.6f}")

        # Also report empirical Lipschitz estimate
        L_g = estimate_lipschitz_constant(model, data)
        max_mask_change = L_g * perturbation_budget
        print(f"    Empirical Lipschitz L_g:      {L_g:.4f}")
        print(f"    Max mask change (L_g·ε):      {max_mask_change:.6f}")

    return is_certified


def certify_nodes_batch(model, data, perturbation_budget, test_mask=None,
                        top_k_frac=0.1, max_nodes=None):
    """
    Per-node explanation certification for node classification.

    Runs IBP once on the full graph, then for each test node checks whether
    its incident-edge explanation (top-k edges by mask score) is stable under
    L-inf feature perturbations of size `perturbation_budget`.

    A node is certified if:
        min_{e in top-k}  mask_low[e]  >  max_{e not in top-k}  mask_high[e]

    Parameters
    ----------
    model              : Aethelred model (eval mode)
    data               : PyG Data object (on same device as model)
    perturbation_budget: float  — L-inf radius in feature space (eps = p * 0.01)
    test_mask          : BoolTensor [N] — which nodes to certify (default: all)
    top_k_frac         : float  — fraction of incident edges treated as salient
    max_nodes          : int    — cap number of test nodes (for large datasets)

    Returns
    -------
    cert_mask : torch.BoolTensor shape [n_test]  — True = certified
    cert_rate : float                            — fraction certified
    """
    model.eval()
    x, edge_index = data.x, data.edge_index

    # Global forward for original mask and IBP bounds
    with torch.no_grad():
        _, original_mask = model(data)
        x_low  = x - perturbation_budget
        x_high = x + perturbation_budget
        mask_low, mask_high = model.causal_core.ibp_forward(x_low, x_high, edge_index)

    # Test node indices
    if test_mask is None:
        test_nodes = torch.arange(x.size(0), device=x.device)
    else:
        test_nodes = test_mask.nonzero(as_tuple=False).view(-1)

    if max_nodes is not None and test_nodes.size(0) > max_nodes:
        test_nodes = test_nodes[:max_nodes]

    # Pre-build incident-edge lookup: node -> list of edge indices
    src = edge_index[0]
    incident_map = {}
    for ei in range(edge_index.size(1)):
        v = src[ei].item()
        if v not in incident_map:
            incident_map[v] = []
        incident_map[v].append(ei)

    cert_flags = []
    for v_t in test_nodes:
        v = v_t.item()
        inc_idx = incident_map.get(v, [])

        if len(inc_idx) == 0:
            # Isolated node — no edges to certify, trivially stable
            cert_flags.append(True)
            continue

        inc_tensor = torch.tensor(inc_idx, dtype=torch.long, device=x.device)
        inc_orig  = original_mask[inc_tensor]
        inc_low   = mask_low[inc_tensor]
        inc_high  = mask_high[inc_tensor]

        k = max(1, int(len(inc_idx) * top_k_frac))

        if k >= len(inc_idx):
            # All edges are salient — trivially certified
            cert_flags.append(True)
            continue

        _, top_k_local = torch.topk(inc_orig, k)
        non_top_k_mask = torch.ones(len(inc_idx), dtype=torch.bool, device=x.device)
        non_top_k_mask[top_k_local] = False

        min_salient    = inc_low[top_k_local].min().item()
        max_nonsalient = inc_high[non_top_k_mask].max().item()

        cert_flags.append(min_salient > max_nonsalient)

    cert_mask = torch.tensor(cert_flags, dtype=torch.bool)
    cert_rate = cert_mask.float().mean().item() if cert_mask.numel() > 0 else 0.0
    return cert_mask, cert_rate


def certify_batch(model, graphs, perturbation_budget=0.1, top_k_frac=0.1):
    """
    Run certification on a batch of graphs.
    Returns the fraction that are certified stable.
    """
    certified = 0
    total = len(graphs)

    for g in graphs:
        is_cert = certify_explanation_stability(
            model, g,
            perturbation_budget=perturbation_budget,
            top_k_frac=top_k_frac,
            verbose=False,
        )
        certified += int(is_cert)

    rate = certified / max(total, 1)
    print(f"Certified: {certified}/{total} ({100*rate:.1f}%)")
    return rate
