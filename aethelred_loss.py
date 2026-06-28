# -*- coding: utf-8 -*-
"""
Project Aethelred — Composite Loss Function

L_total = L_task + α·L_invariance + β·L_IB + γ·L_sparsity + δ·L_acyclicity + ε·L_certify

Each term is individually computable and differentiable.
"""

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Individual loss terms
# --------------------------------------------------------------------------

def compute_acyclicity_loss(causal_edge_mask, edge_index, num_nodes):
    """
    Enforces the learned causal graph to be a DAG using the NOTEARS
    characterisation:  L_acyclicity = Tr(e^{A_c ∘ A_c}) − d

    For large graphs this is expensive; we skip gracefully on OOM.
    """
    device = causal_edge_mask.device

    # Build dense weighted adjacency from edge mask
    A = torch.zeros(num_nodes, num_nodes, device=device)
    src, dst = edge_index[0], edge_index[1]
    A[src, dst] = causal_edge_mask

    A_sq = A * A  # element-wise (Hadamard)
    try:
        trace = torch.trace(torch.matrix_exp(A_sq))
        return trace - num_nodes
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        # Fallback: approximate with power-series truncation
        # Tr(I + A² + A⁴/2) − d  ≈ Tr(A²) + Tr(A⁴)/2
        A2 = A_sq
        A4 = A2 @ A2
        return torch.trace(A2) + torch.trace(A4) / 2.0


def compute_invariance_loss(losses_per_env):
    """
    IRM-style penalty: Var_{e ∈ E}(L_task^e)
    Encourages prediction stability across artificial environments.
    """
    if len(losses_per_env) < 2:
        return torch.tensor(0.0, device=losses_per_env[0].device)
    return torch.var(torch.stack(losses_per_env))


def compute_sparsity_loss(causal_edge_mask):
    """L1 norm on causal mask — encourages minimality."""
    return torch.norm(causal_edge_mask, p=1) / causal_edge_mask.numel()


def compute_certification_loss(mask_low, mask_high, original_mask, top_k_frac=0.1, tau=0.5):
    """
    IBP-based certification loss.
    Pushes lower bounds of salient edges above τ and
    upper bounds of non-salient edges below τ.

    L_certify = Σ_{i ∈ salient} ReLU(τ − m_low_i)
              + Σ_{j ∈ non-salient} ReLU(m_high_j − τ)
    """
    k = max(1, int(original_mask.numel() * top_k_frac))
    _, top_k_idx = torch.topk(original_mask, k)

    non_salient_mask = torch.ones_like(original_mask, dtype=torch.bool)
    non_salient_mask[top_k_idx] = False

    loss_salient = F.relu(tau - mask_low[top_k_idx]).sum()
    loss_nonsalient = F.relu(mask_high[non_salient_mask] - tau).sum()

    total = (loss_salient + loss_nonsalient) / original_mask.numel()
    return total


def compute_mask_margin_loss(causal_edge_mask, top_k_frac=0.10, tau_margin=0.20):
    """
    Mask margin loss: explicitly push the score gap between salient and
    non-salient edges to be at least tau_margin.

    margin = min_{e ∈ top-K} mask[e]  −  max_{e ∉ top-K} mask[e]
    L_margin = ReLU(tau_margin − margin)   [hinge: zero when margin ≥ tau_margin]

    Root cause addressed: with only node-score averaging, the mask margin
    was ~0.075, making hijack_rate=1.0 and broken-cert>0.5. This loss
    directly trains the model to maintain a larger salient/non-salient gap,
    increasing the threshold the attacker must exceed.
    """
    E = causal_edge_mask.size(0)
    k = max(1, int(E * top_k_frac))
    if k >= E:
        return torch.tensor(0.0, device=causal_edge_mask.device)

    sorted_mask, _ = causal_edge_mask.sort(descending=True)
    min_salient    = sorted_mask[k - 1]   # lowest score still in top-K
    max_nonsalient = sorted_mask[k]        # highest score just outside top-K
    margin = min_salient - max_nonsalient  # positive = gap exists

    return F.relu(tau_margin - margin)     # penalise when gap < tau_margin


def compute_contrastive_negative_loss(structural_scores, negative_scores, margin=0.30):
    """
    Hinge: push mean structural-edge score above max negative-edge score by `margin`.

    Root cause addressed: the hijack attack works by scoring random non-edges
    with the edge MLP. With no negatives in training, the MLP has no incentive
    to score them low — they end up scoring as high as or higher than real edges.
    This loss explicitly trains the MLP on negative samples each step.

        L_contrastive = ReLU(margin − (mean(pos) − max(neg)))
    """
    if negative_scores is None or negative_scores.numel() == 0:
        return torch.tensor(0.0, device=structural_scores.device)
    mean_pos = structural_scores.mean()
    max_neg  = negative_scores.max()
    return F.relu(margin - (mean_pos - max_neg))


def compute_hijack_adversarial_loss(struct_scores, cand_scores, top_k_frac=0.10,
                                     struct_floor=0.50, cand_ceiling=0.30):
    """
    Adversarial training against the mask-top-K hijack attack, using *two
    independent absolute hinges* so the loss CANNOT be satisfied by score
    collapse (the failure mode of a relative-margin formulation):

        threshold = sort(struct, desc)[k-1]                 # lowest top-K structural
        L_struct  = ReLU(struct_floor − threshold)          # pull structural up
        L_cand    = ReLU(max(cand) − cand_ceiling)          # push candidates down
        L_hijack  = L_struct + L_cand

    `struct_scores` and `cand_scores` MUST be computed on the perturbed graph
    (edge_index with symmetric candidate non-edges appended) so the gradient
    targets the failure regime the attack actually operates in.
    """
    if cand_scores is None or cand_scores.numel() == 0:
        return torch.tensor(0.0, device=struct_scores.device)
    E = struct_scores.size(0)
    k = max(1, min(int(E * top_k_frac), E))
    sorted_struct, _ = struct_scores.sort(descending=True)
    threshold = sorted_struct[k - 1]
    max_cand = cand_scores.max()
    return F.relu(struct_floor - threshold) + F.relu(max_cand - cand_ceiling)


def compute_score_stability_loss(mask_clean, mask_struct_perturbed):
    """
    MSE anchor: structural-edge scores on the perturbed graph should match
    their scores on the clean graph. Prevents the GCN aggregation from
    silently rearranging which edges count as salient when an attacker
    injects extra edges.

    `mask_clean` is detached so the anchor pulls the perturbed scores toward
    the (current) clean scores rather than collapsing both.
    """
    return F.mse_loss(mask_struct_perturbed, mask_clean.detach())


def compute_mask_floor_loss(causal_edge_mask, top_k_frac=0.10, tau_floor=0.50):
    """
    Absolute floor on the salient mask: lowest top-K score must exceed tau_floor.

    Root cause addressed: the mask margin loss only constrains the *gap*; the
    optimizer can satisfy it by collapsing all scores toward zero (which is
    what we observed: scores ≈ 0.04 with margin ≈ 0). This term forces the
    salient scores to stay near sigmoid-saturation, keeping the absolute scale
    high so candidate-edge scores cannot drift above the top-K threshold.
    """
    E = causal_edge_mask.size(0)
    k = max(1, int(E * top_k_frac))
    sorted_mask, _ = causal_edge_mask.sort(descending=True)
    min_salient = sorted_mask[k - 1]
    return F.relu(tau_floor - min_salient)


# --------------------------------------------------------------------------
# Composite loss
# --------------------------------------------------------------------------

def compute_composite_loss(
    final_logits,
    causal_edge_mask,
    data,
    train_mask,
    losses_per_env,
    hparams,
    mask_low=None,
    mask_high=None,
    task='node',
    negative_scores=None,
    perturbed_struct_scores=None,
    perturbed_cand_scores=None,
    clean_mask_for_stability=None,
):
    """
    Computes the full Aethelred composite loss.

    Parameters
    ----------
    final_logits : Tensor   — model predictions
    causal_edge_mask : Tensor — learned causal mask
    data : Data             — PyG data object
    train_mask : Tensor     — boolean mask for training nodes/graphs
    losses_per_env : list[Tensor] — per-environment task losses
    hparams : dict          — loss hyperparameters (alpha, beta, gamma, delta, epsilon)
    mask_low, mask_high : optional IBP bounds on the mask
    task : 'node' or 'graph'

    Returns
    -------
    total_loss, loss_dict
    """
    # L_task
    if task == 'node':
        loss_task = F.cross_entropy(final_logits[train_mask], data.y[train_mask])
    else:
        loss_task = F.cross_entropy(final_logits, data.y)

    # L_invariance
    loss_invariance = compute_invariance_loss(losses_per_env)

    # L_sparsity
    loss_sparsity = compute_sparsity_loss(causal_edge_mask)

    # L_IB  (sparsity as proxy for compression I(G; G_c))
    loss_ib = loss_sparsity

    # L_acyclicity
    # Guard: matrix_exp on a dense [N×N] adjacency is O(N³).  A PROTEINS
    # batch of 64 graphs (~2560 nodes) already costs ~16B FLOPs per step —
    # way too expensive.  500 nodes is the safe upper limit for this loss.
    alpha_acyc = hparams.get('delta', 1.0)
    if alpha_acyc > 0 and data.x.size(0) <= 500:
        loss_acyclicity = compute_acyclicity_loss(
            causal_edge_mask, data.edge_index, data.x.size(0)
        )
    else:
        loss_acyclicity = torch.tensor(0.0, device=final_logits.device)

    # L_certify
    if mask_low is not None and mask_high is not None:
        loss_certify = compute_certification_loss(
            mask_low, mask_high, causal_edge_mask,
            top_k_frac=hparams.get('certify_top_k', 0.1),
            tau=hparams.get('certify_tau', 0.5),
        )
    else:
        loss_certify = torch.tensor(0.0, device=final_logits.device)

    # L_mask_margin: push salient/non-salient gap ≥ tau_margin.
    # Directly addresses hijack_rate=1.0 and high broken-cert caused by
    # the tiny natural gap (~0.075) from node-score averaging in CausalDiscoveryCore.
    mask_margin_w = hparams.get('mask_margin_w', 0.0)
    if mask_margin_w > 0.0:
        loss_mask_margin = compute_mask_margin_loss(
            causal_edge_mask,
            top_k_frac=hparams.get('certify_top_k', 0.1),
            tau_margin=hparams.get('mask_margin_tau', 0.20),
        )
    else:
        loss_mask_margin = torch.tensor(0.0, device=final_logits.device)

    # L_contrastive: push mean(structural) − max(negative) ≥ contrastive_margin.
    # Trains the edge MLP on random non-edges so candidate-edge scores can't
    # exceed structural-edge scores (the hijack failure mode).
    contrastive_w = hparams.get('contrastive_w', 0.0)
    if contrastive_w > 0.0 and negative_scores is not None:
        loss_contrastive = compute_contrastive_negative_loss(
            causal_edge_mask, negative_scores,
            margin=hparams.get('contrastive_margin', 0.30),
        )
    else:
        loss_contrastive = torch.tensor(0.0, device=final_logits.device)

    # L_floor: keep salient-edge scores near saturation so the margin loss
    # cannot be satisfied by collapsing all scores toward zero.
    floor_w = hparams.get('mask_floor_w', 0.0)
    if floor_w > 0.0:
        loss_floor = compute_mask_floor_loss(
            causal_edge_mask,
            top_k_frac=hparams.get('certify_top_k', 0.1),
            tau_floor=hparams.get('mask_floor_tau', 0.50),
        )
    else:
        loss_floor = torch.tensor(0.0, device=final_logits.device)

    # L_hijack_adv: trains directly against the perturbed-graph hijack regime.
    # Requires the caller to forward causal_core on (edge_index ‖ candidate_ei)
    # and split the resulting mask into structural and candidate slices.
    hijack_adv_w = hparams.get('hijack_adv_w', 0.0)
    if (hijack_adv_w > 0.0
            and perturbed_struct_scores is not None
            and perturbed_cand_scores is not None):
        loss_hijack_adv = compute_hijack_adversarial_loss(
            perturbed_struct_scores, perturbed_cand_scores,
            top_k_frac=hparams.get('certify_top_k', 0.1),
            struct_floor=hparams.get('hijack_adv_struct_floor', 0.50),
            cand_ceiling=hparams.get('hijack_adv_cand_ceiling', 0.30),
        )
    else:
        loss_hijack_adv = torch.tensor(0.0, device=final_logits.device)

    # L_stability: anchor perturbed-graph structural scores to clean-graph scores.
    stab_w = hparams.get('score_stability_w', 0.0)
    if (stab_w > 0.0
            and clean_mask_for_stability is not None
            and perturbed_struct_scores is not None):
        loss_stability = compute_score_stability_loss(
            clean_mask_for_stability, perturbed_struct_scores
        )
    else:
        loss_stability = torch.tensor(0.0, device=final_logits.device)

    total_loss = (
        loss_task
        + hparams.get('alpha', 1.0)  * loss_invariance
        + hparams.get('beta',  0.01) * loss_ib
        + hparams.get('gamma', 0.1)  * loss_sparsity
        + hparams.get('delta', 1.0)  * loss_acyclicity
        + hparams.get('epsilon', 0.1) * loss_certify
        + mask_margin_w               * loss_mask_margin
        + contrastive_w               * loss_contrastive
        + floor_w                     * loss_floor
        + hijack_adv_w                * loss_hijack_adv
        + stab_w                      * loss_stability
    )

    loss_dict = {
        'total':       total_loss.item(),
        'task':        loss_task.item(),
        'invariance':  loss_invariance.item(),
        'sparsity':    loss_sparsity.item(),
        'acyclicity':  loss_acyclicity.item(),
        'certify':     loss_certify.item(),
        'mask_margin': loss_mask_margin.item(),
        'contrastive': loss_contrastive.item(),
        'floor':       loss_floor.item(),
        'hijack_adv':  loss_hijack_adv.item(),
        'stability':   loss_stability.item(),
    }

    return total_loss, loss_dict
