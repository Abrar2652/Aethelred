# -*- coding: utf-8 -*-
"""
DIR-GNN Baseline — Discovering Invariant Rationales for Graph Neural Networks
(Wu et al., ICLR 2022)

Faithful implementation of DIR's core training:
  1. Rationale generator  : GCN producing per-edge importance (causal subgraph)
  2. Causal classifier    : GCN on the rationale subgraph → class prediction
  3. Context classifier   : GCN on the complement subgraph → class prediction
  4. Min-max training:
       - causal_cls + context_cls trained to minimise their CE (standard)
       - rationale_gen trained to minimise causal CE and MAXIMISE context CE
         (adversarial: make complement un-classifiable)
  5. IRM penalty across edge-drop environments
  6. Sparsity + binary-entropy regularisation on the mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class DIRRationaleGenerator(nn.Module):
    """
    2-layer GCN that outputs per-edge soft importance in [0, 1].
    Architecturally identical to Aethelred's CausalDiscoveryCore so
    comparisons are apples-to-apples on capacity.
    """

    def __init__(self, num_features, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        node_scores = self.conv2(h, edge_index).squeeze(-1)
        edge_scores = (node_scores[edge_index[0]] + node_scores[edge_index[1]]) / 2.0
        return torch.sigmoid(edge_scores)   # [num_edges] in [0, 1]


class _GCNClassifier(nn.Module):
    """Simple 2-layer GCN with global mean pooling for graph classification."""

    def __init__(self, num_features, num_classes, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, edge_weight, batch):
        h = F.relu(self.conv1(x, edge_index, edge_weight))
        h = F.relu(self.conv2(h, edge_index, edge_weight))
        h = global_mean_pool(h, batch)
        return self.lin(h)


class DIRModel(nn.Module):
    """
    Full DIR model for graph classification.

    forward() returns (logits_causal, logits_context, edge_mask).
    The edge_mask is the explanation (rationale subgraph).
    """

    def __init__(self, num_features, num_classes, hidden_dim=64):
        super().__init__()
        self.rationale_gen = DIRRationaleGenerator(num_features, hidden_dim)
        self.causal_cls = _GCNClassifier(num_features, num_classes, hidden_dim)
        self.context_cls = _GCNClassifier(num_features, num_classes, hidden_dim)

    def forward(self, data):
        x = data.x.float()
        ei = data.edge_index
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None \
            else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        edge_mask = self.rationale_gen(x, ei)          # [E]
        context_mask = 1.0 - edge_mask                  # complement

        logits_causal = self.causal_cls(x, ei, edge_mask, batch)
        logits_context = self.context_cls(x, ei, context_mask, batch)

        return logits_causal, logits_context, edge_mask


def train_dir(
    train_graphs,
    val_graphs,
    test_graphs,
    num_features,
    num_classes,
    device='cuda',
    hidden_dim=64,
    epochs=200,
    lr=0.001,
    n_envs=5,
    irm_lambda=1.0,
    warmup_epochs=60,
    adv_w=0.20,
    seed=42,
):
    """
    Train a DIR model for graph classification.

    Faithful to Wu et al. (ICLR 2022):
      - context_cls is trained to classify from the complement (standard CE)
      - rationale_gen is trained ADVERSARIALLY w.r.t. context_cls:
        it maximises the context loss (KL to uniform), forcing the mask
        to put all class-relevant edges into the causal subgraph
      - Sparsity regulariser pins the mask to a target budget

    Parameters
    ----------
    train_graphs, val_graphs, test_graphs : lists of PyG Data objects
    num_features, num_classes : int
    device       : 'cuda' or 'cpu'
    hidden_dim   : hidden dimension for all GCN layers
    epochs       : total training epochs
    lr           : Adam learning rate
    n_envs       : number of IRM environments (edge-drop views)
    irm_lambda   : weight for the IRM variance penalty
    warmup_epochs: epochs to train classifiers only before activating the
                   adversarial objective (prevents early collapse on small
                   datasets; adv weight ramps linearly 0→adv_w after warmup)
    adv_w        : final adversarial loss weight (default 0.20, reduced from
                   0.50 for stability on small datasets like MUTAG)
    seed         : RNG seed

    Returns
    -------
    model : trained DIRModel (eval mode)
    test_acc : float
    """
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Batch

    torch.manual_seed(seed)

    train_graphs = [g.cpu() for g in train_graphs]
    val_graphs   = [g.cpu() for g in val_graphs]
    test_graphs  = [g.cpu() for g in test_graphs]

    model = DIRModel(num_features, num_classes, hidden_dim).to(device)

    # Two optimisers: one for classifiers, one for rationale generator
    opt_cls = torch.optim.Adam(
        list(model.causal_cls.parameters()) + list(model.context_cls.parameters()),
        lr=lr, weight_decay=5e-4,
    )
    opt_gen = torch.optim.Adam(
        model.rationale_gen.parameters(),
        lr=lr, weight_decay=5e-4,
    )

    loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    best_val = 0.0
    best_test = 0.0
    best_state = None

    mask_budget = 0.25       # target sparsity (matching Aethelred)
    spar_w = 0.30
    ent_w = 0.10
    # adv_w comes from parameter; ramped up after warmup_epochs

    ramp_epochs = max(1, epochs - warmup_epochs)

    for epoch in range(epochs):
        # Linear ramp: adv and IRM only activate after warmup
        if epoch < warmup_epochs:
            cur_adv_w = 0.0
            cur_irm_w = 0.0
        else:
            t = (epoch - warmup_epochs) / ramp_epochs
            cur_adv_w = adv_w * t
            cur_irm_w = irm_lambda * t

        model.train()
        for batch in loader:
            batch = batch.to(device)
            b_v = batch.batch if hasattr(batch, 'batch') and batch.batch is not None \
                else torch.zeros(batch.x.size(0), dtype=torch.long, device=device)

            # ── Step 1: Train classifiers (rationale_gen frozen) ─────────
            for p in model.rationale_gen.parameters():
                p.requires_grad_(False)

            opt_cls.zero_grad()
            lc, lctx, edge_mask = model(batch)
            loss_c = F.cross_entropy(lc, batch.y)
            loss_ctx = F.cross_entropy(lctx, batch.y)
            (loss_c + loss_ctx).backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.causal_cls.parameters()) + list(model.context_cls.parameters()), 2.0)
            opt_cls.step()

            for p in model.rationale_gen.parameters():
                p.requires_grad_(True)

            # ── Step 2: Train rationale generator (context_cls frozen) ───
            for p in model.context_cls.parameters():
                p.requires_grad_(False)

            opt_gen.zero_grad()

            lc2, lctx2, edge_mask2 = model(batch)
            loss_causal = F.cross_entropy(lc2, batch.y)

            # Adversarial: push context toward uniform (ramped in after warmup)
            if cur_adv_w > 0:
                log_p_ctx = F.log_softmax(lctx2, dim=1)
                uniform = torch.full_like(log_p_ctx, 1.0 / num_classes)
                loss_adv = F.kl_div(log_p_ctx, uniform, reduction='batchmean')
            else:
                loss_adv = torch.tensor(0.0, device=device)

            # IRM across edge-drop environments (ramped in after warmup)
            if cur_irm_w > 0:
                env_losses = [loss_causal]
                for _ in range(n_envs - 1):
                    keep = torch.rand(batch.edge_index.size(1), device=device) > 0.15
                    if keep.sum() < 4:
                        continue
                    ei_env = batch.edge_index[:, keep]
                    mask_env = model.rationale_gen(batch.x.float(), ei_env)
                    lc_env = model.causal_cls(batch.x.float(), ei_env, mask_env, b_v)
                    env_losses.append(F.cross_entropy(lc_env, batch.y))
                irm_penalty = torch.var(torch.stack(env_losses)) if len(env_losses) > 1 \
                    else torch.tensor(0.0, device=device)
            else:
                irm_penalty = torch.tensor(0.0, device=device)

            loss_spar = (edge_mask2.mean() - mask_budget).abs()
            eps_e = 1e-6
            loss_ent = -(
                edge_mask2 * (edge_mask2 + eps_e).log()
                + (1.0 - edge_mask2) * (1.0 - edge_mask2 + eps_e).log()
            ).mean()

            loss_gen = (loss_causal
                        + cur_adv_w * loss_adv
                        + cur_irm_w * irm_penalty
                        + spar_w * loss_spar
                        + ent_w * loss_ent)
            loss_gen.backward()
            torch.nn.utils.clip_grad_norm_(model.rationale_gen.parameters(), 2.0)
            opt_gen.step()

            for p in model.context_cls.parameters():
                p.requires_grad_(True)

        # Validation
        model.eval()
        with torch.no_grad():
            vb = Batch.from_data_list(val_graphs).to(device)
            vl, _, _ = model(vb)
            val_acc = (vl.argmax(1) == vb.y).float().mean().item()

            tb = Batch.from_data_list(test_graphs).to(device)
            tl, _, _ = model(tb)
            test_acc = (tl.argmax(1) == tb.y).float().mean().item()

        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_test


def dir_get_explanation(model, data, device='cuda'):
    """
    Get DIR edge importance mask for a single graph.

    Parameters
    ----------
    model : DIRModel (eval mode)
    data  : PyG Data object

    Returns
    -------
    edge_mask : Tensor [num_edges] in [0, 1]
    """
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        _, _, edge_mask = model(data)
    return edge_mask.cpu()
