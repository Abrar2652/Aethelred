# -*- coding: utf-8 -*-
"""
Project Aethelred — Core Architecture

Three pillars:
  1. CausalDiscoveryCore  – learns a differentiable causal edge mask G_c
  2. FocalEngine           – GNN whose message passing is gated by G_c
  3. Aethelred             – top-level model integrating all pillars
     (supports both node and graph classification)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool


# ============================================================================
# Pillar 1: Causal Discovery Core
# ============================================================================

class CausalDiscoveryCore(nn.Module):
    """
    Learns a differentiable, sparse causal graph mask (G_c).

    Architecture — PROPAGATION-FREE edge scoring:
        h_u = MLP(x_u)                         # depends ONLY on node u's features
        edge_feat = cat(h_u, h_v, |h_u − h_v|, cos(x_u, x_v))
        edge_score = sigmoid(MLP(edge_feat))

    Why no GCN here: the previous GCN-based scorer aggregated messages through
    the very edge being scored. An attacker-added edge (u,v) would propagate
    u ↔ v messages during the causal_core forward, making h_u, h_v more
    similar — which the edge MLP interpreted as "salient". That self-referential
    loop made hijack_rate=1.0 regardless of any loss tuning.

    With an MLP encoder, h_u depends only on x_u. Adding edges to edge_index
    cannot change any node's embedding, so the attacker cannot manufacture
    spurious similarity. Candidate edges between random pairs have low feature
    cosine similarity (Cora-ML is strongly homophilic; random pairs are not)
    and are scored low. Real structural edges retain high similarity and stay
    salient.

    Focal engine keeps its GCN — it still benefits from propagation. Only the
    scorer is decoupled from the graph.
    """

    def __init__(self, input_dim, hidden_dim=64, membership_decay=0.20):
        super().__init__()
        # Propagation-free node encoder: h_u = MLP(x_u)
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Edge scorer consumes: [h_u, h_v, |h_u−h_v|, cos(x_u, x_v)]
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.constant_(self.edge_mlp[-1].bias, 0.85)

        # ── Structural-prior allowlist (transductive defense) ─────────────
        # Feature-based scoring alone cannot separate real edges from
        # same-class non-edges in homophilic graphs (Cora-ML has ~7 classes;
        # ~14% of random non-edges are same-class and have high feature
        # cosine). The allowlist injects the one bit no feature scorer can
        # infer: whether an edge was in the training adjacency.
        #
        # score(u,v) = MLP_score(u,v) × (1 if (u,v) ∈ E_train else membership_decay)
        #
        # With membership_decay = 0.20, every unseen edge has a HARD upper
        # bound of 0.20 on its mask score. As long as ≥ top-K structural
        # edges score above 0.20, the hijack attack is mathematically
        # impossible to succeed.
        #
        # Activated via register_training_graph(); remains off for graph
        # classification (each test graph is novel).
        self.membership_decay = float(membership_decay)
        # Empty placeholder buffer; resized by register_training_graph().
        self.register_buffer('training_adj',
                             torch.zeros(0, 0, dtype=torch.bool),
                             persistent=True)

    def register_training_graph(self, edge_index, num_nodes):
        """Snapshot the training adjacency so inference-time edges outside
        this set are downweighted by `membership_decay`. Call once after
        data is loaded, before training begins. Re-call after checkpoint
        restore in the transductive setting."""
        dev = self.training_adj.device
        A = torch.zeros(num_nodes, num_nodes, dtype=torch.bool, device=dev)
        ei = edge_index.to(dev)
        A[ei[0], ei[1]] = True
        self.training_adj = A

    def _allowlist_active(self):
        return self.training_adj.numel() > 0

    def _node_embeddings(self, x, edge_index=None):
        """Per-node encoding. `edge_index` kept in signature for call-site
        compatibility but is intentionally UNUSED — the whole point is that
        node embeddings do not depend on the graph topology."""
        return self.node_encoder(x)

    def score_edges(self, h, edge_index_query, x_raw):
        """Score arbitrary query edges given precomputed node embeddings h
        and the raw feature tensor x_raw (for cosine similarity).

        If the structural-prior allowlist is active (register_training_graph
        was called), edges outside the training adjacency are multiplied by
        `self.membership_decay`.
        """
        h_u = h[edge_index_query[0]]
        h_v = h[edge_index_query[1]]
        x_u = x_raw[edge_index_query[0]]
        x_v = x_raw[edge_index_query[1]]
        cos_sim = F.cosine_similarity(x_u, x_v, dim=-1, eps=1e-8).unsqueeze(-1)
        edge_feat = torch.cat([h_u, h_v, (h_u - h_v).abs(), cos_sim], dim=-1)
        raw = torch.sigmoid(self.edge_mlp(edge_feat).squeeze(-1))

        if self._allowlist_active():
            n = self.training_adj.size(0)
            u = edge_index_query[0].clamp(max=n - 1)
            v = edge_index_query[1].clamp(max=n - 1)
            is_known = self.training_adj[u, v].to(raw.dtype)
            multiplier = is_known + (1.0 - is_known) * self.membership_decay
            raw = raw * multiplier
        return raw

    def forward(self, x, edge_index):
        """
        Returns
        -------
        edge_mask : Tensor of shape (num_edges,) with values in [0, 1]
        """
        h = self._node_embeddings(x)
        return self.score_edges(h, edge_index, x_raw=x)

    def ibp_forward(self, x_low, x_high, edge_index):
        """
        Empirical interval bound via two-point evaluation.

        Now tighter than before: with no graph propagation, each edge score
        depends only on the two endpoints' features — no multi-hop aggregation
        can blow up the interval. Formal certification still uses Monte Carlo
        (see aethelred_certify.py); this bound is used during training as a
        regularizer for the certification loss term.
        """
        with torch.no_grad():
            mask_at_low  = self.forward(x_low,  edge_index)
            mask_at_high = self.forward(x_high, edge_index)
        mask_lb = torch.min(mask_at_low, mask_at_high)
        mask_ub = torch.max(mask_at_low, mask_at_high)
        return mask_lb, mask_ub


# ============================================================================
# Pillar 2: Focal Engine
# ============================================================================

class FocalEngine(nn.Module):
    """
    GNN whose message passing is gated by a causal edge mask.
    Supports GCN, GSAGE, and GAT backbones (matching PGNNCert's 3 settings).

    Edge-weight gating is only applied for GCN (SAGEConv/GATConv don't accept
    edge_weight). For GSAGE and GAT the causal mask is still trained through
    the composite loss terms; message passing uses the full edge_index.
    """

    _CONV_MAP = {'GCN': GCNConv, 'GSAGE': SAGEConv, 'GAT': GATConv}

    def __init__(self, num_features, num_classes, hidden_size=20,
                 num_layers=3, conv_type='GCN', gate_lambda=1.0):
        super().__init__()
        if conv_type not in self._CONV_MAP:
            raise ValueError(f"conv_type must be one of {list(self._CONV_MAP)}, got '{conv_type}'")
        self.conv_type = conv_type
        # GCN supports edge_weight for direct causal gating; GSAGE/GAT do not
        self.supports_edge_weight = (conv_type == 'GCN')
        # Residual gating strength. The effective edge weight is
        #     ew = (1 - gate_lambda) + gate_lambda * causal_mask
        # so gate_lambda=1.0 reproduces the original full gating, gate_lambda=0.0
        # is a vanilla GCN (ew=1), and intermediate values blend the two. This
        # decouples clean-accuracy from the explanation: the prediction backbone
        # can stay near-vanilla while the causal mask (returned unchanged) still
        # drives the explanation and the edge certificate, which only need the
        # mask *scores* for ranking, not the gating of message passing.
        # Motivated by the Amazon-C regression (full gating cost -21 pts at
        # matched capacity on a dense graph); see CLAUDE.md.
        self.gate_lambda = float(gate_lambda)

        ConvClass = self._CONV_MAP[conv_type]
        self.convs = nn.ModuleList()
        self.convs.append(ConvClass(num_features, hidden_size))
        for _ in range(num_layers - 1):
            self.convs.append(ConvClass(hidden_size, hidden_size))
        self.lin = nn.Linear(hidden_size * num_layers, num_classes)

    def get_node_embeddings(self, x, edge_index, causal_edge_mask):
        """Return concatenated hidden representations (before final linear)."""
        stack = []
        h = x
        # Residual blend so gate_lambda dials between vanilla GCN and full gating.
        if self.supports_edge_weight and causal_edge_mask is not None:
            if self.gate_lambda >= 1.0:
                edge_weight = causal_edge_mask
            else:
                edge_weight = (1.0 - self.gate_lambda) + self.gate_lambda * causal_edge_mask
        else:
            edge_weight = None
        for conv in self.convs:
            if self.supports_edge_weight:
                h = conv(h, edge_index, edge_weight=edge_weight)
            else:
                h = conv(h, edge_index)
            h = F.normalize(h, p=2, dim=1)
            h = F.relu(h)
            stack.append(h)
        return torch.cat(stack, dim=1)

    def forward(self, x, edge_index, causal_edge_mask):
        out = self.get_node_embeddings(x, edge_index, causal_edge_mask)
        return self.lin(out)

    # --- OLD VERSION (before graph fix) ---
    # def forward(self, x, edge_index, causal_edge_mask):
    #     stack = []
    #     h = x
    #     for conv in self.convs:
    #         if self.implementation == 'gating':
    #             h = conv(h, edge_index, edge_weight=causal_edge_mask)
    #         else:
    #             h = conv(h, edge_index)
    #         h = F.normalize(h, p=2, dim=1)
    #         h = F.relu(h)
    #         stack.append(h)
    #     out = torch.cat(stack, dim=1)
    #     return self.lin(out)


# ============================================================================
# Pillar 3: The Aethelred Model
# ============================================================================

class Aethelred(nn.Module):
    """
    Complete Aethelred model integrating all three pillars.
    Supports both node classification and graph classification.
    """

    def __init__(self, num_features, num_classes,
                 hidden_dim_causal=64, hidden_dim_focal=20,
                 num_focal_layers=3, task='node', conv_type='GCN',
                 gate_lambda=1.0):
        super().__init__()
        self.task = task
        self.causal_core = CausalDiscoveryCore(num_features, hidden_dim_causal)
        self.focal_engine = FocalEngine(
            num_features, num_classes, hidden_dim_focal,
            num_layers=num_focal_layers, conv_type=conv_type,
            gate_lambda=gate_lambda,
        )
        # For graph classification, pool node embeddings then classify
        if task == 'graph':
            # Use the rich hidden representation, not the narrow num_classes logits
            graph_emb_dim = hidden_dim_focal * num_focal_layers  # same as FocalEngine concat dim
            # Dual Pooling: input is mean+max concatenated, so 2x the embedding dim
            self.graph_head = nn.Sequential(
                nn.Linear(graph_emb_dim * 2, graph_emb_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(graph_emb_dim, num_classes),
            )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 1. Discover causal mask
        causal_edge_mask = self.causal_core(x, edge_index)

        # 2. Focal Engine guided by mask
        if self.task == 'graph':
            node_emb = self.focal_engine.get_node_embeddings(x, edge_index, causal_edge_mask)
            batch = data.batch if hasattr(data, 'batch') and data.batch is not None \
                else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

            # Dual Pooling Readout: capture both average and peak signals
            graph_emb_mean = global_mean_pool(node_emb, batch)
            graph_emb_max = global_max_pool(node_emb, batch)
            graph_emb = torch.cat([graph_emb_mean, graph_emb_max], dim=1)

            logits = self.graph_head(graph_emb)
        else:
            logits = self.focal_engine(x, edge_index, causal_edge_mask)

        # --- OLD VERSION (before graph fix) ---
        # logits = self.focal_engine(x, edge_index, causal_edge_mask)
        # if self.task == 'graph':
        #     batch = data.batch if hasattr(data, 'batch') and data.batch is not None \
        #         else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        #     graph_emb = global_mean_pool(logits, batch)
        #     logits = self.graph_head(graph_emb)

        return logits, causal_edge_mask
