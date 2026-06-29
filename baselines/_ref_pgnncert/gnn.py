# -*- coding: utf-8 -*-
"""
GNN backbone architectures for both PGNNCert (baseline) and SECert (ours).

Includes:
  - Original independent classifiers: NodeGCN, GraphGCN, NodeGSAGE, GraphGSAGE, NodeGAT, GraphGAT
  - Shared-backbone variants for SECert: SharedNodeGCN, SharedGraphGCN, etc.
"""

import torch
from torch.nn import ReLU, Linear
from torch_geometric.nn import GCNConv, global_mean_pool, SAGEConv, GATConv
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Original PGNNCert backbones (kept identical for fair comparison)
# =============================================================================

class NodeGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_size=20):
        super(NodeGCN, self).__init__()
        self.embedding_size = hidden_size * 3
        self.conv1 = GCNConv(num_features, hidden_size)
        self.relu1 = ReLU()
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.conv3 = GCNConv(hidden_size, hidden_size)
        self.relu3 = ReLU()
        self.lin = Linear(3 * hidden_size, num_classes)

    def forward(self, x, edge_index):
        input_lin = self.embedding(x, edge_index)
        final = self.lin(input_lin)
        return final

    def embedding(self, x, edge_index):
        stack = []
        out1 = self.conv1(x, edge_index, edge_weight=None)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)
        out1 = self.relu1(out1)
        stack.append(out1)
        out2 = self.conv2(out1, edge_index)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)
        out2 = self.relu2(out2)
        stack.append(out2)
        out3 = self.conv3(out2, edge_index)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)
        out3 = self.relu3(out3)
        stack.append(out3)
        input_lin = torch.cat(stack, dim=1)
        return input_lin


class GraphGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_size=32):
        super(GraphGCN, self).__init__()
        self.embedding_size = hidden_size * 3
        self.conv1 = GCNConv(num_features, hidden_size)
        self.relu1 = ReLU()
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.conv3 = GCNConv(hidden_size, hidden_size)
        self.relu3 = ReLU()
        self.lin = Linear(hidden_size * 3, num_classes)

    def forward(self, x, edge_index, batch=None):
        input_lin = self.embedding(x, edge_index, batch=batch)
        final = self.lin(input_lin)
        return final

    def embedding(self, x, edge_index, batch=None):
        stack = []
        out1 = self.conv1(x, edge_index)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)
        out1 = self.relu1(out1)
        stack.append(out1)
        out2 = self.conv2(out1, edge_index)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)
        out2 = self.relu2(out2)
        stack.append(out2)
        out3 = self.conv3(out2, edge_index)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)
        out3 = self.relu3(out3)
        stack.append(out3)
        stack = torch.cat(stack, dim=1)
        if batch is None:
            batch = torch.zeros(stack.size(0), dtype=torch.long, device=stack.device)
        input_lin = global_mean_pool(stack, batch=batch)
        return input_lin


class NodeGSAGE(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_size=20):
        super(NodeGSAGE, self).__init__()
        self.embedding_size = hidden_size * 3
        self.conv1 = SAGEConv(num_features, hidden_size)
        self.relu1 = ReLU()
        self.conv2 = SAGEConv(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.conv3 = SAGEConv(hidden_size, hidden_size)
        self.relu3 = ReLU()
        self.lin = Linear(3 * hidden_size, num_classes)

    def forward(self, x, edge_index):
        input_lin = self.embedding(x, edge_index.to(torch.int64))
        final = self.lin(input_lin)
        return final

    def embedding(self, x, edge_index):
        stack = []
        out1 = self.conv1(x, edge_index)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)
        out1 = self.relu1(out1)
        stack.append(out1)
        out2 = self.conv2(out1, edge_index)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)
        out2 = self.relu2(out2)
        stack.append(out2)
        out3 = self.conv3(out2, edge_index)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)
        out3 = self.relu3(out3)
        stack.append(out3)
        input_lin = torch.cat(stack, dim=1)
        return input_lin


class GraphGSAGE(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_size=32):
        super(GraphGSAGE, self).__init__()
        self.embedding_size = hidden_size * 3
        self.conv1 = SAGEConv(num_features, hidden_size)
        self.relu1 = ReLU()
        self.conv2 = SAGEConv(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.conv3 = SAGEConv(hidden_size, hidden_size)
        self.relu3 = ReLU()
        self.lin = Linear(hidden_size * 3, num_classes)

    def forward(self, x, edge_index, batch=None):
        input_lin = self.embedding(x, edge_index, batch=batch)
        final = self.lin(input_lin)
        return final

    def embedding(self, x, edge_index, batch=None):
        stack = []
        out1 = self.conv1(x, edge_index)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)
        out1 = self.relu1(out1)
        stack.append(out1)
        out2 = self.conv2(out1, edge_index)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)
        out2 = self.relu2(out2)
        stack.append(out2)
        out3 = self.conv3(out2, edge_index)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)
        out3 = self.relu3(out3)
        stack.append(out3)
        stack = torch.cat(stack, dim=1)
        if batch is None:
            batch = torch.zeros(stack.size(0), dtype=torch.long, device=stack.device)
        input_lin = global_mean_pool(stack, batch=batch)
        return input_lin


class NodeGAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_size=20):
        super(NodeGAT, self).__init__()
        self.embedding_size = hidden_size * 3
        self.conv1 = GATConv(num_features, hidden_size)
        self.relu1 = ReLU()
        self.conv2 = GATConv(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.conv3 = GATConv(hidden_size, hidden_size)
        self.relu3 = ReLU()
        self.lin = Linear(3 * hidden_size, num_classes)

    def forward(self, x, edge_index):
        input_lin = self.embedding(x, edge_index)
        final = self.lin(input_lin)
        return final

    def embedding(self, x, edge_index):
        stack = []
        out1 = self.conv1(x, edge_index)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)
        out1 = self.relu1(out1)
        stack.append(out1)
        out2 = self.conv2(out1, edge_index)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)
        out2 = self.relu2(out2)
        stack.append(out2)
        out3 = self.conv3(out2, edge_index)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)
        out3 = self.relu3(out3)
        stack.append(out3)
        input_lin = torch.cat(stack, dim=1)
        return input_lin


class GraphGAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_size=32):
        super(GraphGAT, self).__init__()
        self.embedding_size = hidden_size * 3
        self.conv1 = GATConv(num_features, hidden_size)
        self.relu1 = ReLU()
        self.conv2 = GATConv(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.conv3 = GATConv(hidden_size, hidden_size)
        self.relu3 = ReLU()
        self.lin = Linear(hidden_size * 3, num_classes)

    def forward(self, x, edge_index, batch=None):
        input_lin = self.embedding(x, edge_index, batch=batch)
        final = self.lin(input_lin)
        return final

    def embedding(self, x, edge_index, batch=None):
        stack = []
        out1 = self.conv1(x, edge_index)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)
        out1 = self.relu1(out1)
        stack.append(out1)
        out2 = self.conv2(out1, edge_index)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)
        out2 = self.relu2(out2)
        stack.append(out2)
        out3 = self.conv3(out2, edge_index)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)
        out3 = self.relu3(out3)
        stack.append(out3)
        stack = torch.cat(stack, dim=1)
        if batch is None:
            batch = torch.zeros(stack.size(0), dtype=torch.long, device=stack.device)
        input_lin = global_mean_pool(stack, batch=batch)
        return input_lin


# =============================================================================
# SECert: Shared-Backbone GNN architectures
# =============================================================================
# Key insight: The GNN backbone is shared across ALL subgraph classifiers.
# Only lightweight per-subgraph classification heads differ.
# This maintains the deterministic certification guarantee because:
#   - At test time, parameters are FROZEN
#   - Perturbation to subgraph j only changes the INPUT for classifier j
#   - Other subgraphs' inputs (and thus outputs) are unaffected
#   - Therefore each perturbation changes at most 1 vote

class SharedNodeBackbone(nn.Module):
    """Shared GNN backbone for node classification (embedding extractor only)."""
    def __init__(self, num_features, hidden_size, conv_type="GCN"):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding_size = hidden_size * 3

        if conv_type == "GCN":
            ConvClass = GCNConv
        elif conv_type == "GSAGE":
            ConvClass = SAGEConv
        elif conv_type == "GAT":
            ConvClass = GATConv
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")

        self.conv1 = ConvClass(num_features, hidden_size)
        self.conv2 = ConvClass(hidden_size, hidden_size)
        self.conv3 = ConvClass(hidden_size, hidden_size)
        self.relu = ReLU()

    def forward(self, x, edge_index):
        stack = []
        out1 = self.conv1(x, edge_index)
        out1 = F.normalize(out1, p=2, dim=1)
        out1 = self.relu(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index)
        out2 = F.normalize(out2, p=2, dim=1)
        out2 = self.relu(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index)
        out3 = F.normalize(out3, p=2, dim=1)
        out3 = self.relu(out3)
        stack.append(out3)

        return torch.cat(stack, dim=1)


class SharedGraphBackbone(nn.Module):
    """Shared GNN backbone for graph classification (with global pooling)."""
    def __init__(self, num_features, hidden_size, conv_type="GCN"):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding_size = hidden_size * 3

        if conv_type == "GCN":
            ConvClass = GCNConv
        elif conv_type == "GSAGE":
            ConvClass = SAGEConv
        elif conv_type == "GAT":
            ConvClass = GATConv
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")

        self.conv1 = ConvClass(num_features, hidden_size)
        self.conv2 = ConvClass(hidden_size, hidden_size)
        self.conv3 = ConvClass(hidden_size, hidden_size)
        self.relu = ReLU()

    def forward(self, x, edge_index):
        stack = []
        out1 = self.conv1(x, edge_index)
        out1 = F.normalize(out1, p=2, dim=1)
        out1 = self.relu(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index)
        out2 = F.normalize(out2, p=2, dim=1)
        out2 = self.relu(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index)
        out3 = F.normalize(out3, p=2, dim=1)
        out3 = self.relu(out3)
        stack.append(out3)

        stack = torch.cat(stack, dim=1)
        return global_mean_pool(stack, batch=None)


class SubgraphHead(nn.Module):
    """Per-subgraph classification head (lightweight).

    Default: simple Linear layer (same as PGNNCert for fair comparison).
    Optional adapter: adds a small hidden layer for subgraph-specific adaptation.
    """
    def __init__(self, embedding_size, num_classes, use_adapter=False):
        super().__init__()
        if use_adapter:
            mid = max(embedding_size // 3, num_classes * 2)
            self.head = nn.Sequential(
                nn.Linear(embedding_size, mid),
                nn.ReLU(),
                nn.Linear(mid, num_classes)
            )
        else:
            self.head = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        return self.head(x)
