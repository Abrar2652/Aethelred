# -*- coding: utf-8 -*-
"""
SPMotif and BA-2Motifs dataset generators.

SPMotif  — from "DIR: Discovering Invariant Rationales for GNNs" (Wu et al., ICLR 2022)
BA-2Motifs — from "Parameterized Explainer for Graph Neural Networks" (Luo et al., NeurIPS 2020)

SPMotif protocol (exact match to DIR spmotif_gen/spmotif.ipynb):
    Graph = base_graph (tree/ladder/wheel, role_id=0) + one causal motif (role_id>0)
    Bias  = P(base_type matches class); controls spurious structural correlation
    GT    = edges where BOTH endpoints have role_id > 0  (motif-internal only)
    Motifs: class0=dircycle, class1=house, class2=varcycle  (all 5 nodes)
    Bases:  class0↔tree, class1↔ladder, class2↔wheel

Each graph's Data object includes:
    data.ground_truth_mask : BoolTensor [num_edges] — True for causal motif edges
    data.y                 : LongTensor [1]         — class label
"""

import random as _random
import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Motif builders — exact structures from DIR spmotif_gen/synthetic_structsim.py
# Returns (undirected_edge_list, n_nodes, role_ids)
# role_ids[i] > 0 for all motif nodes → used for GT computation
# ---------------------------------------------------------------------------

def _dircycle(n_offset):
    """5-node directed-cycle motif (class 0). All role_ids > 0."""
    edges = [
        (n_offset,     n_offset + 1),
        (n_offset + 1, n_offset + 2),
        (n_offset + 3, n_offset + 2),
        (n_offset,     n_offset + 4),
        (n_offset + 4, n_offset + 3),
    ]
    role_ids = [1, 2, 3, 3, 2]   # col_start=1 (max backbone role=0, then +1)
    return edges, 5, role_ids


def _house(n_offset):
    """5-node house motif (class 1). All role_ids > 0."""
    edges = [
        (n_offset + 1, n_offset + 2),
        (n_offset + 4, n_offset + 3),
        (n_offset + 3, n_offset + 2),
        (n_offset + 4, n_offset + 1),
        (n_offset,     n_offset + 1),
        (n_offset,     n_offset + 4),
    ]
    role_ids = [1, 2, 3, 3, 2]
    return edges, 5, role_ids


def _varcycle(n_offset):
    """5-node varcycle/crane motif (class 2). All role_ids > 0."""
    edges = [
        (n_offset,     n_offset + 1),
        (n_offset + 1, n_offset + 2),
        (n_offset + 3, n_offset + 2),
        (n_offset,     n_offset + 4),
        (n_offset + 4, n_offset + 3),
        (n_offset,     n_offset + 2),
        (n_offset,     n_offset + 3),
    ]
    role_ids = [1, 2, 3, 3, 2]
    return edges, 5, role_ids


_MOTIF_FNS = {0: _dircycle, 1: _house, 2: _varcycle}


# ---------------------------------------------------------------------------
# Base graph builders — all role_ids = 0  (spurious background)
# Exact types from DIR: tree, ladder, wheel
# ---------------------------------------------------------------------------

def _build_tree(rng, height):
    """Balanced r-ary tree, r ~ randint(2,3), height ∈ {0,1,2}."""
    r = int(rng.randint(2, 4))   # numpy randint upper is exclusive → gives 2 or 3
    G = nx.balanced_tree(r, height)
    return G


def _build_ladder(rng, width):
    """nx.ladder_graph(width) → 2*width nodes."""
    return nx.ladder_graph(width)


def _build_wheel(rng, width):
    """nx.wheel_graph(width) → width nodes (1 hub + width-1 rim)."""
    return nx.wheel_graph(width)


_BASE_BUILDERS = {1: _build_tree, 2: _build_ladder, 3: _build_wheel}

# Class-to-base-type mapping (from DIR notebook):
#   class 0 → tree (base_num=1) with prob bias
#   class 1 → ladder (base_num=2) with prob bias
#   class 2 → wheel (base_num=3) with prob bias
_CLASS_PREFERRED_BASE = {0: 1, 1: 2, 2: 3}


def _sample_base_num(cls, bias, rng):
    """Choose base type for a given class using DIR's bias scheme."""
    probs = np.array([(1 - bias) / 2] * 3)
    probs[cls] = bias
    return int(rng.choice([1, 2, 3], p=probs))


def _sample_width(base_num, rng):
    """Width parameter for each base type (from DIR notebook)."""
    if base_num == 1:   # tree: height ∈ {0, 1, 2}
        return int(rng.choice(range(3)))
    elif base_num == 2: # ladder: width ∈ {8..11}
        return int(rng.choice(range(8, 12)))
    else:               # wheel: width ∈ {15..19}
        return int(rng.choice(range(15, 20)))


# ---------------------------------------------------------------------------
# SPMotif generator — exact DIR protocol
# ---------------------------------------------------------------------------

def generate_spmotif(
    n_graphs=3000,
    bias=0.33,
    n_base=None,      # unused — kept for API compatibility
    feat_dim=4,
    seed=42,
    random_features=False,
):
    """
    Generate the SPMotif benchmark dataset (exact DIR protocol).

    Matches DIR spmotif_gen/spmotif.ipynb:
      - 3 classes: 0=dircycle, 1=house, 2=varcycle
      - Base graph type (tree/ladder/wheel) is the spurious signal
      - Bias = P(base type matches class); bias=0.33 → balanced, bias=0.90 → high spurious
      - Ground truth = edges where both endpoints have role_id > 0 (motif-internal only)
      - random_features=True → x ~ Uniform[0,1]^4 for all nodes (DIR Tables 5/6)
      - random_features=False → class-encoded features on motif nodes (Tables 1/4)

    Parameters
    ----------
    n_graphs        : int   — total graphs (split equally across 3 classes)
    bias            : float — spurious correlation strength
    feat_dim        : int   — node feature dim (default 4, matches DIR)
    seed            : int   — master RNG seed
    random_features : bool  — True → Uniform[0,1] features (DIR protocol)
    """
    rng = np.random.RandomState(seed)
    graphs = []
    n_per_class = n_graphs // 3

    for cls in range(3):
        motif_fn = _MOTIF_FNS[cls]

        for gi in range(n_per_class):
            # ── Base graph (spurious) ─────────────────────────────────────
            base_num = _sample_base_num(cls, bias, rng)
            width    = _sample_width(base_num, rng)
            base_G   = _BASE_BUILDERS[base_num](rng, width)

            # Relabel base nodes to start from 0
            n_base_nodes = base_G.number_of_nodes()
            mapping = {old: new for new, old in enumerate(sorted(base_G.nodes()))}
            base_G  = nx.relabel_nodes(base_G, mapping)
            base_role_ids = [0] * n_base_nodes

            # ── Causal motif ──────────────────────────────────────────────
            motif_offset = n_base_nodes
            motif_edges_local, n_motif, motif_role_ids = motif_fn(motif_offset)

            # Bridge: motif node 0 (= motif_offset) connects to a random base node
            plugin = int(rng.choice(n_base_nodes))
            bridge_edges = [(motif_offset, plugin)]   # undirected → one edge

            # ── Assemble graph ────────────────────────────────────────────
            all_undirected = (list(base_G.edges())
                              + motif_edges_local
                              + bridge_edges)

            role_ids = base_role_ids + motif_role_ids
            n_total  = n_base_nodes + n_motif

            # Bidirectional edges (both directions for each undirected edge)
            bidi = []
            for u, v in all_undirected:
                bidi += [(u, v), (v, u)]

            src = torch.tensor([e[0] for e in bidi], dtype=torch.long)
            dst = torch.tensor([e[1] for e in bidi], dtype=torch.long)
            edge_index = torch.stack([src, dst], dim=0)

            # Ground truth: both endpoints have role_id > 0  (DIR: find_gd)
            role_arr = np.array(role_ids)
            gt_mask  = torch.tensor(
                (role_arr[src.numpy()] > 0) & (role_arr[dst.numpy()] > 0),
                dtype=torch.bool,
            )

            # ── Node features ─────────────────────────────────────────────
            if random_features:
                x = rng.uniform(0.0, 1.0, (n_total, feat_dim)).astype(np.float32)
            else:
                x = rng.randn(n_total, feat_dim).astype(np.float32) * 0.1
                motif_feat = np.zeros(feat_dim, dtype=np.float32)
                motif_feat[cls % feat_dim] = 1.0
                x[motif_offset : motif_offset + n_motif] += motif_feat

            data = Data(
                x=torch.from_numpy(x),
                edge_index=edge_index,
                y=torch.tensor([cls], dtype=torch.long),
                ground_truth_mask=gt_mask,
            )
            graphs.append(data)

    perm = rng.permutation(len(graphs))
    graphs = [graphs[i] for i in perm]
    return graphs


def split_spmotif(graphs, train_frac=0.8, val_frac=0.1, seed=42):
    """Split SPMotif graphs into train/val/test."""
    rng = np.random.RandomState(seed)
    n = len(graphs)
    idx = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    return (
        [graphs[i] for i in idx[:n_train]],
        [graphs[i] for i in idx[n_train: n_train + n_val]],
        [graphs[i] for i in idx[n_train + n_val:]],
    )


# ---------------------------------------------------------------------------
# BA-2Motifs generator
# ---------------------------------------------------------------------------

def _cycle_edges(n=5):
    edges = []
    for i in range(n):
        j = (i + 1) % n
        edges += [(i, j), (j, i)]
    return edges, n


def _house_edges():
    edges = [
        (0,1),(1,0),(0,3),(3,0),(1,2),(2,1),(2,3),(3,2),(0,4),(4,0),(1,4),(4,1),
    ]
    return edges, 5


def _ba_graph_edges(n_nodes, m=2, seed=0):
    rng = np.random.RandomState(seed)
    if n_nodes <= m:
        edges = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                edges += [(i, j), (j, i)]
        return edges
    edges = []
    degree = np.zeros(n_nodes, dtype=np.float64)
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            edges += [(i, j), (j, i)]
            degree[i] += 1
            degree[j] += 1
    for new_node in range(m + 1, n_nodes):
        probs = degree[:new_node] / degree[:new_node].sum()
        targets = rng.choice(new_node, size=m, replace=False, p=probs)
        for t in targets:
            edges += [(new_node, t), (t, new_node)]
            degree[new_node] += 1
            degree[t] += 1
    return edges


def generate_ba2motifs(n_graphs=1000, n_base=20, feat_dim=10, seed=42):
    """
    Generate BA-2Motifs dataset.
    Class 0: BA + house motif. Class 1: BA + 5-cycle motif.
    """
    rng = np.random.RandomState(seed)
    graphs = []
    n_per_class = n_graphs // 2
    class_motifs = {0: _house_edges, 1: _cycle_edges}

    for cls, motif_fn in class_motifs.items():
        motif_edges_local, n_motif = motif_fn()

        for gi in range(n_per_class):
            g_seed = seed * 100_000 + cls * 10_000 + gi
            ba_edges = _ba_graph_edges(n_base, m=2, seed=g_seed)
            motif_edges_global = [
                (u + n_base, v + n_base) for u, v in motif_edges_local
            ]
            bridge_backbone = int(rng.randint(0, n_base))
            bridge_edges = [
                (bridge_backbone, n_base), (n_base, bridge_backbone),
            ]
            all_edges = ba_edges + motif_edges_global + bridge_edges
            n_total = n_base + n_motif

            src = torch.tensor([e[0] for e in all_edges], dtype=torch.long)
            dst = torch.tensor([e[1] for e in all_edges], dtype=torch.long)
            edge_index = torch.stack([src, dst], dim=0)

            n_ba = len(ba_edges)
            gt_mask = torch.zeros(len(all_edges), dtype=torch.bool)
            gt_mask[n_ba: n_ba + len(motif_edges_global)] = True

            x = rng.randn(n_total, feat_dim).astype(np.float32) * 0.1
            motif_feat = np.zeros(feat_dim, dtype=np.float32)
            motif_feat[cls] = 2.0
            x[n_base:n_total] += motif_feat

            graphs.append(Data(
                x=torch.from_numpy(x),
                edge_index=edge_index,
                y=torch.tensor([cls], dtype=torch.long),
                ground_truth_mask=gt_mask.bool(),
            ))

    perm = rng.permutation(len(graphs))
    graphs = [graphs[i] for i in perm]
    return graphs


def split_ba2motifs(graphs, train_frac=0.8, val_frac=0.1, seed=42):
    """Split BA-2Motifs into train/val/test."""
    rng = np.random.RandomState(seed)
    n = len(graphs)
    idx = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    return (
        [graphs[i] for i in idx[:n_train]],
        [graphs[i] for i in idx[n_train: n_train + n_val]],
        [graphs[i] for i in idx[n_train + n_val:]],
    )
