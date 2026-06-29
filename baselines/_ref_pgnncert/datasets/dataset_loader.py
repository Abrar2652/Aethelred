# -*- coding: utf-8 -*-

import os
import warnings

import numpy as np
import scipy.sparse as sp
import torch
from numpy.random.mtrand import RandomState
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import TUDataset


DATASETS_DIR = os.path.abspath(os.path.dirname(__file__))
PAPER_NODE_DATASETS = ["Cora-ML", "CiteSeer", "PubMed", "Amazon-C"]
PAPER_GRAPH_DATASETS = ["AIDS", "MUTAG", "PROTEINS", "DD"]
PAPER_EXTRA_DATASETS = ["CIFAR10"]
ALL_PAPER_DATASETS = PAPER_NODE_DATASETS + PAPER_GRAPH_DATASETS + PAPER_EXTRA_DATASETS


def matri_to_index(A):
    V = A.shape[0]
    edge_index_0 = []
    edge_index_1 = []

    for i in range(V):
        for j in range(i, V):
            if A[i, j] == 1:
                edge_index_0.append(i)
                edge_index_1.append(j)
                if i != j:
                    edge_index_0.append(j)
                    edge_index_1.append(i)
    return np.array([edge_index_0, edge_index_1])


def matri_to_index_directed(A):
    V = A.shape[0]
    edge_index_0 = []
    edge_index_1 = []

    for i in range(V):
        for j in range(V):
            if A[i, j] == 1:
                edge_index_0.append(i)
                edge_index_1.append(j)
    return np.array([edge_index_0, edge_index_1])


class MaskableGraphList:
    """List-like graph container with boolean-mask indexing support."""

    def __init__(self, graphs):
        self.graphs = list(graphs)

    def __len__(self):
        return len(self.graphs)

    def __iter__(self):
        return iter(self.graphs)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return MaskableGraphList(self.graphs[idx])

        if torch.is_tensor(idx):
            if idx.dtype == torch.bool:
                idx = idx.nonzero(as_tuple=False).view(-1).tolist()
            else:
                idx = idx.view(-1).tolist()
            return MaskableGraphList([self.graphs[i] for i in idx])

        if isinstance(idx, np.ndarray):
            if idx.dtype == np.bool_:
                idx = np.where(idx)[0].tolist()
            else:
                idx = idx.tolist()
            return MaskableGraphList([self.graphs[i] for i in idx])

        if isinstance(idx, (list, tuple)):
            return MaskableGraphList([self.graphs[i] for i in idx])

        return self.graphs[idx]


def _normalize_node_dataset_name(name):
    aliases = {
        "cora-ml": "Cora-ML",
        "cora_ml": "Cora-ML",
        "citeseer": "CiteSeer",
        "pubmed": "PubMed",
        "pubmed ": "PubMed",
        "amazon-c": "computers",
        "amazonc": "computers",
        "computers": "computers",
    }
    key = name.strip()
    return aliases.get(key.lower(), key)


def _planetoid_root(name):
    return os.path.join(DATASETS_DIR, name)


def _amazon_root():
    return os.path.join(DATASETS_DIR, "Amazon-C")


def _cora_ml_dir():
    return os.path.join(DATASETS_DIR, "Cora-ML")


def _graph_dataset_candidates(name):
    key = name.strip()
    key_l = key.lower()

    if key_l in {"mutag", "mutagenicity"}:
        # MUTAG is the paper dataset name. Mutagenicity is supported as fallback alias.
        return ["MUTAG", "Mutagenicity"]
    if key_l in {"proteins", "protein"}:
        return ["PROTEINS"]
    if key_l == "aids":
        return ["AIDS"]
    if key_l == "dd":
        return ["DD"]
    if key_l == "cifar10":
        return ["CIFAR10"]
    if key_l == "mnist":
        return ["MNIST"]
    return [key]


def _ensure_graph_features(graphs):
    for g in graphs:
        if g.x is None:
            g.x = torch.ones((g.num_nodes, 1), dtype=torch.float32)
        else:
            g.x = g.x.to(torch.float32)
    return graphs


def _load_cora_ml(directed=False):
    candidate_paths = [
        os.path.join(_cora_ml_dir(), "cora_ml.npz"),
        os.path.join(DATASETS_DIR, "cora_ml.npz"),
        os.path.join(os.path.dirname(DATASETS_DIR), "cora_ml.npz"),
    ]
    data_name = None
    for p in candidate_paths:
        if os.path.exists(p):
            data_name = p
            break

    if data_name is None:
        warnings.warn(
            "cora_ml.npz not found. Falling back to Planetoid Cora "
            "(this does NOT exactly match paper's Cora-ML).",
            RuntimeWarning,
        )
        dataset = Planetoid(root=_planetoid_root("Cora"), name="Cora", num_train_per_class=50)
        data = dataset[0]
        data.train_mask = torch.zeros(data.y.size(), dtype=torch.bool)
        data.val_mask = torch.zeros(data.y.size(), dtype=torch.bool)
        data.test_mask = torch.zeros(data.y.size(), dtype=torch.bool)
        return data, dataset.num_classes

    with np.load(data_name, allow_pickle=True) as loader:
        loader = dict(loader)
        A = sp.csr_matrix(
            (loader["adj_data"], loader["adj_indices"], loader["adj_indptr"]),
            shape=loader["adj_shape"],
        )
        adj = A.toarray()
        X = sp.csr_matrix(
            (loader["attr_data"], loader["attr_indices"], loader["attr_indptr"]),
            shape=loader["attr_shape"],
        )
        x = X.toarray()
        y = loader.get("labels")
        if directed:
            edge_index = matri_to_index_directed(adj)
        else:
            edge_index = matri_to_index(adj)

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.int64),
        y=torch.tensor(y),
    )
    data.train_mask = torch.zeros(y.size, dtype=torch.bool)
    data.val_mask = torch.zeros(y.size, dtype=torch.bool)
    data.test_mask = torch.zeros(y.size, dtype=torch.bool)
    num_classes = len(np.unique(y))
    return data, num_classes


def _apply_node_splits(data, num_classes, train_frac, val_frac, test_frac):
    prng = RandomState(12)

    data.train_mask.fill_(False)
    data.val_mask.fill_(False)
    data.test_mask.fill_(False)

    for c in range(num_classes):
        idx = (data.y == c).nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue

        perm = torch.tensor(prng.permutation(idx.numel()), dtype=torch.long)
        idx = idx[perm]

        num_train = int(idx.shape[0] * train_frac)
        num_val = int(idx.shape[0] * val_frac)

        train_idx = idx[:num_train]
        val_idx = idx[num_train:num_train + num_val]

        if test_frac is None:
            test_idx = idx[num_train + num_val:]
        else:
            num_test = int(idx.shape[0] * test_frac)
            test_idx = idx[num_train + num_val:num_train + num_val + num_test]

        data.train_mask[train_idx] = True
        data.val_mask[val_idx] = True
        data.test_mask[test_idx] = True


def load_node_data(name, train_frac=0.3, val_frac=0.1, test_frac=0.3, directed=False):
    name = _normalize_node_dataset_name(name)

    if name in {"CiteSeer", "PubMed", "Cora"}:
        dataset = Planetoid(root=_planetoid_root(name), name=name, num_train_per_class=50)
        data = dataset[0]
        num_classes = dataset.num_classes

    elif name == "computers":
        dataset = Amazon(root=_amazon_root(), name=name)
        data = dataset[0]
        data.train_mask = torch.zeros(data.y.size(), dtype=torch.bool)
        data.val_mask = torch.zeros(data.y.size(), dtype=torch.bool)
        data.test_mask = torch.zeros(data.y.size(), dtype=torch.bool)
        num_classes = dataset.num_classes

    elif name == "Cora-ML":
        data, num_classes = _load_cora_ml(directed=directed)

    else:
        raise ValueError(
            f"Unsupported node dataset '{name}'. "
            f"Supported: Cora-ML, CiteSeer, PubMed, computers (Amazon-C alias)."
        )

    _apply_node_splits(
        data=data,
        num_classes=num_classes,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
    )

    if name == "NELL":
        data.x = data.x.to_dense()
    num_node_features = data.x.shape[1]
    return data, num_node_features, num_classes


def _load_tu_graph_data(name, train_frac, val_frac, test_frac):
    last_error = None
    dataset = None
    used_name = None

    for candidate in _graph_dataset_candidates(name):
        try:
            dataset = TUDataset(root=DATASETS_DIR, name=candidate, use_node_attr=True)
            used_name = candidate
            break
        except Exception as exc:
            last_error = exc

    if dataset is None:
        raise RuntimeError(
            f"Unable to load TU dataset for name '{name}'. Last error: {last_error}"
        )

    graphs = [dataset[i] for i in range(len(dataset))]
    _ensure_graph_features(graphs)
    graphs = MaskableGraphList(graphs)

    num_node_features = graphs[0].x.shape[1]
    ys = [graphs[i].y.item() for i in range(len(graphs))]
    num_classes = len(np.unique(ys))
    rng = np.random.RandomState(12)

    train_mask = torch.zeros(len(graphs), dtype=torch.bool)
    val_mask = torch.zeros(len(graphs), dtype=torch.bool)
    test_mask = torch.zeros(len(graphs), dtype=torch.bool)

    ys_t = torch.tensor(ys)
    for c in range(num_classes):
        idx = (ys_t == c).nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue

        perm = torch.tensor(rng.permutation(idx.numel()), dtype=torch.long)
        idx = idx[perm]

        num_train = int(idx.shape[0] * train_frac)
        num_val = int(idx.shape[0] * val_frac)

        train_idx = idx[:num_train]
        val_idx = idx[num_train:num_train + num_val]

        if test_frac is None:
            test_idx = idx[num_train + num_val:]
        else:
            num_test = int(idx.shape[0] * test_frac)
            test_idx = idx[num_train + num_val:num_train + num_val + num_test]

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

    if used_name != name:
        print(f"[dataset_loader] Requested '{name}', loaded '{used_name}'.")

    return graphs, num_node_features, num_classes, [train_mask, val_mask, test_mask], ys


def _load_gnn_benchmark_graph_data(name):
    train_set = GNNBenchmarkDataset(root=DATASETS_DIR, name=name, split="train")
    val_set = GNNBenchmarkDataset(root=DATASETS_DIR, name=name, split="val")
    test_set = GNNBenchmarkDataset(root=DATASETS_DIR, name=name, split="test")

    graphs = [g for g in train_set] + [g for g in val_set] + [g for g in test_set]
    _ensure_graph_features(graphs)
    graphs = MaskableGraphList(graphs)

    ys = [int(g.y.item()) for g in graphs]
    num_classes = len(np.unique(ys))
    num_node_features = graphs[0].x.shape[1]

    n_train = len(train_set)
    n_val = len(val_set)
    n_test = len(test_set)
    total = n_train + n_val + n_test

    train_mask = torch.zeros(total, dtype=torch.bool)
    val_mask = torch.zeros(total, dtype=torch.bool)
    test_mask = torch.zeros(total, dtype=torch.bool)

    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True

    return graphs, num_node_features, num_classes, [train_mask, val_mask, test_mask], ys


def load_graph_data(name, train_frac=0.5, val_frac=0.2, test_frac=None, directed=False):
    candidates = _graph_dataset_candidates(name)
    primary = candidates[0]

    if primary in {"CIFAR10", "MNIST"}:
        return _load_gnn_benchmark_graph_data(primary)

    return _load_tu_graph_data(
        name=primary,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
    )
