#!/usr/bin/env python3
"""
Code verification script: runs both PGNNCert and SECert on synthetic data
for 5+ epochs to verify correctness.
"""

import torch
import numpy as np
import os
import sys
import time

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def create_node_data():
    """Create synthetic node classification data."""
    num_nodes = 200
    num_features = 50
    num_classes = 5

    x = torch.randn(num_nodes, num_features)
    y = torch.randint(0, num_classes, (num_nodes,))

    # Random edges
    num_edges = 800
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    mask = src != dst
    src, dst = src[mask], dst[mask]
    edge_index = torch.stack([
        torch.cat([src, dst]),
        torch.cat([dst, src])
    ])

    # Masks
    perm = torch.randperm(num_nodes)
    n_train = int(num_nodes * 0.3)
    n_val = int(num_nodes * 0.1)
    n_test = int(num_nodes * 0.3)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val:n_train + n_val + n_test]] = True

    return x, y, edge_index, train_mask, val_mask, test_mask, num_features, num_classes


def create_graph_data():
    """Create synthetic graph classification data."""
    from torch_geometric.data import Data

    num_graphs = 50
    num_features = 7
    num_classes = 2

    graphs = []
    labels = []
    for i in range(num_graphs):
        n = np.random.randint(10, 30)
        x_g = torch.randn(n, num_features)
        y_g = torch.tensor(np.random.randint(0, num_classes))

        ne = n * 3
        src = torch.randint(0, n, (ne,))
        dst = torch.randint(0, n, (ne,))
        m = src != dst
        src, dst = src[m], dst[m]
        ei = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])

        data = Data(x=x_g, edge_index=ei, y=y_g)
        graphs.append(data)
        labels.append(y_g.item())

    # Use a list wrapper that supports boolean indexing like TUDataset
    class GraphList:
        def __init__(self, data_list):
            self.data_list = data_list
        def __len__(self):
            return len(self.data_list)
        def __getitem__(self, idx):
            if isinstance(idx, (torch.Tensor, np.ndarray)):
                if hasattr(idx, 'dtype') and idx.dtype == torch.bool:
                    idx = idx.nonzero(as_tuple=False).view(-1).tolist()
                elif hasattr(idx, 'dtype') and idx.dtype == np.bool_:
                    idx = np.where(idx)[0].tolist()
                else:
                    idx = idx.tolist() if hasattr(idx, 'tolist') else list(idx)
                return GraphList([self.data_list[i] for i in idx])
            elif isinstance(idx, list):
                return GraphList([self.data_list[i] for i in idx])
            elif isinstance(idx, slice):
                return GraphList(self.data_list[idx])
            return self.data_list[idx]
        def __iter__(self):
            return iter(self.data_list)

    graphs = GraphList(graphs)

    # Masks
    n_train = int(num_graphs * 0.5)
    n_val = int(num_graphs * 0.2)
    train_mask = torch.zeros(num_graphs, dtype=torch.bool)
    val_mask = torch.zeros(num_graphs, dtype=torch.bool)
    test_mask = torch.zeros(num_graphs, dtype=torch.bool)
    perm = torch.randperm(num_graphs)
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]] = True

    return graphs, labels, train_mask, val_mask, test_mask, num_features, num_classes


def test_pgnncert_e_node():
    """Test PGNNCert-E node classification."""
    print("\n" + "=" * 60)
    print("TEST 1: PGNNCert-E Node Classification (5 epochs)")
    print("=" * 60)

    from edge_hash import HashAgent, RobustNodeClassifier
    from utils import evaluate

    x, y, edge_index, train_mask, val_mask, test_mask, num_x, num_labels = create_node_data()

    T = 10
    hasher = HashAgent(h="md5", T=T)
    model = RobustNodeClassifier(hasher, edge_index, x, y,
                                 train_mask, val_mask, test_mask,
                                 num_x, num_labels, GNN="GCN")

    train_args = {
        "dataset": "synthetic",
        "paper": "GCN",
        "lr": 0.002,
        "epochs": 5,
        "clip_max": 2.0,
        "batch_size": 64,
        "early_stopping": 100,
        "seed": 42,
    }

    model.train(train_args)
    test_acc, M = model.test()
    print(f"\nPGNNCert-E Node Test Accuracy: {test_acc:.4f}")
    print(f"Certified margin distribution: min={M.min():.0f}, max={M.max():.0f}, mean={M.mean():.2f}")
    return True


def test_secert_e_node():
    """Test SECert-E node classification."""
    print("\n" + "=" * 60)
    print("TEST 2: SECert-E Node Classification (5 epochs)")
    print("=" * 60)

    from secert_edge_hash import HashAgent, SECertNodeClassifier
    from utils import evaluate

    x, y, edge_index, train_mask, val_mask, test_mask, num_x, num_labels = create_node_data()

    T = 10
    hasher = HashAgent(h="md5", T=T)
    model = SECertNodeClassifier(hasher, edge_index, x, y,
                                 train_mask, val_mask, test_mask,
                                 num_x, num_labels, GNN="GCN")

    train_args = {
        "dataset": "synthetic",
        "paper": "GCN",
        "lr": 0.002,
        "epochs": 5,
        "clip_max": 2.0,
        "batch_size": 64,
        "early_stopping": 100,
        "seed": 42,
    }

    model.train_model(train_args)
    test_acc, M = model.test()
    print(f"\nSECert-E Node Test Accuracy: {test_acc:.4f}")
    print(f"Certified margin distribution: min={M.min():.0f}, max={M.max():.0f}, mean={M.mean():.2f}")

    # Count parameters
    pgnncert_params = sum(p.numel() for p in model.parameters())
    print(f"SECert parameters: {pgnncert_params}")
    return True


def test_pgnncert_e_graph():
    """Test PGNNCert-E graph classification."""
    print("\n" + "=" * 60)
    print("TEST 3: PGNNCert-E Graph Classification (5 epochs)")
    print("=" * 60)

    from edge_hash import HashAgent, RobustGraphClassifier
    from utils import evaluate

    graphs, labels, train_mask, val_mask, test_mask, num_x, num_labels = create_graph_data()

    T = 10
    hasher = HashAgent(h="md5", T=T)
    model = RobustGraphClassifier(hasher, graphs, labels,
                                  train_mask, val_mask, test_mask,
                                  num_x, num_labels, GNN="GCN")

    train_args = {
        "dataset": "synthetic",
        "paper": "GCN",
        "lr": 0.002,
        "epochs": 5,
        "clip_max": 2.0,
        "batch_size": 64,
        "early_stopping": 100,
        "seed": 42,
    }

    model.train(train_args)
    test_acc, M = model.test()
    print(f"\nPGNNCert-E Graph Test Accuracy: {test_acc:.4f}")
    print(f"Certified margin distribution: min={M.min():.0f}, max={M.max():.0f}, mean={M.mean():.2f}")
    return True


def test_secert_e_graph():
    """Test SECert-E graph classification."""
    print("\n" + "=" * 60)
    print("TEST 4: SECert-E Graph Classification (5 epochs)")
    print("=" * 60)

    from secert_edge_hash import HashAgent, SECertGraphClassifier
    from utils import evaluate

    graphs, labels, train_mask, val_mask, test_mask, num_x, num_labels = create_graph_data()

    T = 10
    hasher = HashAgent(h="md5", T=T)
    model = SECertGraphClassifier(hasher, graphs, labels,
                                  train_mask, val_mask, test_mask,
                                  num_x, num_labels, GNN="GCN")

    train_args = {
        "dataset": "synthetic",
        "paper": "GCN",
        "lr": 0.002,
        "epochs": 5,
        "clip_max": 2.0,
        "batch_size": 64,
        "early_stopping": 100,
        "seed": 42,
    }

    model.train_model(train_args)
    test_acc, M = model.test()
    print(f"\nSECert-E Graph Test Accuracy: {test_acc:.4f}")
    print(f"Certified margin distribution: min={M.min():.0f}, max={M.max():.0f}, mean={M.mean():.2f}")
    return True


def test_secert_n_node():
    """Test SECert-N node classification."""
    print("\n" + "=" * 60)
    print("TEST 5: SECert-N Node Classification (5 epochs)")
    print("=" * 60)

    from secert_node_hash import HashAgent, SECertNodeClassifier
    from utils import evaluate

    x, y, edge_index, train_mask, val_mask, test_mask, num_x, num_labels = create_node_data()

    T = 10
    hasher = HashAgent(h="md5", T=T)
    model = SECertNodeClassifier(hasher, edge_index, x, y,
                                 train_mask, val_mask, test_mask,
                                 num_x, num_labels, GNN="GCN")

    train_args = {
        "dataset": "synthetic",
        "paper": "GCN",
        "lr": 0.002,
        "epochs": 5,
        "clip_max": 2.0,
        "batch_size": 64,
        "early_stopping": 100,
        "seed": 42,
    }

    model.train_model(train_args)
    test_acc, M = model.test()
    print(f"\nSECert-N Node Test Accuracy: {test_acc:.4f}")
    print(f"Certified margin distribution: min={M.min():.0f}, max={M.max():.0f}, mean={M.mean():.2f}")
    return True


def count_parameters():
    """Compare parameter counts between PGNNCert and SECert."""
    print("\n" + "=" * 60)
    print("PARAMETER COMPARISON")
    print("=" * 60)

    from edge_hash import HashAgent as EHashAgent
    from edge_hash import RobustNodeClassifier
    from secert_edge_hash import HashAgent as SEHashAgent
    from secert_edge_hash import SECertNodeClassifier

    x, y, edge_index, train_mask, val_mask, test_mask, num_x, num_labels = create_node_data()
    T = 50

    hasher1 = EHashAgent(h="md5", T=T)
    pgnn = RobustNodeClassifier(hasher1, edge_index, x, y,
                                train_mask, val_mask, test_mask,
                                num_x, num_labels, GNN="GCN")
    pgnn_params = sum(p.numel() for p in pgnn.parameters())

    hasher2 = SEHashAgent(h="md5", T=T)
    secert = SECertNodeClassifier(hasher2, edge_index, x, y,
                                  train_mask, val_mask, test_mask,
                                  num_x, num_labels, GNN="GCN")
    secert_params = sum(p.numel() for p in secert.parameters())

    print(f"T = {T} subgraphs")
    print(f"PGNNCert parameters: {pgnn_params:,}")
    print(f"SECert parameters:   {secert_params:,}")
    print(f"Parameter reduction: {pgnn_params / secert_params:.1f}x")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    passed = 0
    failed = 0
    tests = [
        ("PGNNCert-E Node", test_pgnncert_e_node),
        ("SECert-E Node", test_secert_e_node),
        ("PGNNCert-E Graph", test_pgnncert_e_graph),
        ("SECert-E Graph", test_secert_e_graph),
        ("SECert-N Node", test_secert_n_node),
    ]

    for name, test_fn in tests:
        try:
            t0 = time.time()
            result = test_fn()
            elapsed = time.time() - t0
            if result:
                print(f"\n  PASSED ({elapsed:.1f}s)")
                passed += 1
            else:
                print(f"\n  FAILED")
                failed += 1
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Parameter comparison
    try:
        count_parameters()
    except Exception as e:
        print(f"Parameter comparison failed: {e}")

    print("\n" + "=" * 60)
    print(f"VERIFICATION RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
