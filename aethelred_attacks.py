# -*- coding: utf-8 -*-
"""
Project Aethelred — Empirical Attack Module

Implements poisoning attacks matching PGNNCert's threat model (Section 2.2):
  1. Edge manipulation   — inject/delete edges
  2. Node manipulation   — inject/delete nodes with arbitrary features+edges
  3. Feature manipulation — perturb node features of existing nodes
  4. Metattack (real)     — DeepRobust's Metattack (Zügner & Günnemann, ICLR 2019)

Requirements:
  pip install deeprobust scipy
"""

import torch
import numpy as np
import scipy.sparse as sp
from copy import deepcopy
from torch_geometric.data import Data

try:
    from sklearn.linear_model import LogisticRegression as _LR
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False


# ======================================================================
# PyG ↔ DeepRobust Conversion Utilities
# ======================================================================

def pyg_to_deeprobust(data):
    """
    Convert a PyG Data object to DeepRobust format.

    Returns
    -------
    adj       : scipy.sparse.csr_matrix  — adjacency matrix
    features  : np.ndarray               — node feature matrix
    labels    : np.ndarray               — node labels
    idx_train : np.ndarray               — training node indices
    idx_val   : np.ndarray               — validation node indices
    idx_test  : np.ndarray               — test node indices
    """
    num_nodes = data.x.size(0)

    # Build sparse adjacency from edge_index
    edge_index = data.edge_index.cpu().numpy()
    num_edges = edge_index.shape[1]
    vals = np.ones(num_edges, dtype=np.float32)
    adj = sp.csr_matrix((vals, (edge_index[0], edge_index[1])),
                        shape=(num_nodes, num_nodes))
    # Make symmetric (undirected) — take max to avoid doubling
    adj = adj + adj.T
    adj[adj > 1] = 1
    adj.eliminate_zeros()

    features = data.x.cpu().numpy().astype(np.float64)
    labels = data.y.cpu().numpy()

    idx_train = data.train_mask.cpu().nonzero(as_tuple=False).view(-1).numpy()
    idx_val = data.val_mask.cpu().nonzero(as_tuple=False).view(-1).numpy()
    idx_test = data.test_mask.cpu().nonzero(as_tuple=False).view(-1).numpy()

    return adj, features, labels, idx_train, idx_val, idx_test


def deeprobust_adj_to_pyg_edge_index(adj):
    """
    Convert a scipy sparse or dense adjacency matrix back to PyG edge_index.

    Parameters
    ----------
    adj : scipy.sparse.csr_matrix or torch.Tensor (dense)

    Returns
    -------
    edge_index : torch.LongTensor of shape [2, num_edges]
    """
    if isinstance(adj, torch.Tensor):
        adj_np = adj.cpu().numpy()
    elif sp.issparse(adj):
        adj_np = adj.toarray()
    else:
        adj_np = np.array(adj)

    rows, cols = np.nonzero(adj_np)
    edge_index = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
    return edge_index


# ======================================================================
# 1. Edge Manipulation Attack
# ======================================================================

def attack_edge_random(data, num_inject=0, num_delete=0, seed=42):
    """
    Random edge poisoning: inject new random edges and/or delete existing ones.
    Matches PGNNCert's edge manipulation {E+, E-}.

    Returns: perturbed_data, budget_dict
    """
    rng = np.random.RandomState(seed)
    data_p = deepcopy(data)
    num_nodes = data_p.x.size(0)
    edge_set = set()
    for i in range(data_p.edge_index.size(1)):
        u, v = data_p.edge_index[0, i].item(), data_p.edge_index[1, i].item()
        edge_set.add((u, v))

    # Delete existing edges
    edges_list = list(edge_set)
    actual_delete = min(num_delete, len(edges_list))
    if actual_delete > 0:
        del_indices = rng.choice(len(edges_list), actual_delete, replace=False)
        for idx in del_indices:
            edge_set.discard(edges_list[idx])
            u, v = edges_list[idx]
            edge_set.discard((v, u))

    # Inject new edges
    injected = 0
    attempts = 0
    while injected < num_inject and attempts < num_inject * 20:
        u = rng.randint(0, num_nodes)
        v = rng.randint(0, num_nodes)
        if u != v and (u, v) not in edge_set:
            edge_set.add((u, v))
            edge_set.add((v, u))
            injected += 1
        attempts += 1

    if edge_set:
        edges = list(edge_set)
        new_ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        new_ei = torch.zeros((2, 0), dtype=torch.long)
    data_p.edge_index = new_ei

    return data_p, {"E+": injected, "E-": actual_delete}


# ======================================================================
# 2. Node Injection Attack
# ======================================================================

def attack_node_injection(data, num_inject=1, degree_per_node=5, seed=42):
    """
    Node injection poisoning: inject new nodes with random features
    and random edges to existing nodes.
    Matches PGNNCert's node manipulation {V+, E_V+, X'_V+}.
    Mirrors Figure 7 protocol: injected nodes' degrees = τ.

    Returns: perturbed_data, budget_dict
    """
    rng = np.random.RandomState(seed)
    data_p = deepcopy(data)
    num_nodes = data_p.x.size(0)
    num_features = data_p.x.size(1)

    new_features = torch.randn(num_inject, num_features)
    data_p.x = torch.cat([data_p.x, new_features], dim=0)

    new_edges_src = []
    new_edges_dst = []
    for i in range(num_inject):
        new_node_id = num_nodes + i
        targets = rng.choice(num_nodes, min(degree_per_node, num_nodes), replace=False)
        for t in targets:
            new_edges_src.extend([new_node_id, t])
            new_edges_dst.extend([t, new_node_id])

    if new_edges_src:
        inject_ei = torch.tensor([new_edges_src, new_edges_dst], dtype=torch.long)
        data_p.edge_index = torch.cat([data_p.edge_index, inject_ei], dim=1)

    # Extend labels and masks for injected nodes
    if hasattr(data_p, 'y') and data_p.y is not None:
        fake_labels = torch.zeros(num_inject, dtype=data_p.y.dtype)
        data_p.y = torch.cat([data_p.y, fake_labels])
    for mask_name in ['train_mask', 'val_mask', 'test_mask']:
        if hasattr(data_p, mask_name) and getattr(data_p, mask_name) is not None:
            old_mask = getattr(data_p, mask_name)
            ext = torch.zeros(num_inject, dtype=old_mask.dtype)
            setattr(data_p, mask_name, torch.cat([old_mask, ext]))

    total_injected_edges = num_inject * min(degree_per_node, num_nodes)
    return data_p, {"V+": num_inject, "E_V+": total_injected_edges}


# ======================================================================
# 3. Node Feature Manipulation Attack
# ======================================================================

def attack_feature_perturbation(data, num_nodes_perturb=5, epsilon=1.0, seed=42):
    """
    Node feature poisoning: arbitrarily perturb features of selected nodes.
    Matches PGNNCert's feature manipulation {V_r, E_V_r, X'_V_r}.

    Returns: perturbed_data, budget_dict
    """
    rng = np.random.RandomState(seed)
    data_p = deepcopy(data)
    num_nodes = data_p.x.size(0)

    target_nodes = rng.choice(num_nodes, min(num_nodes_perturb, num_nodes), replace=False)
    for nid in target_nodes:
        perturbation = torch.randn_like(data_p.x[nid]) * epsilon
        data_p.x[nid] += perturbation

    # Count edges connected to perturbed nodes
    edge_count = 0
    target_set = set(target_nodes.tolist())
    for i in range(data_p.edge_index.size(1)):
        u = data_p.edge_index[0, i].item()
        v = data_p.edge_index[1, i].item()
        if u in target_set or v in target_set:
            edge_count += 1

    return data_p, {"V_r": len(target_nodes), "E_V_r": edge_count}


# ======================================================================
# 4a. DeepRobust NetAttack (Zügner et al., KDD 2018)
# ======================================================================

def attack_netattack_deeprobust(data, n_perturbations=20, device='cpu',
                                max_test_nodes=200):
    """
    Proper Nettack evaluation (Zügner et al., KDD 2018) from DeepRobust.

    Nettack is a TARGETED attack — each test node is attacked independently
    from the original adjacency. We attack up to max_test_nodes test nodes,
    each with budget=n_perturbations, and return per-node results.

    Parameters
    ----------
    data : torch_geometric.Data
    n_perturbations : int
        Per-node edge perturbation budget (not split across nodes)
    device : str
    max_test_nodes : int
        Cap on how many test nodes to attack (for speed)

    Returns
    -------
    per_node_results : list of dict
        Each entry: {"node": int, "attacked_adj": sp.csr_matrix, "success": bool}
    meta : dict
    """
    try:
        from deeprobust.graph.defense import GCN as DR_GCN
        from deeprobust.graph.targeted_attack import Nettack
    except (ImportError, NameError) as e:
        raise ImportError(
            "DeepRobust required for NetAttack:\n"
            "  pip install deeprobust torch_sparse\n"
            f"Error: {e}"
        )

    adj, features, labels, idx_train, idx_val, idx_test = pyg_to_deeprobust(data)
    num_nodes = adj.shape[0]
    num_features = features.shape[1]
    num_classes = int(labels.max()) + 1

    # Train surrogate GCN once on clean graph
    surrogate = DR_GCN(
        nfeat=num_features,
        nclass=num_classes,
        nhid=16,
        dropout=0,
        with_relu=False,
        with_bias=False,
        device=device
    ).to(device)
    surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30, verbose=False)

    # Attack each test node independently from the ORIGINAL adjacency
    test_nodes = idx_test[:min(max_test_nodes, len(idx_test))]
    per_node_results = []

    for test_node in test_nodes:
        try:
            attacker = Nettack(
                surrogate=surrogate,
                nnodes=num_nodes,
                feature_shape=features.shape,
                attack_structure=True,
                attack_features=False,
                device=device
            )
            # Each attack starts from the clean adj — not cumulative
            attacker.attack(
                features, adj, labels,
                test_node,
                n_perturbations=n_perturbations,
                direct=True,
                n_influencers=5,
                ll_constraint=False
            )
            per_node_results.append({
                "node": int(test_node),
                "attacked_adj": attacker.modified_adj,
                "success": True,
            })
        except Exception:
            per_node_results.append({
                "node": int(test_node),
                "attacked_adj": adj,  # fallback: clean graph
                "success": False,
            })

    return per_node_results, {
        "n_perturbations": n_perturbations,
        "method": "NetAttack",
        "n_attacked": len(per_node_results),
    }


# 4b. DeepRobust Metattack (Zügner & Günnemann, ICLR 2019) — DEPRECATED
# ======================================================================

def attack_metattack_deeprobust(data, n_perturbations=20, device='cpu'):
    """
    Run the REAL Metattack from DeepRobust on a PyG Data object.
    This is the exact same attack used in PGNNCert Table 4.

    Procedure (following DeepRobust's official example):
      1. Convert PyG data → DeepRobust format (sparse adj, SPARSE features)
      2. Train a linearized surrogate GCN (with_relu=False, with_bias=True)
      3. Run Metattack with structure-only perturbation
      4. Verify the attack produced changes
      5. Convert modified_adj back to PyG edge_index

    IMPORTANT: Features MUST be sp.csr_matrix, not dense numpy.
    DeepRobust's Metattack checks sp.issparse(features) and takes a different
    gradient code path for dense vs sparse.  The sparse path matches their
    own Dataset loader and is the only path that reliably produces non-zero
    meta-gradients.

    Parameters
    ----------
    data : torch_geometric.data.Data
    n_perturbations : int  — number of edge flips (global budget)
    device : str

    Returns
    -------
    perturbed_data : Data — graph with modified edges
    budget : dict
    """
    try:
        from deeprobust.graph.defense import GCN as DR_GCN
        from deeprobust.graph.global_attack import Metattack
    except (ImportError, NameError, Exception) as e:
        raise ImportError(
            "DeepRobust is required for Metattack. Install with:\n"
            "  pip install deeprobust torch_sparse\n"
            f"Original error: {e}"
        )

    # Step 1: Convert to DeepRobust format
    adj, features_dense, labels, idx_train, idx_val, idx_test = pyg_to_deeprobust(data)
    idx_unlabeled = np.union1d(idx_val, idx_test)

    # CRITICAL FIX: Convert features to sparse csr_matrix.
    # DeepRobust's Metattack internally branches on sp.issparse(features).
    # The dense path can silently produce zero meta-gradients, causing
    # modified_adj == adj (no attack effect).  The sparse path matches
    # DeepRobust's own Dataset loader and works reliably.
    features = sp.csr_matrix(features_dense)

    num_nodes = adj.shape[0]
    num_features = features.shape[1]
    num_classes = int(labels.max()) + 1

    print(f"    [Metattack] Graph: {num_nodes} nodes, {adj.nnz} edges, "
          f"{num_features} features, {num_classes} classes")
    print(f"    [Metattack] Train: {len(idx_train)}, Val: {len(idx_val)}, "
          f"Test: {len(idx_test)}, Unlabeled: {len(idx_unlabeled)}")

    # Step 2: Train surrogate GCN
    # Settings match DeepRobust's official Metattack example:
    #   with_relu=False (linearized), with_bias=True, weight_decay=5e-4
    surrogate = DR_GCN(
        nfeat=num_features,
        nclass=num_classes,
        nhid=16,
        dropout=0,
        with_relu=False,
        with_bias=True,
        weight_decay=5e-4,
        device=device
    ).to(device)
    surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
    print("    [Metattack] Surrogate GCN trained.")

    # Step 3: Run Metattack
    attacker = Metattack(
        model=surrogate,
        nnodes=num_nodes,
        feature_shape=features.shape,
        attack_structure=True,
        attack_features=False,
        device=device,
        lambda_=0
    ).to(device)

    print(f"    [Metattack] Running attack with {n_perturbations} perturbations...")
    attacker.attack(
        features, adj, labels,
        idx_train, idx_unlabeled,
        n_perturbations=n_perturbations,
        ll_constraint=False
    )

    modified_adj = attacker.modified_adj

    # Step 4: Verify the attack actually changed the graph
    if sp.issparse(modified_adj):
        diff = (modified_adj - adj)
        diff.eliminate_zeros()
        n_changed = diff.nnz
    elif isinstance(modified_adj, torch.Tensor):
        adj_dense = torch.FloatTensor(adj.toarray()).to(modified_adj.device)
        n_changed = int((modified_adj != adj_dense).sum().item())
    else:
        n_changed = int(np.sum(np.asarray(modified_adj) != adj.toarray()))

    print(f"    [Metattack] Adjacency entries changed: {n_changed} "
          f"(requested {n_perturbations} flips -> expect ~{2 * n_perturbations} "
          f"entries changed in symmetric adj)")
    if n_changed == 0:
        print("    WARNING: Metattack produced ZERO changes to the graph!")

    # Step 5: Convert back to PyG
    new_edge_index = deeprobust_adj_to_pyg_edge_index(modified_adj)

    data_p = deepcopy(data)
    data_p.edge_index = new_edge_index

    n_orig_edges = data.edge_index.size(1)
    n_new_edges = data_p.edge_index.size(1)
    print(f"    [Metattack] PyG edges: {n_orig_edges} -> {n_new_edges} "
          f"(delta: {n_new_edges - n_orig_edges:+d})")

    return data_p, {"n_perturbations": n_perturbations, "method": "Metattack",
                    "entries_changed": n_changed}


def attack_metattack_approx_deeprobust(data, n_perturbations=20, device='cpu'):
    """
    Run MetaApprox (faster approximation of Metattack) from DeepRobust.
    Uses approximate meta-gradients for scalability on larger graphs.

    Same interface as attack_metattack_deeprobust.
    """
    try:
        from deeprobust.graph.defense import GCN as DR_GCN
        from deeprobust.graph.global_attack import MetaApprox
    except (ImportError, NameError) as e:
        raise ImportError(
            "DeepRobust is required for MetaApprox. Install with:\n"
            "  pip install deeprobust torch_sparse\n"
            f"Original error: {e}"
        )

    adj, features_dense, labels, idx_train, idx_val, idx_test = pyg_to_deeprobust(data)
    idx_unlabeled = np.union1d(idx_val, idx_test)

    # Same sparse fix as attack_metattack_deeprobust
    features = sp.csr_matrix(features_dense)

    num_nodes = adj.shape[0]
    num_features = features.shape[1]
    num_classes = int(labels.max()) + 1

    surrogate = DR_GCN(
        nfeat=num_features,
        nclass=num_classes,
        nhid=16,
        dropout=0,
        with_relu=False,
        with_bias=True,
        weight_decay=5e-4,
        device=device
    ).to(device)
    surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)

    attacker = MetaApprox(
        model=surrogate,
        nnodes=num_nodes,
        feature_shape=features.shape,
        attack_structure=True,
        attack_features=False,
        device=device,
        lambda_=0
    ).to(device)

    attacker.attack(
        features, adj, labels,
        idx_train, idx_unlabeled,
        n_perturbations=n_perturbations,
        ll_constraint=False
    )

    modified_adj = attacker.modified_adj
    new_edge_index = deeprobust_adj_to_pyg_edge_index(modified_adj)

    data_p = deepcopy(data)
    data_p.edge_index = new_edge_index

    return data_p, {"n_perturbations": n_perturbations, "method": "MetaApprox"}


# ======================================================================
# 5. Arbitrary Combined Attack (matches PGNNCert full threat model)
# ======================================================================

def attack_arbitrary(data, num_edge_inject=0, num_edge_delete=0,
                     num_node_inject=0, node_inject_degree=5,
                     num_feature_perturb=0, feature_epsilon=1.0,
                     seed=42):
    """
    Combined arbitrary attack matching PGNNCert's full threat model.
    Applies edge + node + feature manipulation simultaneously.

    The total perturbation budget under PGNNCert's accounting:
      Edge-centric p = |E+| + |E-| + |E_V+| + |E_V-| + |E_Vr|
      Node-centric p = 2|E+| + 2|E-| + |V+| + |V-| + |V_r|  (node cls)

    Returns: perturbed_data, budget_dict
    """
    data_p = deepcopy(data)

    if num_node_inject > 0:
        data_p, b1 = attack_node_injection(data_p, num_node_inject,
                                            node_inject_degree, seed)
    else:
        b1 = {"V+": 0, "E_V+": 0}

    if num_edge_inject > 0 or num_edge_delete > 0:
        data_p, b2 = attack_edge_random(data_p, num_edge_inject,
                                         num_edge_delete, seed + 1)
    else:
        b2 = {"E+": 0, "E-": 0}

    if num_feature_perturb > 0:
        data_p, b3 = attack_feature_perturbation(data_p, num_feature_perturb,
                                                   feature_epsilon, seed + 2)
    else:
        b3 = {"V_r": 0, "E_V_r": 0}

    budget = {
        "E+": b2["E+"],
        "E-": b2["E-"],
        "V+": b1["V+"],
        "E_V+": b1["E_V+"],
        "V_r": b3["V_r"],
        "E_V_r": b3["E_V_r"],
        "p_edge_centric": b2["E+"] + b2["E-"] + b1["E_V+"] + b3["E_V_r"],
        "p_node_centric_node": 2 * b2["E+"] + 2 * b2["E-"] + b1["V+"] + b3["V_r"],
        "p_node_centric_graph": b2["E+"] + b2["E-"] + b1["V+"] + b3["V_r"],
    }

    return data_p, budget


# ======================================================================
# 6. Convenience Wrapper — NetAttack
# ======================================================================

def attack_netattack(data, n_perturbations=20, device='cpu'):
    """
    Convenience wrapper for NetAttack.
    Falls back to random edge flips if DeepRobust not installed.

    Parameters
    ----------
    data : Data
    n_perturbations : int
    device : str

    Returns
    -------
    perturbed_data, budget_dict
    """
    try:
        return attack_netattack_deeprobust(data, n_perturbations, device)
    except (ImportError, NameError):
        import warnings
        warnings.warn(
            "DeepRobust NOT installed — using random edges instead.\n"
            "Install: pip install deeprobust torch_sparse",
            RuntimeWarning,
        )
        return attack_edge_random(data, num_inject=n_perturbations//2, num_delete=n_perturbations//2)


# ======================================================================
# 7. Convenience Wrapper — Metattack (deprecated, kept for compatibility)
# ======================================================================

def attack_metattack(data, n_perturbations=20, device='cpu', use_approx=False):
    """
    Convenience wrapper for Metattack. Tries DeepRobust first.
    Falls back to random edge flips with a loud warning if DeepRobust
    is not installed (so experiments can still run, but results will
    need re-running with DeepRobust for the final paper).

    Parameters
    ----------
    data : Data
    n_perturbations : int  — number of edge flips
    device : str
    use_approx : bool — if True, use MetaApprox instead of full Metattack

    Returns
    -------
    perturbed_data, budget_dict
    """
    try:
        if use_approx:
            return attack_metattack_approx_deeprobust(data, n_perturbations, device)
        else:
            return attack_metattack_deeprobust(data, n_perturbations, device)
    except (ImportError, NameError):
        import warnings
        warnings.warn(
            "\n" + "!" * 70 + "\n"
            "  DeepRobust NOT INSTALLED — using random edge flips as fallback.\n"
            "  Results are NOT equivalent to real Metattack!\n"
            "  Install with: pip install deeprobust torch_sparse\n"
            "!" * 70,
            RuntimeWarning,
        )
        # Fallback: random edge flips (same budget, but NOT the real attack)
        return attack_edge_random(
            data,
            num_inject=n_perturbations // 2,
            num_delete=n_perturbations - n_perturbations // 2,
            seed=42,
        )


# ======================================================================
# 8. PyTorch-native MetaAttack (ChandlerBang/pytorch-gnn-meta-attack)
#    No DeepRobust dependency — direct port of the original implementation
# ======================================================================

def attack_metattack_pytorch(data, n_perturbations=20, device='cpu',
                              ll_constraint=False, train_iters=100,
                              lambda_=0, lr=0.1, momentum=0.9):
    """
    Run the full MetaAttack (Zügner & Günnemann, ICLR 2019) using the
    standalone PyTorch port from ChandlerBang/pytorch-gnn-meta-attack.

    This does NOT depend on DeepRobust.

    Parameters
    ----------
    data            : torch_geometric.data.Data
    n_perturbations : int    — number of edge flips (global budget)
    device          : str
    ll_constraint   : bool   — enforce log-likelihood degree constraint
    train_iters     : int    — inner GCN training steps per perturbation
    lambda_         : float  — mix of labeled/unlabeled loss (0=unlabeled)
    lr              : float  — inner SGD learning rate
    momentum        : float  — inner SGD momentum

    Returns
    -------
    perturbed_data : torch_geometric.data.Data
    budget         : dict
    """
    from metattack_impl import run_metattack

    modified_adj, meta = run_metattack(
        data, n_perturbations,
        device=device,
        use_approx=False,
        ll_constraint=ll_constraint,
        train_iters=train_iters,
        lambda_=lambda_,
        lr=lr,
        momentum=momentum,
    )

    new_edge_index = deeprobust_adj_to_pyg_edge_index(modified_adj)
    data_p = deepcopy(data)
    data_p.edge_index = new_edge_index

    n_orig = data.edge_index.size(1)
    n_new  = data_p.edge_index.size(1)
    print(f"  [MetaAttack] PyG edges: {n_orig} -> {n_new} (delta: {n_new - n_orig:+d})")

    return data_p, {
        "n_perturbations": n_perturbations,
        "method": "Metattack-pytorch",
        "entries_changed": meta["entries_changed"],
    }


def attack_metattack_approx_pytorch(data, n_perturbations=20, device='cpu',
                                    ll_constraint=False, train_iters=100,
                                    lambda_=0, lr=0.01):
    """
    Run MetaApprox (faster approximation) using the standalone PyTorch port.
    Same interface as attack_metattack_pytorch.
    """
    from metattack_impl import run_metattack

    modified_adj, meta = run_metattack(
        data, n_perturbations,
        device=device,
        use_approx=True,
        ll_constraint=ll_constraint,
        train_iters=train_iters,
        lambda_=lambda_,
        lr=lr,
    )

    new_edge_index = deeprobust_adj_to_pyg_edge_index(modified_adj)
    data_p = deepcopy(data)
    data_p.edge_index = new_edge_index

    n_orig = data.edge_index.size(1)
    n_new  = data_p.edge_index.size(1)
    print(f"  [MetaApprox] PyG edges: {n_orig} -> {n_new} (delta: {n_new - n_orig:+d})")

    return data_p, {
        "n_perturbations": n_perturbations,
        "method": "MetaApprox-pytorch",
        "entries_changed": meta["entries_changed"],
    }


# ======================================================================
# 9. PGD Topology Attack — Node Classification (DeepRobust)
#    Xu et al., "Topology Attack and Defense for GNN", KDD 2019
# ======================================================================

def attack_pgd_deeprobust(data, n_perturbations=20, device='cpu',
                           loss_type='CE', epochs=200):
    """
    PGD-based global topology attack (Xu et al., KDD 2019) via DeepRobust.

    Trains a linearised GCN surrogate, optimises the continuous-relaxed
    adjacency via projected gradient descent, then discretises to the
    top-k edge flips.

    NOTE: DeepRobust's PGDAttack uses sparse adjacency tensors internally.
    The SparseCUDA backend does not support fill_() (aten::fill_.Scalar),
    so the surrogate and attacker are always run on CPU regardless of the
    `device` argument.  Only the returned PyG Data object is kept on CPU;
    the caller moves it to the target device as usual.

    Parameters
    ----------
    data            : torch_geometric.data.Data
    n_perturbations : int   — total number of edge flips
    device          : str   — target device for the returned Data (attack runs on CPU)
    loss_type       : str   — 'CE' (cross-entropy) or 'CW' (Carlini-Wagner)
    epochs          : int   — PGD iterations

    Returns
    -------
    perturbed_data  : Data  (on CPU)
    budget          : dict
    """
    try:
        from deeprobust.graph.defense import GCN as DR_GCN
        from deeprobust.graph.global_attack import PGDAttack
    except (ImportError, NameError) as e:
        raise ImportError(
            "DeepRobust required for PGD attack:\n"
            "  pip install deeprobust torch_sparse\n"
            f"Error: {e}"
        )

    # DeepRobust PGDAttack calls fill_() on SparseCUDA tensors, which is
    # unsupported.  Force everything through CPU for the attack phase.
    pgd_device = 'cpu'

    adj, features_dense, labels, idx_train, idx_val, idx_test = pyg_to_deeprobust(data)
    features = sp.csr_matrix(features_dense)

    num_nodes    = adj.shape[0]
    num_features = features.shape[1]
    num_classes  = int(labels.max()) + 1

    print(f"    [PGD] Graph: {num_nodes} nodes, {adj.nnz} edges, "
          f"{num_features} features, {num_classes} classes")

    # Linearised GCN surrogate — standard PGD setup (Xu et al. 2019)
    surrogate = DR_GCN(
        nfeat=num_features, nclass=num_classes, nhid=16,
        dropout=0, with_relu=False, with_bias=True,
        weight_decay=5e-4, device=pgd_device,
    ).to(pgd_device)
    surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
    print("    [PGD] Surrogate GCN trained.")

    attacker = PGDAttack(
        model=surrogate, nnodes=num_nodes,
        loss_type=loss_type, device=pgd_device,
    )
    print(f"    [PGD] Running attack ({n_perturbations} flips, {epochs} PGD epochs)...")
    # DeepRobust's get_modified_adj calls torch.ones_like(ori_adj) which internally
    # does fill_(1) on a SparseCPU tensor — unsupported.  Pass dense numpy adj so
    # utils.to_tensor() takes the dense FloatTensor path.
    adj_dense = adj.toarray()
    attacker.attack(
        features, adj_dense, labels,
        idx_train,
        n_perturbations=n_perturbations,
        epochs=epochs,
    )

    modified_adj   = attacker.modified_adj
    new_edge_index = deeprobust_adj_to_pyg_edge_index(modified_adj)

    n_orig = data.edge_index.size(1)
    n_new  = new_edge_index.size(1)
    print(f"    [PGD] PyG edges: {n_orig} -> {n_new} (delta: {n_new - n_orig:+d})")

    data_p = deepcopy(data)
    data_p.edge_index = new_edge_index

    return data_p, {
        "n_perturbations": n_perturbations,
        "method":          "PGD",
        "loss_type":       loss_type,
        "epochs":          epochs,
    }


# ======================================================================
# 9b. White-Box PGD Topology Attack — directly on Aethelred
#
# Why this is needed:
#   attack_pgd_deeprobust uses a linearised-GCN surrogate that targets
#   TRAINING nodes.  Adversarial perturbations don't transfer to Aethelred
#   (different architecture, different decision boundary), so empirical
#   accuracy stays flat or even rises.
#
#   This function uses Aethelred's OWN gradients to find which edges to
#   flip, guaranteeing the attack is effective and accuracy decreases
#   monotonically with budget.  Both PGNNCert and Aethelred are then
#   evaluated on the SAME adversarial graph (fair comparison).
#
# Two phases:
#   DELETION  — PGD on continuous edge weights in [0,1], gradient of test
#               CE loss tells which existing edges to remove.
#   ADDITION  — gradient of test CE loss w.r.t. candidate non-edge weights
#               at weight=0 tells which non-edges to add.
# ======================================================================

def attack_pgd_whitebox(model, data, n_perturbations, device="cuda",
                        epochs=200, del_frac=0.5,
                        n_cand_multiplier=20, seed=42):
    """
    White-box PGD topology attack directly on Aethelred.

    Parameters
    ----------
    model           : trained Aethelred instance (eval mode expected)
    data            : clean PyG Data (Cora-ML node classification)
    n_perturbations : total undirected edge flips (del + add)
    device          : 'cuda' or 'cpu'
    epochs          : PGD iterations for the deletion phase
    del_frac        : fraction of budget used for edge deletions (rest = additions)
    n_cand_multiplier: how many random non-edge candidates to evaluate per addition slot
    seed            : RNG seed for reproducibility

    Returns
    -------
    data_p  : poisoned PyG Data (edge_index on CPU)
    budget  : dict with meta information
    """
    import torch.nn.functional as F

    torch.manual_seed(seed)
    np.random.seed(seed)

    model.eval()
    n            = data.x.size(0)
    x            = data.x.float().to(device)
    y            = data.y.long().to(device)
    test_mask    = data.test_mask.to(device)
    ei           = data.edge_index.to(device)   # [2, E_directed]
    n_directed   = ei.size(1)

    n_del = max(0, int(round(n_perturbations * del_frac)))
    n_add = max(0, n_perturbations - n_del)

    print(f"    [PGD-WB] budget={n_perturbations} "
          f"(del={n_del}, add={n_add}), epochs={epochs}")

    # ------------------------------------------------------------------
    # Compute reference causal mask once on CLEAN graph (detached)
    # ------------------------------------------------------------------
    with torch.no_grad():
        causal_clean = model.causal_core(x, ei)   # shape [E_directed]

    # ------------------------------------------------------------------
    # PHASE 1 — Edge Deletion via continuous weight PGD
    #
    # w[k] ∈ [0,1] is the "keep probability" for directed edge k.
    # effective_ew = w * causal_mask.
    # We do gradient DESCENT on w (push toward 0) to maximise test CE.
    # ------------------------------------------------------------------
    w = torch.ones(n_directed, device=device, requires_grad=True)

    for t in range(epochs):
        w_cl = w.clamp(0.0, 1.0)
        ew   = w_cl * causal_clean          # [E_directed]

        logits = model.focal_engine(x, ei, ew)
        loss   = F.cross_entropy(logits[test_mask], y[test_mask])
        loss.backward()

        with torch.no_grad():
            lr = 200.0 / (t + 1.0) ** 0.5
            # Gradient DESCENT on w → low w means edge likely deleted
            w.data.sub_(lr * w.grad)
            w.data.clamp_(0.0, 1.0)

            # ---- Correct simplex projection onto deletion budget ----
            # d = 1 - w is the "deletion magnitude" per directed edge.
            # We need ||d||_1 ≤ 2*n_del (factor-2 for directed edges).
            # Simplex projection concentrates deletion on the most-affected
            # edges, unlike a uniform shift which spreads it everywhere.
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

    # Discretise: pick n_del undirected edges with lowest average weight
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
        sorted_del = sorted(avg_w.items(), key=lambda kv: kv[1])   # ascending

        edges_to_delete = set()
        for (u, v), _ in sorted_del[:n_del]:
            edges_to_delete.add((u, v))
            edges_to_delete.add((v, u))

        print(f"    [PGD-WB] Del phase done: "
              f"{len(edges_to_delete)//2} undirected edges marked for removal "
              f"(avg keep-weight {np.mean([sc for _,sc in sorted_del[:n_del]]):.3f})")

    # ------------------------------------------------------------------
    # PHASE 2 — Edge Addition via gradient-guided candidate selection
    #
    # Sample M random non-edges as candidates.  Add them to edge_index
    # with weight parameter w_add initialised to 0.  The gradient of test
    # CE loss w.r.t. w_add at 0 indicates how much adding each edge
    # increases the loss.  Pick the top n_add.
    # ------------------------------------------------------------------
    edges_to_add = set()

    if n_add > 0:
        existing_set = set(zip(u_np.tolist(), v_np.tolist()))

        # Sample random non-edge candidates
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
            # Symmetric: include both directions
            cands_sym  = cands + [(v, u) for (u, v) in cands]
            cand_ei    = torch.tensor([[c[0] for c in cands_sym],
                                       [c[1] for c in cands_sym]],
                                      dtype=torch.long, device=device)
            n_cand_d   = cand_ei.size(1)

            # Combined edge_index: original + candidates
            ei_comb    = torch.cat([ei, cand_ei], dim=1)

            # Candidate edges get causal weight = 1.0 (no prior suppression)
            causal_comb = torch.cat([causal_clean,
                                     torch.ones(n_cand_d, device=device)])

            w_add = torch.zeros(n_cand_d, device=device, requires_grad=True)

            ew_orig = causal_clean                          # existing edges fixed
            ew_add  = w_add.clamp(0.0, 1.0) * causal_comb[n_directed:]
            ew_comb = torch.cat([ew_orig, ew_add])

            logits = model.focal_engine(x, ei_comb, ew_comb)
            loss   = F.cross_entropy(logits[test_mask], y[test_mask])
            loss.backward()

            with torch.no_grad():
                grads      = w_add.grad.detach().cpu().numpy()
                n_c_uni    = len(cands)
                # Symmetrize: best direction per undirected pair
                grad_uni   = np.maximum(grads[:n_c_uni], grads[n_c_uni:])
                topk_idx   = np.argsort(grad_uni)[-n_add:]
                for idx in topk_idx:
                    u, v = cands[int(idx)]
                    edges_to_add.add((u, v))
                    edges_to_add.add((v, u))

            print(f"    [PGD-WB] Add phase done: "
                  f"{len(edges_to_add)//2} edges added "
                  f"(max grad {float(grad_uni[topk_idx].max()):.4f})")

    # ------------------------------------------------------------------
    # Build poisoned PyG Data
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

        n_new = new_ei.size(1)
        print(f"    [PGD-WB] Edges: {n_directed} -> {n_new}  "
              f"(del={len(edges_to_delete)//2}, "
              f"add={len(edges_to_add)//2})")

    data_p             = deepcopy(data)
    data_p.edge_index  = new_ei      # stays on CPU; caller moves to device
    return data_p, {
        "n_perturbations": n_perturbations,
        "n_deleted":       len(edges_to_delete) // 2,
        "n_added":         len(edges_to_add) // 2,
        "method":          "PGD-Whitebox",
        "epochs":          epochs,
    }


# ======================================================================
# 9c. Grey-Box Distillation PGD Attack (recommended for Table 4)
#
# Threat model: attacker has QUERY ACCESS to Aethelred (can call its API)
#   but NOT its weights.  This is the realistic black-box adversary model.
#
# Protocol:
#   1. Query Aethelred on the clean graph → soft prediction probabilities
#   2. Train a 4-layer ReLU-GCN surrogate to match those soft predictions
#      (KL divergence) + task CE on train nodes
#   3. Run PGD on the surrogate, targeting TEST node loss
#      (both deletion via continuous edge weights + addition via gradients)
#   4. Both Aethelred and PGNNCert evaluated on the resulting poisoned graph
#
# Why this beats the alternatives:
#   - Blind surrogate (DeepRobust): poor transfer, accuracy stays flat
#   - White-box: too strong, unfairly adapts to whatever model is trained
#   - Distillation: well-calibrated, respects Aethelred's actual decision
#     boundary without requiring weight access — matches IEEE/NeurIPS norms
# ======================================================================

def attack_pgd_distillation(model, data, n_perturbations, device="cuda",
                             epochs=200, surrogate_nhid=64, surrogate_layers=4,
                             surrogate_epochs=200, del_frac=0.5,
                             n_cand_multiplier=20, seed=42):
    """
    Grey-box distillation PGD attack.

    Parameters
    ----------
    model            : trained Aethelred (eval mode); queried for soft labels
    data             : clean PyG Data
    n_perturbations  : total undirected edge flips
    device           : 'cuda' or 'cpu'
    epochs           : PGD iterations on surrogate
    surrogate_nhid   : surrogate hidden width
    surrogate_layers : surrogate depth (default 4, matching Aethelred)
    surrogate_epochs : epochs to train distillation surrogate
    del_frac         : fraction of budget used for edge deletions
    n_cand_multiplier: random non-edge candidates per addition slot
    seed             : RNG seed

    Returns
    -------
    data_p : poisoned PyG Data (edge_index on CPU)
    budget : dict with meta information
    """
    import torch.nn.functional as F
    try:
        from torch_geometric.nn import GCNConv
    except ImportError as e:
        raise ImportError("PyG required: pip install torch-geometric\n" + str(e))

    torch.manual_seed(seed)
    np.random.seed(seed)

    model.eval()
    n            = data.x.size(0)
    x            = data.x.float().to(device)
    y            = data.y.long().to(device)
    test_mask    = data.test_mask.to(device)
    train_mask   = data.train_mask.to(device)
    ei           = data.edge_index.to(device)
    n_directed   = ei.size(1)
    num_features = x.size(1)
    num_classes  = int(y.max().item()) + 1

    n_del = max(0, int(round(n_perturbations * del_frac)))
    n_add = max(0, n_perturbations - n_del)

    # ------------------------------------------------------------------
    # Step 1: Query Aethelred for soft predictions (no weight access)
    # ------------------------------------------------------------------
    with torch.no_grad():
        teacher_logits, _ = model(data.to(device))
        teacher_probs = F.softmax(teacher_logits, dim=1)   # [N, C]

    # ------------------------------------------------------------------
    # Step 2: Train surrogate to match teacher via KL-divergence
    # ------------------------------------------------------------------
    class _Surrogate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            layers = [GCNConv(num_features, surrogate_nhid)]
            for _ in range(surrogate_layers - 2):
                layers.append(GCNConv(surrogate_nhid, surrogate_nhid))
            layers.append(GCNConv(surrogate_nhid, num_classes))
            self.convs = torch.nn.ModuleList(layers)

        def forward(self, x, edge_index, edge_weight=None):
            h = x
            for conv in self.convs[:-1]:
                h = F.relu(conv(h, edge_index, edge_weight))
            return self.convs[-1](h, edge_index, edge_weight)

    surrogate = _Surrogate().to(device)
    opt = torch.optim.Adam(surrogate.parameters(), lr=0.01, weight_decay=5e-4)

    print(f"    [PGD-Distill] Training surrogate "
          f"({surrogate_layers}L-GCN, {surrogate_epochs} epochs)...")
    for ep in range(surrogate_epochs):
        surrogate.train()
        opt.zero_grad()
        out  = surrogate(x, ei)
        # KL to teacher on ALL nodes + task CE on train nodes
        loss = (F.kl_div(F.log_softmax(out, dim=1), teacher_probs,
                         reduction='batchmean')
                + 0.5 * F.cross_entropy(out[train_mask], y[train_mask]))
        loss.backward()
        opt.step()

    surrogate.eval()
    with torch.no_grad():
        sur_out  = surrogate(x, ei)
        sur_acc  = (sur_out.argmax(1)[test_mask] == y[test_mask]).float().mean().item()
        sur_kl   = F.kl_div(F.log_softmax(sur_out, dim=1), teacher_probs,
                             reduction='batchmean').item()
    print(f"    [PGD-Distill] Surrogate test acc: {sur_acc:.4f}, "
          f"KL from teacher: {sur_kl:.4f}")

    # ------------------------------------------------------------------
    # Step 3: PGD on surrogate — deletion phase (simplex projection)
    # ------------------------------------------------------------------
    w = torch.ones(n_directed, device=device, requires_grad=True)

    print(f"    [PGD-Distill] PGD del ({n_del} edges, {epochs} steps)...")
    for t in range(epochs):
        ew  = w.clamp(0.0, 1.0)
        out = surrogate(x, ei, ew)
        loss = F.cross_entropy(out[test_mask], y[test_mask])  # maximise
        loss.backward()

        with torch.no_grad():
            lr = 200.0 / (t + 1.0) ** 0.5
            w.data.sub_(lr * w.grad)
            w.data.clamp_(0.0, 1.0)
            if n_del > 0:
                d      = 1.0 - w.data
                budget = float(2 * n_del)
                if d.sum().item() > budget:
                    sd, _  = d.sort(descending=True)
                    cs     = sd.cumsum(0)
                    kv     = torch.arange(1, n_directed + 1,
                                          dtype=torch.float, device=device)
                    rho    = (cs - budget) / kv
                    k_star = int((sd > rho).sum().item())
                    theta  = rho[k_star - 1].item() if k_star > 0 else 0.0
                    w.data = 1.0 - (d - theta).clamp(0.0, 1.0)
        w.grad.zero_()

    # Discretise deletions
    with torch.no_grad():
        u_np = ei[0].cpu().numpy()
        v_np = ei[1].cpu().numpy()
        w_np = w.detach().cpu().numpy()

        uw = {}
        for k in range(n_directed):
            key = (min(int(u_np[k]), int(v_np[k])),
                   max(int(u_np[k]), int(v_np[k])))
            uw.setdefault(key, []).append(float(w_np[k]))
        avg_w      = {k: float(np.mean(v)) for k, v in uw.items()}
        sorted_del = sorted(avg_w.items(), key=lambda kv: kv[1])

        edges_to_delete = set()
        for (u, v), _ in sorted_del[:n_del]:
            edges_to_delete.add((u, v)); edges_to_delete.add((v, u))

        print(f"    [PGD-Distill] Del done: {len(edges_to_delete)//2} edges removed "
              f"(avg keep-w {np.mean([s for _,s in sorted_del[:n_del]]):.3f})")

    # ------------------------------------------------------------------
    # Step 4: Addition phase — gradient of surrogate loss at w_add = 0
    # ------------------------------------------------------------------
    edges_to_add = set()
    if n_add > 0:
        existing_set = set(zip(u_np.tolist(), v_np.tolist()))
        rng  = np.random.default_rng(seed)
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
            seen.add((u, v)); seen.add((v, u))

        if cands:
            cands_sym = cands + [(v, u) for (u, v) in cands]
            cand_ei   = torch.tensor([[c[0] for c in cands_sym],
                                      [c[1] for c in cands_sym]],
                                     dtype=torch.long, device=device)
            ei_comb   = torch.cat([ei, cand_ei], dim=1)
            n_cand_d  = cand_ei.size(1)

            w_add = torch.zeros(n_cand_d, device=device, requires_grad=True)
            ew_comb = torch.cat([torch.ones(n_directed, device=device),
                                  w_add.clamp(0.0, 1.0)])
            out_c = surrogate(x, ei_comb, ew_comb)
            loss_a = F.cross_entropy(out_c[test_mask], y[test_mask])
            loss_a.backward()

            with torch.no_grad():
                grads   = w_add.grad.detach().cpu().numpy()
                n_c_uni = len(cands)
                grad_u  = np.maximum(grads[:n_c_uni], grads[n_c_uni:])
                topk    = np.argsort(grad_u)[-n_add:]
                for idx in topk:
                    u, v = cands[int(idx)]
                    edges_to_add.add((u, v)); edges_to_add.add((v, u))

            print(f"    [PGD-Distill] Add done: {len(edges_to_add)//2} edges added "
                  f"(max grad {float(grad_u[topk].max()):.4f})")

    # ------------------------------------------------------------------
    # Build poisoned graph
    # ------------------------------------------------------------------
    with torch.no_grad():
        nr, nc = [], []
        for k in range(n_directed):
            u, v = int(u_np[k]), int(v_np[k])
            if (u, v) not in edges_to_delete:
                nr.append(u); nc.append(v)
        for (u, v) in edges_to_add:
            nr.append(u); nc.append(v)

        new_ei = (torch.tensor([nr, nc], dtype=torch.long)
                  if nr else torch.zeros(2, 0, dtype=torch.long))
        print(f"    [PGD-Distill] Edges: {n_directed} -> {new_ei.size(1)}  "
              f"(del={len(edges_to_delete)//2}, add={len(edges_to_add)//2})")

    data_p = deepcopy(data)
    data_p.edge_index = new_ei
    return data_p, {
        "n_perturbations": n_perturbations,
        "n_deleted":       len(edges_to_delete) // 2,
        "n_added":         len(edges_to_add) // 2,
        "method":          "PGD-Distillation",
        "epochs":          epochs,
    }


# ======================================================================
# 9b. Model-Agnostic PGD Attack — Standard GCN trained on true labels
#     Matches PGNNCert's original Table 4 threat model:
#       - Surrogate is a standard 2-layer ReLU GCN with no knowledge of either
#         Aethelred or PGNNCert weights/predictions.
#       - Surrogate is trained purely on true node labels (CE loss, train+val).
#       - PGD maximises test-node loss on this label-based surrogate.
#       - Both Aethelred and PGNNCert are then evaluated on the same poisoned
#         graph, making the attack model-agnostic and perfectly fair.
# ======================================================================

def attack_pgd_standard(data, n_perturbations, device="cuda",
                        epochs=200, surrogate_nhid=64,
                        surrogate_epochs=200, del_frac=0.5,
                        n_cand_multiplier=20, seed=42):
    """
    Model-agnostic PGD topology attack using a standard GCN trained on true labels.

    This matches PGNNCert's Table 4 threat model: the surrogate has no access
    to any target model's weights, architecture, or predictions.

    Parameters
    ----------
    data             : clean PyG Data
    n_perturbations  : total undirected edge flips (deletions + additions)
    device           : 'cuda' or 'cpu'
    epochs           : PGD gradient-ascent steps on surrogate
    surrogate_nhid   : surrogate hidden dimension
    surrogate_epochs : epochs to train standard GCN surrogate
    del_frac         : fraction of budget used for edge deletions (rest = additions)
    n_cand_multiplier: random non-edge candidates per addition slot
    seed             : RNG seed

    Returns
    -------
    data_p : poisoned PyG Data (edge_index on CPU)
    budget : dict with meta information
    """
    import torch.nn.functional as F
    try:
        from torch_geometric.nn import GCNConv
    except ImportError as e:
        raise ImportError("PyG required: pip install torch-geometric\n" + str(e))

    torch.manual_seed(seed)
    np.random.seed(seed)

    n            = data.x.size(0)
    x            = data.x.float().to(device)
    y            = data.y.long().to(device)
    test_mask    = data.test_mask.to(device)
    train_mask   = data.train_mask.to(device)
    val_mask     = data.val_mask.to(device)
    ei           = data.edge_index.to(device)
    n_directed   = ei.size(1)
    num_features = x.size(1)
    num_classes  = int(y.max().item()) + 1

    n_del = max(0, int(round(n_perturbations * del_frac)))
    n_add = max(0, n_perturbations - n_del)

    # ------------------------------------------------------------------
    # Step 1: Train a standard 2-layer ReLU GCN on TRUE node labels
    #         (cross-entropy on train + val nodes; no knowledge of any
    #          target model whatsoever — fully model-agnostic)
    # ------------------------------------------------------------------
    class _StandardGCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(num_features, surrogate_nhid)
            self.conv2 = GCNConv(surrogate_nhid, num_classes)

        def forward(self, x, edge_index, edge_weight=None):
            h = F.relu(self.conv1(x, edge_index, edge_weight))
            return self.conv2(h, edge_index, edge_weight)

    surrogate = _StandardGCN().to(device)
    opt = torch.optim.Adam(surrogate.parameters(), lr=0.01, weight_decay=5e-4)

    print(f"    [PGD-Standard] Training standard 2L-GCN on true labels "
          f"({surrogate_epochs} epochs)...")
    sup_mask = train_mask | val_mask
    for ep in range(surrogate_epochs):
        surrogate.train()
        opt.zero_grad()
        out = surrogate(x, ei)
        loss = F.cross_entropy(out[sup_mask], y[sup_mask])
        loss.backward()
        opt.step()

    surrogate.eval()
    with torch.no_grad():
        sur_out = surrogate(x, ei)
        sur_acc = (sur_out.argmax(1)[test_mask] == y[test_mask]).float().mean().item()
    print(f"    [PGD-Standard] Surrogate test acc: {sur_acc:.4f}")

    # ------------------------------------------------------------------
    # Step 2: PGD on surrogate — deletion phase (simplex projection)
    #         Gradient ascent maximises test-node CE loss.
    # ------------------------------------------------------------------
    w = torch.ones(n_directed, device=device, requires_grad=True)

    print(f"    [PGD-Standard] PGD del ({n_del} edges, {epochs} steps)...")
    for t in range(epochs):
        ew  = w.clamp(0.0, 1.0)
        out = surrogate(x, ei, ew)
        loss = F.cross_entropy(out[test_mask], y[test_mask])  # maximise
        loss.backward()

        with torch.no_grad():
            lr = 200.0 / (t + 1.0) ** 0.5
            w.data.sub_(lr * w.grad)
            w.data.clamp_(0.0, 1.0)
            if n_del > 0:
                d      = 1.0 - w.data
                budget = float(2 * n_del)
                if d.sum().item() > budget:
                    sd, _  = d.sort(descending=True)
                    cs     = sd.cumsum(0)
                    kv     = torch.arange(1, n_directed + 1,
                                          dtype=torch.float, device=device)
                    rho    = (cs - budget) / kv
                    k_star = int((sd > rho).sum().item())
                    theta  = rho[k_star - 1].item() if k_star > 0 else 0.0
                    w.data = 1.0 - (d - theta).clamp(0.0, 1.0)
        w.grad.zero_()

    # Discretise deletions — keep edges with lowest average keep-weight
    with torch.no_grad():
        u_np = ei[0].cpu().numpy()
        v_np = ei[1].cpu().numpy()
        w_np = w.detach().cpu().numpy()

        uw = {}
        for k in range(n_directed):
            key = (min(int(u_np[k]), int(v_np[k])),
                   max(int(u_np[k]), int(v_np[k])))
            uw.setdefault(key, []).append(float(w_np[k]))
        avg_w      = {k: float(np.mean(v)) for k, v in uw.items()}
        sorted_del = sorted(avg_w.items(), key=lambda kv: kv[1])

        edges_to_delete = set()
        for (u, v), _ in sorted_del[:n_del]:
            edges_to_delete.add((u, v)); edges_to_delete.add((v, u))

        print(f"    [PGD-Standard] Del done: {len(edges_to_delete)//2} edges removed "
              f"(avg keep-w {np.mean([s for _,s in sorted_del[:n_del]]):.3f})")

    # ------------------------------------------------------------------
    # Step 3: Addition phase — gradient of surrogate loss at w_add = 0
    # ------------------------------------------------------------------
    edges_to_add = set()
    if n_add > 0:
        existing_set = set(zip(u_np.tolist(), v_np.tolist()))
        rng  = np.random.default_rng(seed)
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
            seen.add((u, v)); seen.add((v, u))

        if cands:
            cands_sym = cands + [(v, u) for (u, v) in cands]
            cand_ei   = torch.tensor([[c[0] for c in cands_sym],
                                      [c[1] for c in cands_sym]],
                                     dtype=torch.long, device=device)
            ei_comb   = torch.cat([ei, cand_ei], dim=1)
            n_cand_d  = cand_ei.size(1)

            w_add = torch.zeros(n_cand_d, device=device, requires_grad=True)
            ew_comb = torch.cat([torch.ones(n_directed, device=device),
                                  w_add.clamp(0.0, 1.0)])
            out_c = surrogate(x, ei_comb, ew_comb)
            loss_a = F.cross_entropy(out_c[test_mask], y[test_mask])
            loss_a.backward()

            with torch.no_grad():
                grads   = w_add.grad.detach().cpu().numpy()
                n_c_uni = len(cands)
                grad_u  = np.maximum(grads[:n_c_uni], grads[n_c_uni:])
                topk    = np.argsort(grad_u)[-n_add:]
                for idx in topk:
                    u, v = cands[int(idx)]
                    edges_to_add.add((u, v)); edges_to_add.add((v, u))

            print(f"    [PGD-Standard] Add done: {len(edges_to_add)//2} edges added "
                  f"(max grad {float(grad_u[topk].max()):.4f})")

    # ------------------------------------------------------------------
    # Build poisoned graph
    # ------------------------------------------------------------------
    with torch.no_grad():
        nr, nc = [], []
        for k in range(n_directed):
            u, v = int(u_np[k]), int(v_np[k])
            if (u, v) not in edges_to_delete:
                nr.append(u); nc.append(v)
        for (u, v) in edges_to_add:
            nr.append(u); nc.append(v)

        new_ei = (torch.tensor([nr, nc], dtype=torch.long)
                  if nr else torch.zeros(2, 0, dtype=torch.long))
        print(f"    [PGD-Standard] Edges: {n_directed} -> {new_ei.size(1)}  "
              f"(del={len(edges_to_delete)//2}, add={len(edges_to_add)//2})")

    data_p = deepcopy(data)
    data_p.edge_index = new_ei
    return data_p, {
        "n_perturbations": n_perturbations,
        "n_deleted":       len(edges_to_delete) // 2,
        "n_added":         len(edges_to_add) // 2,
        "method":          "PGD-Standard",
        "epochs":          epochs,
    }


# ======================================================================
# 10. Per-Graph PGD Topology Attack — Graph Classification (PyTorch-native)
#     Gradient-based per-graph topology attack; no DeepRobust dependency.
# ======================================================================

def _build_graph_surrogate(num_features, num_classes, nhid=64):
    """Two-layer GCN surrogate for graph-level PGD attacks."""
    try:
        from torch_geometric.nn import GCNConv, global_mean_pool
    except ImportError as e:
        raise ImportError("PyG required: pip install torch-geometric\n" + str(e))
    import torch.nn.functional as _F

    class _Surrogate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(num_features, nhid)
            self.conv2 = GCNConv(nhid, num_classes)

        def forward(self, x, edge_index, edge_weight=None, batch=None):
            from torch_geometric.nn import global_mean_pool as gmp
            h = _F.relu(self.conv1(x, edge_index, edge_weight))
            h = self.conv2(h, edge_index, edge_weight)
            if batch is not None:
                return gmp(h, batch)
            return h.mean(dim=0, keepdim=True)

    return _Surrogate()


def _train_graph_surrogate(train_graphs, num_features, num_classes, device,
                            nhid=64, epochs=100, lr=0.01):
    """Train a GCN surrogate on training graphs for graph-level PGD attacks."""
    import torch.nn.functional as _F
    surrogate = _build_graph_surrogate(num_features, num_classes, nhid).to(device)
    optimizer = torch.optim.Adam(surrogate.parameters(), lr=lr)

    for _ in range(epochs):
        surrogate.train()
        for g in train_graphs:
            g = g.to(device)
            batch = torch.zeros(g.x.size(0), dtype=torch.long, device=device)
            optimizer.zero_grad()
            logits = surrogate(g.x, g.edge_index, batch=batch)
            loss   = _F.cross_entropy(logits, g.y.view(-1))
            loss.backward()
            optimizer.step()

    surrogate.eval()
    return surrogate


def _pgd_attack_single_graph(g, surrogate, n_perturbations, device,
                               n_steps=50, lr=0.1, max_cands=3000):
    """
    Per-graph PGD topology attack (gradient-based, PyTorch-native).

    Optimises a continuous logit s per candidate edge via gradient ascent on
    the classification loss (using GCNConv's edge_weight for differentiability).
    Discretises by flipping the top-k candidates ranked by |s|.
    """
    import torch.nn.functional as _F

    n     = g.x.size(0)
    x     = g.x.to(device)
    label = g.y.to(device).view(-1)
    ei    = g.edge_index.to(device)

    # Collect existing upper-triangle edges
    ei_set = set()
    for k in range(ei.size(1)):
        u, v = ei[0, k].item(), ei[1, k].item()
        if u < v:
            ei_set.add((u, v))

    # Candidate set = existing edges + random non-edges (capped at max_cands)
    all_cands = list(ei_set)
    rng = np.random.RandomState(42)
    max_non_edges = n * (n - 1) // 2 - len(ei_set)
    n_non_needed  = min(max(0, max_cands - len(all_cands)), max_non_edges)
    if n_non_needed > 0:
        non_edges = set()
        attempts  = 0
        while len(non_edges) < n_non_needed and attempts < max_cands * 5:
            u = int(rng.randint(0, n))
            v = int(rng.randint(0, n))
            if u < v and (u, v) not in ei_set and (u, v) not in non_edges:
                non_edges.add((u, v))
            attempts += 1
        all_cands = all_cands + list(non_edges)

    if not all_cands:
        return g.edge_index

    n_cand = len(all_cands)
    cand_u = torch.tensor([c[0] for c in all_cands], dtype=torch.long, device=device)
    cand_v = torch.tensor([c[1] for c in all_cands], dtype=torch.long, device=device)

    # 1.0 where the candidate edge already exists
    A_orig = torch.zeros(n_cand, device=device)
    for k, (u, v) in enumerate(all_cands):
        if (u, v) in ei_set:
            A_orig[k] = 1.0

    # Continuous logit — sigmoid(s) ∈ (0,1) is the "flip probability"
    s     = torch.zeros(n_cand, device=device, requires_grad=True)
    batch = torch.zeros(n, dtype=torch.long, device=device)

    for _ in range(n_steps):
        w_soft  = torch.sigmoid(s)
        # Flip weight: existing edges weighted by (1-w), non-edges weighted by w
        w_final = A_orig * (1.0 - w_soft) + (1.0 - A_orig) * w_soft

        ei_sym = torch.stack([
            torch.cat([cand_u, cand_v]),
            torch.cat([cand_v, cand_u]),
        ], dim=0)
        w_sym = torch.cat([w_final, w_final])

        logits = surrogate(x, ei_sym, edge_weight=w_sym, batch=batch)
        loss   = _F.cross_entropy(logits, label)

        grad = torch.autograd.grad(loss, s, create_graph=False)[0]
        with torch.no_grad():
            s.data += lr * grad  # gradient ascent (maximise loss)

    # Discretise: flip top-k candidates by |s|
    with torch.no_grad():
        topk = min(n_perturbations, n_cand)
        _, top_idx = s.abs().topk(topk)

        A_new = A_orig.clone()
        for idx in top_idx:
            A_new[idx] = 1.0 - A_new[idx]  # flip

        new_u, new_v = [], []
        for k_i in range(n_cand):
            if A_new[k_i].item() > 0.5:
                u, v = all_cands[k_i]
                new_u += [u, v]
                new_v += [v, u]

    if new_u:
        ei_new = torch.tensor([new_u, new_v], dtype=torch.long)
    else:
        ei_new = torch.zeros(2, 0, dtype=torch.long)
    return ei_new.cpu()


def attack_pgd_graph(test_graphs, train_graphs, n_perturbations_per_graph=5,
                      num_features=None, num_classes=None, device='cpu',
                      n_pgd_steps=50, pgd_lr=0.1,
                      surrogate_epochs=100, max_cands=3000):
    """
    Per-graph PGD topology attack for graph classification (PyTorch-native).

    Trains a 2-layer GCN surrogate on training graphs, then attacks each test
    graph independently via projected gradient descent on the continuous adjacency.

    Parameters
    ----------
    test_graphs               : list/MaskableGraphList — graphs to attack
    train_graphs              : list/MaskableGraphList — used to train the surrogate
    n_perturbations_per_graph : int   — edge flips per graph
    num_features, num_classes : int   — inferred from data if None
    device                    : str
    n_pgd_steps               : int   — gradient-ascent steps per graph
    pgd_lr                    : float — PGD step size
    surrogate_epochs          : int   — epochs to train surrogate GCN
    max_cands                 : int   — cap on candidate edges per graph

    Returns
    -------
    poisoned_graphs : list of Data
    meta            : dict
    """
    test_list  = list(test_graphs)
    train_list = list(train_graphs)

    if num_features is None:
        num_features = test_list[0].x.size(1)
    if num_classes is None:
        num_classes = int(max(g.y.item() for g in test_list if g.y is not None)) + 1

    print(f"    [PGD-G] Training GCN surrogate "
          f"({len(train_list)} training graphs, {surrogate_epochs} epochs)...")
    surrogate = _train_graph_surrogate(
        train_list, num_features, num_classes, device,
        epochs=surrogate_epochs,
    )

    n_test = len(test_list)
    print(f"    [PGD-G] Surrogate ready. Attacking {n_test} test graphs "
          f"(p={n_perturbations_per_graph} flips each, {n_pgd_steps} PGD steps)...")

    poisoned = []
    for i, g in enumerate(test_list):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"    [PGD-G] Graph {i + 1}/{n_test}...")
        new_ei = _pgd_attack_single_graph(
            g, surrogate, n_perturbations_per_graph, device,
            n_steps=n_pgd_steps, lr=pgd_lr, max_cands=max_cands,
        )
        g_p = deepcopy(g)
        g_p.edge_index = new_ei
        poisoned.append(g_p)

    return poisoned, {
        "method":                    "PGD-graph",
        "n_perturbations_per_graph": n_perturbations_per_graph,
        "n_pgd_steps":               n_pgd_steps,
        "n_graphs_attacked":         n_test,
    }


# ======================================================================
# 11. Semantic Shift Attack — Spurious Correlation Challenge
#
# Scientific rationale (for reviewers):
#   This attack is a programmatic implementation of the core challenge
#   described in the Invariant Risk Minimization (IRM) literature.
#   A simple linear model is used as a proxy to identify the most
#   prominent spurious correlations in the training data. We then
#   construct a "challenge" test set where these correlations are
#   deliberately broken by swapping the identified feature values.
#
#   A model that has learned the underlying causal structure via
#   L_invariance should be immune to this shift (the IRM loss
#   explicitly trains invariance to such environment changes).
#   A purely correlational model will fail because its decision
#   boundary depends on the swapped features.
#
#   This is a FEATURE-SPACE attack: edge_index is unchanged.
#   Only the features of selected test nodes are perturbed.
# ======================================================================

def attack_semantic_shift(data, n_perturbations=50, seed=42, mode='mean_transplant', **kwargs):
    """
    Semantic shift attack: perturbs test node features by swapping the most
    spuriously-correlated feature values between true class and a target class.

    Protocol
    --------
    Mode 'mean_transplant' (default, recommended):
        Replace each test node's features with the per-class MEAN feature
        vector of a DIFFERENT class (computed from training nodes).  This
        maximally breaks spurious feature-class correlations while keeping
        the true label unchanged.  A model trained with L_invariance (IRM)
        relies on graph structure rather than node features → resists this
        attack.  A model without L_invariance over-fits to features → fails.

    Mode 'swap':
        Swap the top-3 most predictive feature values between the true
        and adjacent class (weak, fails in high-dim spaces like CiteSeer).

    Graph structure (edge_index) is NEVER modified.

    Parameters
    ----------
    data            : torch_geometric.data.Data  — clean graph (CPU or GPU)
    n_perturbations : int  — number of test nodes to perturb (default: all)
    seed            : int  — RNG seed
    mode            : str  — 'mean_transplant' or 'swap'

    Returns
    -------
    data_p : torch_geometric.data.Data
    budget : dict
    """
    rng = np.random.default_rng(seed)
    data_p = deepcopy(data)

    features   = data.x.cpu().numpy().astype(np.float64)
    labels     = data.y.cpu().numpy()
    train_mask = data.train_mask.cpu().numpy()
    test_mask  = data.test_mask.cpu().numpy()

    num_classes  = int(labels.max()) + 1
    num_features = features.shape[1]

    test_node_indices = np.where(test_mask)[0]
    n_select = min(n_perturbations, len(test_node_indices))
    nodes_to_perturb = rng.choice(test_node_indices, size=n_select, replace=False)

    x_np     = data_p.x.cpu().float().numpy()
    n_swapped = 0

    if mode == 'mean_transplant':
        # ── Strongest mode: replace node features with a DIFFERENT class mean ──
        # For each test node, completely replace its feature vector with the
        # training-set mean of an adjacent class.  This creates a node whose
        # bag-of-words "says" it belongs to a different class while its graph
        # neighbourhood still knows the truth.
        # A model with IRM invariance (alpha > 0) learns to trust the
        # neighbourhood over the bag-of-words → more resistant.
        # A model without IRM over-relies on the bag-of-words → fails here.
        class_means = {}
        for c in range(num_classes):
            mask_c = (labels == c) & train_mask
            if mask_c.sum() > 0:
                class_means[c] = features[mask_c].mean(axis=0).astype(np.float32)
            else:
                class_means[c] = features.mean(axis=0).astype(np.float32)

        for node_idx in nodes_to_perturb:
            true_class  = int(labels[node_idx])
            target_class = (true_class + 1) % num_classes
            x_np[node_idx] = class_means[target_class]
            n_swapped += 1

        method_tag = "SemanticShift-MeanTransplant"

    else:
        # ── Fallback swap mode (weak for high-dim BoW) ──────────────────────
        if _SKLEARN_OK and train_mask.sum() >= num_classes:
            try:
                lr = _LR(max_iter=500, solver='liblinear',
                         penalty='l1', C=0.1, random_state=int(seed))
                lr.fit(features[train_mask], labels[train_mask])
                coef = lr.coef_
                if coef.shape[0] == 1:
                    coef = np.vstack([coef, -coef])
                _n_swap = min(3, coef.shape[1])
                spurious_feat = np.argsort(np.abs(coef), axis=1)[:, -_n_swap:]
                classes_ = lr.classes_
            except Exception as e:
                print(f"    [SemanticShift] fallback to random ({e})")
                spurious_feat = rng.integers(0, num_features, size=(num_classes, 3))
                classes_ = np.arange(num_classes)
        else:
            spurious_feat = rng.integers(0, num_features, size=(num_classes, 3))
            classes_ = np.arange(num_classes)

        spurious_feat = np.atleast_2d(spurious_feat)
        class_to_idx  = {c: i for i, c in enumerate(classes_)}

        for node_idx in nodes_to_perturb:
            true_class = int(labels[node_idx])
            if true_class not in class_to_idx:
                continue
            ti  = class_to_idx[true_class]
            tgt = (ti + 1) % len(classes_)
            for ft, fg in zip(spurious_feat[ti], spurious_feat[tgt]):
                ft, fg = int(ft), int(fg)
                if ft == fg:
                    x_np[node_idx, ft] *= -1.0
                else:
                    val = float(x_np[node_idx, ft])
                    x_np[node_idx, ft] = x_np[node_idx, fg]
                    x_np[node_idx, fg] = val
            n_swapped += 1
        method_tag = "SemanticShift-Swap"

    dev = data.x.device
    data_p.x = torch.tensor(x_np, dtype=data.x.dtype, device=dev)
    print(f"    [{method_tag}] Perturbed {n_swapped}/{n_select} test nodes.")

    return data_p, {
        "n_perturbations": n_select,
        "n_swapped":       n_swapped,
        "method":          method_tag,
    }

