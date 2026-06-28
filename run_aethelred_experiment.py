# -*- coding: utf-8 -*-
"""
Project Aethelred — Experiment Runner

Replaces run_node_experiment.py and run_graph_experiment.py.
Supports both node and graph classification via --task flag.
"""

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from datasets.dataset_loader import load_node_data, load_graph_data
from aethelred_core import Aethelred
from aethelred_loss import compute_composite_loss
from aethelred_certify import certify_explanation_stability
from utils import evaluate, store_checkpoint


device = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================================================
# Environment generation (replaces static hashing from old paradigm)
# ======================================================================

def generate_environments_node(data, num_envs=5, edge_drop_rate=0.1):
    """
    Artificial environment generation for node-level tasks via edge dropping.
    The original graph is always included as the first environment.
    """
    environments = [data]
    for _ in range(num_envs - 1):
        env_data = data.clone()
        num_edges = env_data.edge_index.shape[1]
        keep_mask = torch.rand(num_edges, device=env_data.edge_index.device) > edge_drop_rate
        env_data.edge_index = env_data.edge_index[:, keep_mask]
        environments.append(env_data)
    return environments


def generate_environments_graph(graph, num_envs=5, edge_drop_rate=0.1):
    """
    Artificial environment generation for a single graph object.
    Returns a list of augmented copies.
    """
    environments = [graph]
    for _ in range(num_envs - 1):
        env = graph.clone()
        num_edges = env.edge_index.shape[1]
        keep_mask = torch.rand(num_edges, device=env.edge_index.device) > edge_drop_rate
        env.edge_index = env.edge_index[:, keep_mask]
        environments.append(env)
    return environments


# ======================================================================
# Training loops
# ======================================================================

def train_aethelred_node(model, data, train_args):
    """Training loop for node classification."""
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_args["lr"])

    best_val_acc = 0.0
    best_test_acc = 0.0
    best_epoch = 0

    for epoch in range(train_args["epochs"]):
        model.train()
        optimizer.zero_grad()

        # 1. Generate artificial environments
        environments = generate_environments_node(
            data, num_envs=train_args["num_envs"],
            edge_drop_rate=train_args.get("edge_drop_rate", 0.1),
        )

        losses_per_env = []
        primary_logits = None
        primary_mask = None

        # 2. Forward pass through each environment
        for i, env_data in enumerate(environments):
            env_data = env_data.to(device)
            logits, causal_mask = model(env_data)

            loss_env = F.cross_entropy(logits[env_data.train_mask], env_data.y[env_data.train_mask])
            losses_per_env.append(loss_env)

            if i == 0:
                primary_logits = logits
                primary_mask = causal_mask

        # 3. (Optional) IBP bounds for certification loss
        mask_low, mask_high = None, None
        if train_args["hparams"].get("epsilon", 0) > 0:
            eps = train_args.get("perturbation_budget", 0.1)
            x_low = data.x - eps
            x_high = data.x + eps
            try:
                mask_low, mask_high = model.causal_core.ibp_forward(x_low, x_high, data.edge_index)
            except Exception:
                pass  # gracefully skip if IBP fails

        # 4. Composite loss
        total_loss, loss_dict = compute_composite_loss(
            primary_logits, primary_mask, data,
            data.train_mask, losses_per_env,
            train_args["hparams"],
            mask_low=mask_low, mask_high=mask_high,
            task='node',
        )

        # 5. Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_args.get("clip_max", 2.0))
        optimizer.step()

        # 6. Evaluation on original graph
        model.eval()
        with torch.no_grad():
            logits, _ = model(data)
            train_acc = evaluate(logits[data.train_mask], data.y[data.train_mask])
            val_acc = evaluate(logits[data.val_mask], data.y[data.val_mask])
            test_acc = evaluate(logits[data.test_mask], data.y[data.test_mask])

        print(
            f"Epoch {epoch:3d} | Loss: {loss_dict['total']:.4f} "
            f"(task={loss_dict['task']:.4f} inv={loss_dict['invariance']:.4f} "
            f"sp={loss_dict['sparsity']:.4f} acyc={loss_dict['acyclicity']:.4f} "
            f"cert={loss_dict['certify']:.4f}) | "
            f"Train: {train_acc:.4f}  Val: {val_acc:.4f}  Test: {test_acc:.4f}"
        )

        # Use validation accuracy for early stopping, but track test accuracy for reporting
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc  # Store test acc at best validation epoch
            best_epoch = epoch
            store_checkpoint("aethelred_node", train_args["dataset"],
                             model, train_acc, val_acc, test_acc)

        if epoch - best_epoch > train_args.get("early_stopping", 100) and best_val_acc > 0.5:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"\nBest test acc: {best_test_acc:.4f} at epoch {best_epoch} (val: {best_val_acc:.4f})")
    return best_test_acc


def train_aethelred_graph(model, graphs, masks, labels, train_args):
    """Training loop for graph classification."""
    train_mask, val_mask, test_mask = masks
    optimizer = torch.optim.Adam(model.parameters(), lr=train_args["lr"])

    best_val_acc = 0.0
    best_test_acc = 0.0
    best_epoch = 0

    # Build data loaders
    train_graphs = [graphs[i] for i in range(len(graphs)) if train_mask[i]]
    val_graphs = [graphs[i] for i in range(len(graphs)) if val_mask[i]]
    test_graphs = [graphs[i] for i in range(len(graphs)) if test_mask[i]]

    train_loader = DataLoader(train_graphs, batch_size=train_args.get("batch_size", 64), shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=train_args.get("batch_size", 64), shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=train_args.get("batch_size", 64), shuffle=False)

    for epoch in range(train_args["epochs"]):
        model.train()
        total_loss_epoch = 0.0
        total_correct = 0
        total_samples = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Generate environments for each graph in batch
            envs = generate_environments_graph(batch, num_envs=train_args["num_envs"],
                                                edge_drop_rate=train_args.get("edge_drop_rate", 0.1))

            losses_per_env = []
            primary_logits = None
            primary_mask = None

            for i, env_data in enumerate(envs):
                env_data = env_data.to(device)
                logits, causal_mask = model(env_data)
                loss_env = F.cross_entropy(logits, env_data.y)
                losses_per_env.append(loss_env)
                if i == 0:
                    primary_logits = logits
                    primary_mask = causal_mask

            # Dummy train_mask for graph classification (all True)
            dummy_mask = torch.ones(primary_logits.size(0), dtype=torch.bool, device=device)

            total_loss, loss_dict = compute_composite_loss(
                primary_logits, primary_mask, batch,
                dummy_mask, losses_per_env,
                train_args["hparams"],
                task='graph',
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_args.get("clip_max", 2.0))
            optimizer.step()

            total_loss_epoch += loss_dict['total'] * primary_logits.size(0)
            total_correct += (primary_logits.argmax(dim=1) == batch.y).sum().item()
            total_samples += primary_logits.size(0)

        train_acc = total_correct / max(total_samples, 1)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits, _ = model(batch)
                val_correct += (logits.argmax(dim=1) == batch.y).sum().item()
                val_total += batch.y.size(0)
        val_acc = val_correct / max(val_total, 1)

        # Test
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                logits, _ = model(batch)
                test_correct += (logits.argmax(dim=1) == batch.y).sum().item()
                test_total += batch.y.size(0)
        test_acc = test_correct / max(test_total, 1)

        avg_loss = total_loss_epoch / max(total_samples, 1)
        print(
            f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
            f"Train: {train_acc:.4f}  Val: {val_acc:.4f}  Test: {test_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc  # Store test acc at best validation epoch
            best_epoch = epoch
            store_checkpoint("aethelred_graph", train_args["dataset"],
                             model, train_acc, val_acc, test_acc)

        if epoch - best_epoch > train_args.get("early_stopping", 100):
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"\nBest test acc: {best_test_acc:.4f} at epoch {best_epoch} (val: {best_val_acc:.4f})")
    return best_test_acc


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Project Aethelred — Experiment Runner")
    parser.add_argument("--dataset", type=str, default="CiteSeer",
                        help="Dataset name (node: Cora-ML, CiteSeer, PubMed, Amazon-C; "
                             "graph: AIDS, MUTAG, PROTEINS, DD)")
    parser.add_argument("--task", type=str, default="node", choices=["node", "graph"],
                        help="Task type: node or graph classification")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_causal", type=int, default=64,
                        help="Hidden dim for CausalDiscoveryCore")
    parser.add_argument("--hidden_focal", type=int, default=20,
                        help="Hidden dim for FocalEngine")
    parser.add_argument("--num_focal_layers", type=int, default=3)
    parser.add_argument("--num_envs", type=int, default=5,
                        help="Number of artificial environments")
    parser.add_argument("--edge_drop_rate", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for graph classification")
    parser.add_argument("--early_stopping", type=int, default=100)
    parser.add_argument("--clip_max", type=float, default=2.0)
    parser.add_argument("--perturbation_budget", type=float, default=0.1,
                        help="ε for IBP certification during training")

    # Composite loss hyperparameters
    parser.add_argument("--alpha", type=float, default=1.0, help="Invariance weight")
    parser.add_argument("--beta", type=float, default=0.01, help="IB weight")
    parser.add_argument("--gamma", type=float, default=0.1, help="Sparsity weight")
    parser.add_argument("--delta", type=float, default=1.0, help="Acyclicity weight")
    parser.add_argument("--epsilon", type=float, default=0.0,
                        help="Certification loss weight (0 = disabled)")

    # Certification
    parser.add_argument("--certify", action="store_true",
                        help="Run certification after training")
    parser.add_argument("--certify_budget", type=float, default=0.1,
                        help="Perturbation budget for post-hoc certification")

    args = parser.parse_args()

    hparams = {
        'alpha': args.alpha,
        'beta': args.beta,
        'gamma': args.gamma,
        'delta': args.delta,
        'epsilon': args.epsilon,
        'certify_top_k': 0.1,
        'certify_tau': 0.5,
    }

    train_args = {
        "dataset": args.dataset,
        "lr": args.lr,
        "epochs": args.epochs,
        "device": device,
        "num_envs": args.num_envs,
        "edge_drop_rate": args.edge_drop_rate,
        "batch_size": args.batch_size,
        "early_stopping": args.early_stopping,
        "clip_max": args.clip_max,
        "perturbation_budget": args.perturbation_budget,
        "hparams": hparams,
    }

    if args.task == "node":
        data, num_features, num_classes = load_node_data(args.dataset)
        print(f"[Aethelred-Node] Dataset={args.dataset}  Features={num_features}  Classes={num_classes}")

        model = Aethelred(
            num_features, num_classes,
            hidden_dim_causal=args.hidden_causal,
            hidden_dim_focal=args.hidden_focal,
            num_focal_layers=args.num_focal_layers,
            task='node',
        ).to(device)

        train_aethelred_node(model, data, train_args)

        if args.certify:
            print("\n--- Post-hoc Certification ---")
            data = data.to(device)
            certify_explanation_stability(model, data, perturbation_budget=args.certify_budget)

    elif args.task == "graph":
        graphs, num_features, num_classes, masks, labels = load_graph_data(args.dataset)
        print(f"[Aethelred-Graph] Dataset={args.dataset}  Features={num_features}  "
              f"Classes={num_classes}  Graphs={len(graphs)}")

        # Adapt hyperparameters for graph classification
        graph_hparams = dict(hparams)
        graph_hparams['gamma'] = min(hparams['gamma'], 0.05)  # less sparsity for small graphs
        graph_hparams['delta'] = 0.0  # disable acyclicity (too expensive per-graph)
        train_args["hparams"] = graph_hparams

        # Use wider focal hidden dim for graph classification
        focal_hidden = max(args.hidden_focal, 32)

        model = Aethelred(
            num_features, num_classes,
            hidden_dim_causal=args.hidden_causal,
            hidden_dim_focal=focal_hidden,
            num_focal_layers=args.num_focal_layers,
            task='graph',
        ).to(device)

        train_aethelred_graph(model, graphs, masks, labels, train_args)

        if args.certify:
            print("\n--- Post-hoc Certification (graph-level) ---")
            from torch_geometric.loader import DataLoader
            test_mask = masks[2]
            test_graphs = [graphs[i] for i in range(len(graphs)) if test_mask[i]]
            certified_count = 0
            total_count = 0
            model.eval()
            for g in test_graphs:
                g = g.to(device)
                is_cert = certify_explanation_stability(model, g,
                                                        perturbation_budget=args.certify_budget,
                                                        verbose=False)
                certified_count += int(is_cert)
                total_count += 1
            print(f"Certified: {certified_count}/{total_count} "
                  f"({100*certified_count/max(total_count,1):.1f}%)")


if __name__ == "__main__":
    main()
