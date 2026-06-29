# -*- coding: utf-8 -*-
"""
Unified Graph Classification Experiment Runner.
Runs both PGNNCert and SECert on the same data with identical protocol.

Usage:
  python run_graph_experiment.py --dataset AIDS --method both --variant E --gnn GCN --T 50
"""

import argparse
import torch
import numpy as np
import os
import sys
import time
import json

# Always resolve imports and dataset paths relative to THIS file's directory,
# regardless of where the script is invoked from.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.chdir(_HERE)

from datasets.dataset_loader import load_graph_data
from normal_baselines import run_normal_graph
from utils import evaluate


def run_pgnncert_e_graph(dataset, gnn, T, train_args, retrain=False):
    """Run PGNNCert-E graph classification."""
    from edge_hash import HashAgent, RobustGraphClassifier

    graphs, num_x, num_labels, mask_split, labels = load_graph_data(dataset)
    train_mask, val_mask, test_mask = mask_split

    hasher = HashAgent(h="md5", T=T)
    r_model = RobustGraphClassifier(hasher, graphs, labels,
                                    train_mask, val_mask, test_mask,
                                    num_x, num_labels, GNN=gnn)

    path = f"./checkpoints/robust_e/{gnn}/{dataset}/{T}/best_model"
    if (not retrain) and os.path.exists(path + "_0"):
        r_model.load_model(path)
    else:
        r_model.train(train_args)
        r_model.load_model(path)

    labels_t = torch.as_tensor(labels)
    out_test, M = r_model.vote(test_mask)
    test_labels = labels_t[test_mask]
    test_acc = evaluate(out_test, test_labels)

    test_preds = out_test.argmax(dim=1)
    cert_results = compute_certified_accuracy(test_preds, test_labels, M)

    return test_acc, cert_results, M


def run_pgnncert_n_graph(dataset, gnn, T, train_args, retrain=False):
    """Run PGNNCert-N graph classification."""
    from node_hash import HashAgent, RobustGraphClassifier

    graphs, num_x, num_labels, mask_split, labels = load_graph_data(dataset)
    train_mask, val_mask, test_mask = mask_split

    hasher = HashAgent(h="md5", T=T)
    r_model = RobustGraphClassifier(hasher, graphs, labels,
                                    train_mask, val_mask, test_mask,
                                    num_x, num_labels, GNN=gnn)

    path = f"./checkpoints/robust_n/{gnn}/{dataset}/{T}/best_model"
    if (not retrain) and os.path.exists(path + "_0"):
        r_model.load_model(path)
    else:
        r_model.train(train_args)
        r_model.load_model(path)

    labels_t = torch.as_tensor(labels)
    out_test, M = r_model.vote(test_mask)
    test_labels = labels_t[test_mask]
    test_acc = evaluate(out_test, test_labels)

    test_preds = out_test.argmax(dim=1)
    cert_results = compute_certified_accuracy(test_preds, test_labels, M)

    return test_acc, cert_results, M


def run_secert_e_graph(dataset, gnn, T, train_args, retrain=False):
    """Run SECert-E graph classification."""
    from secert_edge_hash import HashAgent, SECertGraphClassifier

    graphs, num_x, num_labels, mask_split, labels = load_graph_data(dataset)
    train_mask, val_mask, test_mask = mask_split

    hasher = HashAgent(h="md5", T=T)
    r_model = SECertGraphClassifier(hasher, graphs, labels,
                                    train_mask, val_mask, test_mask,
                                    num_x, num_labels, GNN=gnn)

    path = f"./checkpoints/secert_e/{gnn}/{dataset}/{T}/best_model"
    if (not retrain) and os.path.exists(path):
        r_model.load_model(path)
    else:
        r_model.train_model(train_args)
        r_model.load_model(path)

    labels_t = torch.as_tensor(labels)
    out_test, M = r_model.vote(test_mask)
    test_labels = labels_t[test_mask]
    test_acc = evaluate(out_test, test_labels)

    test_preds = out_test.argmax(dim=1)
    cert_results = compute_certified_accuracy(test_preds, test_labels, M)

    return test_acc, cert_results, M


def run_secert_n_graph(dataset, gnn, T, train_args, retrain=False):
    """Run SECert-N graph classification."""
    from secert_node_hash import HashAgent, SECertGraphClassifier

    graphs, num_x, num_labels, mask_split, labels = load_graph_data(dataset)
    train_mask, val_mask, test_mask = mask_split

    hasher = HashAgent(h="md5", T=T)
    r_model = SECertGraphClassifier(hasher, graphs, labels,
                                    train_mask, val_mask, test_mask,
                                    num_x, num_labels, GNN=gnn)

    path = f"./checkpoints/secert_n/{gnn}/{dataset}/{T}/best_model"
    if (not retrain) and os.path.exists(path):
        r_model.load_model(path)
    else:
        r_model.train_model(train_args)
        r_model.load_model(path)

    labels_t = torch.as_tensor(labels)
    out_test, M = r_model.vote(test_mask)
    test_labels = labels_t[test_mask]
    test_acc = evaluate(out_test, test_labels)

    test_preds = out_test.argmax(dim=1)
    cert_results = compute_certified_accuracy(test_preds, test_labels, M)

    return test_acc, cert_results, M


def compute_certified_accuracy(preds, labels, M):
    """Compute certified accuracy at various perturbation sizes."""
    correct = (preds == labels)
    total = len(M)
    results = {}
    for p in [0, 1, 2, 3, 5, 10, 15, 20, 25, 30]:
        certified = (M >= p) & correct
        results[p] = certified.sum().item() / total if total > 0 else 0.0
    return results


_ALL_GRAPH_DATASETS = ["AIDS", "MUTAG", "PROTEINS", "DD"]
_ALL_ARCHS = ["GCN", "GSAGE", "GAT"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="AIDS",
                        choices=["AIDS", "MUTAG", "Mutagenicity", "PROTEINS", "DD", "CIFAR10", "MNIST"])
    parser.add_argument("--method", type=str, default="both",
                        choices=["normal", "pgnncert", "secert", "both", "all"])
    parser.add_argument("--variant", type=str, default="E", choices=["E", "N", "both"])
    parser.add_argument("--gnn", type=str, default="GCN", choices=["GCN", "GSAGE", "GAT"])
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--retrain", action="store_true",
                        help="Ignore existing checkpoints and retrain all requested models.")
    parser.add_argument("--all-datasets", action="store_true",
                        help="Run all graph datasets (AIDS, MUTAG, PROTEINS, DD)")
    parser.add_argument("--all-archs", action="store_true",
                        help="Run all GNN archs (GCN, GSAGE, GAT)")
    args = parser.parse_args()

    datasets_to_run = _ALL_GRAPH_DATASETS if args.all_datasets else [args.dataset]
    archs_to_run    = _ALL_ARCHS          if args.all_archs   else [args.gnn]

    for current_dataset in datasets_to_run:
        for current_gnn in archs_to_run:

            train_args = {
                "dataset": current_dataset,
                "paper": current_gnn,
                "lr": args.lr,
                "epochs": args.epochs,
                "clip_max": 2.0,
                "batch_size": 64,
                "early_stopping": 100,
                "seed": 42,
                "eval_enabled": True
            }

            print(f"\n{'='*60}")
            print(f"Graph Classification: {current_dataset} | GNN: {current_gnn} | T={args.T}")
            print(f"{'='*60}")
            if torch.cuda.is_available():
                print(f"CUDA enabled: {torch.cuda.get_device_name(0)} | torch CUDA {torch.version.cuda}")
            else:
                print("CUDA not available. Running on CPU.")

            results = {}

            variants = ["E", "N"] if args.variant == "both" else [args.variant]
            execution_plan = []

            if args.method in {"normal", "all"}:
                execution_plan.append(("normal", None))
            if args.method in {"pgnncert", "both", "all"}:
                for variant in variants:
                    execution_plan.append(("pgnncert", variant))
            if args.method in {"secert", "both", "all"}:
                for variant in variants:
                    execution_plan.append(("secert", variant))

            args.dataset = current_dataset
            args.gnn     = current_gnn

            for method, variant in execution_plan:
                t_start = time.time()
                cert = None

                if method == "normal":
                    acc = run_normal_graph(args.dataset, args.gnn, train_args, retrain=args.retrain)
                    key = "Normal"
                elif method == "pgnncert" and variant == "E":
                    acc, cert, M = run_pgnncert_e_graph(args.dataset, args.gnn, args.T, train_args, retrain=args.retrain)
                    key = "PGNNCert-E"
                elif method == "pgnncert" and variant == "N":
                    acc, cert, M = run_pgnncert_n_graph(args.dataset, args.gnn, args.T, train_args, retrain=args.retrain)
                    key = "PGNNCert-N"
                elif method == "secert" and variant == "E":
                    acc, cert, M = run_secert_e_graph(args.dataset, args.gnn, args.T, train_args, retrain=args.retrain)
                    key = "SECert-E"
                elif method == "secert" and variant == "N":
                    acc, cert, M = run_secert_n_graph(args.dataset, args.gnn, args.T, train_args, retrain=args.retrain)
                    key = "SECert-N"
                else:
                    continue

                elapsed = time.time() - t_start
                result = {"accuracy": acc, "time": elapsed}
                if cert is not None:
                    result["certified"] = cert
                results[key] = result

                print(f"\n{key}:")
                print(f"  Test Accuracy: {acc:.4f}")
                print(f"  Time: {elapsed:.1f}s")
                if cert is not None:
                    print(f"  Certified accuracy at p=0: {cert[0]:.4f}")
                    for p in [5, 10, 15]:
                        if p in cert:
                            print(f"  Certified accuracy at p={p}: {cert[p]:.4f}")

            if results:
                print("\nSummary:")
                for key, info in results.items():
                    print(f"  {key:12s} | Acc: {info['accuracy']:.4f} | Time: {info['time']:.1f}s")

            os.makedirs("results", exist_ok=True)
            result_path = f"results/graph_{current_dataset}_{current_gnn}_T{args.T}.json"
            with open(result_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {result_path}")
