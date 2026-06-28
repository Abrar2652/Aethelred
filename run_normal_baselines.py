# -*- coding: utf-8 -*-
"""
Normal GNN Baseline Runner  —  Table 1 (left columns: GCN / GSAGE / GAT)

Trains and evaluates plain GCN, GSAGE, and GAT on all 8 datasets
with NO certification / NO Aethelred.  Reproduces the "normally trained GNN"
columns from PGNNCert Table 1.

Hyperparameters match PGNNCert exactly:
  lr=0.002, epochs=200, clip_max=2.0, early_stopping=100, batch_size=64

Usage examples
--------------
  # Single dataset, single arch
  python run_normal_baselines.py --arch GCN --task node --dataset Cora-ML
  python run_normal_baselines.py --arch GSAGE --task graph --dataset PROTEINS

  # All datasets for one arch
  python run_normal_baselines.py --arch GCN --all
  python run_normal_baselines.py --arch GSAGE --all
  python run_normal_baselines.py --arch GAT --all

  # Full Table 1 baseline (all archs, all datasets)
  python run_normal_baselines.py --all-archs

  # Force retrain even if checkpoint exists
  python run_normal_baselines.py --arch GCN --all --retrain
"""

import argparse
import json
import os

from normal_baselines import run_normal_node, run_normal_graph


NODE_DATASETS  = ["Cora-ML", "CiteSeer", "PubMed", "Amazon-C"]
GRAPH_DATASETS = ["AIDS", "MUTAG", "PROTEINS", "DD"]
ARCHS          = ["GCN", "GSAGE", "GAT"]

# PGNNCert-compatible hyperparameters (identical to their training code)
TRAIN_ARGS = {
    "lr":             0.002,
    "epochs":         200,
    "clip_max":       2.0,
    "batch_size":     64,
    "early_stopping": 100,
    "seed":           42,
    "eval_enabled":   True,
}


# ──────────────────────────────────────────────────────────────────────────────

def run_one(arch, task, dataset, retrain=False):
    """Train/load one (arch, task, dataset) combo and return test accuracy."""
    args = dict(TRAIN_ARGS, paper=arch, dataset=dataset)
    if task == "node":
        acc = run_normal_node(dataset, arch, args, retrain=retrain)
    else:
        acc = run_normal_graph(dataset, arch, args, retrain=retrain)
    return acc


def run_all_for_arch(arch, retrain=False):
    """Run all 8 datasets for one arch.  Returns results dict."""
    results = {}
    print(f"\n{'='*70}")
    print(f"  Arch: {arch}  —  All datasets")
    print(f"{'='*70}")

    print(f"\n--- Node classification ---")
    for ds in NODE_DATASETS:
        try:
            acc = run_one(arch, "node", ds, retrain)
            results[ds] = acc
            print(f"  {ds:<14}  test_acc = {acc:.4f}")
        except Exception as e:
            results[ds] = None
            print(f"  {ds:<14}  FAILED: {e}")

    print(f"\n--- Graph classification ---")
    for ds in GRAPH_DATASETS:
        try:
            acc = run_one(arch, "graph", ds, retrain)
            results[ds] = acc
            print(f"  {ds:<14}  test_acc = {acc:.4f}")
        except Exception as e:
            results[ds] = None
            print(f"  {ds:<14}  FAILED: {e}")

    return results


def run_all_archs(retrain=False):
    """Run all 3 archs × 8 datasets and print the full Table 1 baseline."""
    all_results = {}
    for arch in ARCHS:
        all_results[arch] = run_all_for_arch(arch, retrain)

    _print_table(all_results)
    _save_results(all_results)


def _print_table(all_results):
    datasets = NODE_DATASETS + GRAPH_DATASETS
    print(f"\n{'='*70}")
    print("TABLE 1 — Normal GNN Baseline (clean accuracy)")
    print(f"{'='*70}")
    header = f"{'Dataset':<14}" + "".join(f"  {a:>8}" for a in ARCHS)
    print(header)
    print("-" * len(header))
    for ds in datasets:
        row = f"{ds:<14}"
        for arch in ARCHS:
            val = all_results.get(arch, {}).get(ds)
            row += f"  {val:.4f}" if val is not None else "      N/A"
        print(row)
    print()


def _save_results(all_results):
    os.makedirs("results", exist_ok=True)
    path = "results/normal_baselines.json"
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {path}")


# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Normal GNN baseline runner (Table 1 clean-accuracy columns)"
    )
    parser.add_argument("--arch", type=str, choices=ARCHS, default=None,
                        help="GNN architecture: GCN, GSAGE, or GAT")
    parser.add_argument("--task", type=str, choices=["node", "graph"], default=None,
                        help="Task: node or graph classification")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Single dataset (e.g. Cora-ML, PROTEINS)")
    parser.add_argument("--all", action="store_true",
                        help="Run all datasets for the chosen --arch")
    parser.add_argument("--all-archs", action="store_true",
                        help="Run all archs × all datasets (full Table 1 baseline)")
    parser.add_argument("--retrain", action="store_true",
                        help="Force retrain even if a checkpoint already exists")
    args = parser.parse_args()

    # ── full table ──────────────────────────────────────────────────────────
    if args.all_archs:
        run_all_archs(retrain=args.retrain)
        return

    # ── all datasets for one arch ──────────────────────────────────────────
    if args.all:
        if not args.arch:
            parser.error("--all requires --arch")
        results = run_all_for_arch(args.arch, retrain=args.retrain)
        _print_table({args.arch: results})
        _save_results({args.arch: results})
        return

    # ── single dataset ─────────────────────────────────────────────────────
    if args.arch and args.task and args.dataset:
        acc = run_one(args.arch, args.task, args.dataset, retrain=args.retrain)
        print(f"\n{args.arch} | {args.task} | {args.dataset}  →  test_acc = {acc:.4f}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
