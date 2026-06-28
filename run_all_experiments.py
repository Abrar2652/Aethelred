#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete experiment runner for SECert paper.
Reproduces Table 1, Table 2, Table 3 with PGNNCert (baseline) vs SECert (ours).

Usage:
  python run_all_experiments.py --table 1    # Table 1 only
  python run_all_experiments.py --table 2    # Table 2 only
  python run_all_experiments.py --table 3    # Table 3 only (CIFAR10)
  python run_all_experiments.py --table all  # All tables
  python run_all_experiments.py --quick      # Quick verification (5 epochs, 1 dataset)
"""

import argparse
import subprocess
import sys
import os
import json
import time


def run_cmd(cmd, timeout=7200):
    """Run a command and return output."""
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if result.stdout:
        print(result.stdout[-2000:])  # Print last 2000 chars
    if result.returncode != 0 and result.stderr:
        print(f"STDERR: {result.stderr[-1000:]}")
    return result


def run_table1(epochs=200):
    """
    Table 1: Node/graph accuracy of normally trained GNN and of
    PGNNCert/SECert with GNN trained on the subgraphs.

    Datasets: Cora-ML, Citeseer, Pubmed, Amazon-C (node), AIDS, MUTAG, Proteins, DD (graph)
    GNNs: GCN, GSAGE, GAT
    """
    print("\n" + "=" * 80)
    print("TABLE 1: Node/Graph Accuracy Comparison")
    print("=" * 80)

    node_datasets = ["Cora-ML", "CiteSeer", "PubMed", "Amazon-C"]
    graph_datasets = ["AIDS", "MUTAG", "PROTEINS", "DD"]
    gnns = ["GCN", "GSAGE", "GAT"]

    # Node classification
    py = sys.executable
    for ds in node_datasets:
        for gnn in gnns:
            run_cmd(f"\"{py}\" run_node_experiment.py --dataset {ds} --gnn {gnn} "
                    f"--method all --variant both --T 60 --epochs {epochs}")

    # Graph classification
    for ds in graph_datasets:
        for gnn in gnns:
            run_cmd(f"\"{py}\" run_graph_experiment.py --dataset {ds} --gnn {gnn} "
                    f"--method all --variant both --T 50 --epochs {epochs}")


def run_table2(epochs=200):
    """
    Table 2: Certified graph accuracy at perturbation edges = 0, 5, 10, 15.
    Datasets: PROTEINS, DD
    """
    print("\n" + "=" * 80)
    print("TABLE 2: Certified Graph Accuracy vs Edge Perturbations")
    print("=" * 80)

    py = sys.executable
    for ds in ["PROTEINS", "DD"]:
        run_cmd(f"\"{py}\" run_graph_experiment.py --dataset {ds} --gnn GCN "
                f"--method both --variant both --T 50 --epochs {epochs}")


def run_table3(epochs=200):
    """
    Table 3: Results on CIFAR10 (superpixel graph).
    Paper reports PGNNCert variants on CIFAR10.
    """
    print("\n" + "=" * 80)
    print("TABLE 3: CIFAR10 Superpixel Results")
    print("=" * 80)

    py = sys.executable
    run_cmd(f"\"{py}\" run_graph_experiment.py --dataset CIFAR10 --gnn GCN "
            f"--method pgnncert --variant both --T 50 --epochs {epochs}")


def run_quick_verification():
    """Quick test: 5 epochs on CiteSeer to verify code works."""
    print("\n" + "=" * 80)
    print("QUICK VERIFICATION: 5 epochs on CiteSeer")
    print("=" * 80)
    py = sys.executable
    run_cmd(f"\"{py}\" run_node_experiment.py --dataset CiteSeer --gnn GCN "
            "--method both --variant E --T 10 --epochs 5")


def collect_results():
    """Collect and display all results in table format."""
    print("\n" + "=" * 80)
    print("COLLECTED RESULTS")
    print("=" * 80)

    result_dir = "results"
    if not os.path.exists(result_dir):
        print("No results directory found.")
        return

    for fname in sorted(os.listdir(result_dir)):
        if fname.endswith(".json"):
            fpath = os.path.join(result_dir, fname)
            with open(fpath, "r") as f:
                data = json.load(f)
            print(f"\n--- {fname} ---")
            for method, info in data.items():
                acc = info.get("accuracy", 0)
                cert = info.get("certified")
                t = info.get("time", 0)
                if cert is None:
                    print(f"  {method:15s} | Acc: {acc:.4f} | Time: {t:.1f}s")
                else:
                    print(f"  {method:15s} | Acc: {acc:.4f} | "
                          f"p=0: {cert.get('0', cert.get(0, 0)):.4f} | "
                          f"p=5: {cert.get('5', cert.get(5, 0)):.4f} | "
                          f"p=10: {cert.get('10', cert.get(10, 0)):.4f} | "
                          f"Time: {t:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=str, default="all",
                        choices=["1", "2", "3", "all"])
    parser.add_argument("--quick", action="store_true",
                        help="Quick 5-epoch verification run")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--collect", action="store_true",
                        help="Just collect and display existing results")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if args.collect:
        collect_results()
    elif args.quick:
        run_quick_verification()
    else:
        if args.table in ["1", "all"]:
            run_table1(args.epochs)
        if args.table in ["2", "all"]:
            run_table2(args.epochs)
        if args.table in ["3", "all"]:
            run_table3(args.epochs)
        collect_results()
