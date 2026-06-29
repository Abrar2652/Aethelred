#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-download all paper datasets into ./datasets so future runs can load locally.

Usage:
  python download_dataset.py
  python download_dataset.py --dataset CiteSeer PubMed
  python download_dataset.py --list
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import TUDataset


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataset_loader import ALL_PAPER_DATASETS
from datasets.dataset_loader import DATASETS_DIR
from datasets.dataset_loader import matri_to_index
from datasets.dataset_loader import matri_to_index_directed


CORA_ML_URL = (
    "https://raw.githubusercontent.com/danielzuegner/gnn-meta-attack/"
    "master/data/cora_ml.npz"
)


def _datasets_dir():
    return Path(DATASETS_DIR)


def _dataset_dir(name):
    return _datasets_dir() / name


def _download_file(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url}")
    print(f"          -> {destination}")
    urllib.request.urlretrieve(url, destination)


def _prepare_cora_ml_artifacts(npz_path):
    with np.load(npz_path, allow_pickle=True) as loader:
        loader = dict(loader)
        adj = sp.csr_matrix(
            (loader["adj_data"], loader["adj_indices"], loader["adj_indptr"]),
            shape=loader["adj_shape"],
        ).toarray()

    dataset_dir = npz_path.parent
    undirected_path = dataset_dir / "core_ml_edge_index.npy"
    directed_path = dataset_dir / "core_ml_edge_index_d.npy"
    np.save(undirected_path, matri_to_index(adj))
    np.save(directed_path, matri_to_index_directed(adj))
    print(f"[saved] {undirected_path}")
    print(f"[saved] {directed_path}")


def download_cora_ml(force=False):
    dataset_dir = _dataset_dir("Cora-ML")
    npz_path = dataset_dir / "cora_ml.npz"

    if npz_path.exists() and not force:
        print(f"[skip] Cora-ML already present at {npz_path}")
    else:
        _download_file(CORA_ML_URL, npz_path)

    _prepare_cora_ml_artifacts(npz_path)
    return dataset_dir


def download_citeseer():
    dataset_dir = _dataset_dir("CiteSeer")
    Planetoid(root=str(dataset_dir), name="CiteSeer", num_train_per_class=50)
    print(f"[ready] CiteSeer cached under {dataset_dir}")
    return dataset_dir


def download_pubmed():
    dataset_dir = _dataset_dir("PubMed")
    Planetoid(root=str(dataset_dir), name="PubMed", num_train_per_class=50)
    print(f"[ready] PubMed cached under {dataset_dir}")
    return dataset_dir


def download_amazon_c():
    dataset_dir = _dataset_dir("Amazon-C")
    Amazon(root=str(dataset_dir), name="computers")
    print(f"[ready] Amazon-C cached under {dataset_dir}")
    return dataset_dir


def download_tu_dataset(name):
    TUDataset(root=str(_datasets_dir()), name=name, use_node_attr=True)
    dataset_dir = _dataset_dir(name)
    print(f"[ready] {name} cached under {dataset_dir}")
    return dataset_dir


def download_cifar10():
    for split in ["train", "val", "test"]:
        GNNBenchmarkDataset(root=str(_datasets_dir()), name="CIFAR10", split=split)
    dataset_dir = _dataset_dir("CIFAR10")
    print(f"[ready] CIFAR10 cached under {dataset_dir}")
    return dataset_dir


DOWNLOADERS = {
    "Cora-ML": download_cora_ml,
    "CiteSeer": download_citeseer,
    "PubMed": download_pubmed,
    "Amazon-C": download_amazon_c,
    "AIDS": lambda: download_tu_dataset("AIDS"),
    "MUTAG": lambda: download_tu_dataset("MUTAG"),
    "PROTEINS": lambda: download_tu_dataset("PROTEINS"),
    "DD": lambda: download_tu_dataset("DD"),
    "CIFAR10": download_cifar10,
}


def _normalize_requested_names(names):
    if not names or "all" in [name.lower() for name in names]:
        return list(ALL_PAPER_DATASETS)

    aliases = {
        "citeseer": "CiteSeer",
        "pubmed": "PubMed",
        "amazon-c": "Amazon-C",
        "amazonc": "Amazon-C",
        "computers": "Amazon-C",
        "cora-ml": "Cora-ML",
        "cora_ml": "Cora-ML",
        "mutagenicity": "MUTAG",
        "mutag": "MUTAG",
        "proteins": "PROTEINS",
        "dd": "DD",
        "aids": "AIDS",
        "cifar10": "CIFAR10",
    }

    resolved = []
    for name in names:
        key = aliases.get(name.strip().lower(), name.strip())
        if key not in DOWNLOADERS:
            raise ValueError(
                f"Unsupported dataset '{name}'. Supported: {', '.join(ALL_PAPER_DATASETS)}"
            )
        resolved.append(key)
    return resolved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["all"],
        help="Datasets to download. Default: all paper datasets.",
    )
    parser.add_argument(
        "--force-cora-ml",
        action="store_true",
        help="Re-download cora_ml.npz even if it already exists.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the supported paper datasets and exit.",
    )
    args = parser.parse_args()

    if args.list:
        print("Supported paper datasets:")
        for name in ALL_PAPER_DATASETS:
            print(f"  - {name}")
        return

    targets = _normalize_requested_names(args.dataset)

    print(f"Dataset cache root: {_datasets_dir()}")
    print(f"Preparing {len(targets)} dataset(s): {', '.join(targets)}")

    for name in targets:
        print(f"\n{'=' * 72}")
        print(f"Downloading {name}")
        print(f"{'=' * 72}")
        if name == "Cora-ML":
            DOWNLOADERS[name](force=args.force_cora_ml)
        else:
            DOWNLOADERS[name]()

    print(f"\nAll requested datasets are ready under {_datasets_dir()}")


if __name__ == "__main__":
    main()
