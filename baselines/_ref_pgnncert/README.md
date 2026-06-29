# SECert: Shared-Encoder Certified Defense for Graph Neural Networks Against Arbitrary Poisoning Attacks

**Target Venue: NeurIPS 2026**

## Abstract

Graph Neural Networks (GNNs) are vulnerable to training-time poisoning attacks that arbitrarily perturb edges, nodes, and node features. While PGNNCert provides the first deterministic certified defense, it suffers from a fundamental **accuracy-robustness tradeoff**: training S independent classifiers on sparse subgraphs yields weak individual models. We introduce **SECert** (Shared-Encoder Certified Defense), which breaks this tradeoff through three innovations:

1. **Shared GNN Backbone**: A single shared feature extractor serves all S subgraph classifiers, with only lightweight per-subgraph classification heads. We prove this maintains the same deterministic certification guarantee while providing ~S x richer gradient signal during training.

2. **Margin-Boosting Training**: A hinge loss directly optimizes the ensemble vote margin, increasing the certified perturbation radius beyond what standard cross-entropy training achieves.

3. **Dramatic Efficiency Gains**: SECert reduces parameters by ~6x (for S=50), enabling faster training and inference while achieving higher clean accuracy and larger certified perturbation sizes.

**Key Theoretical Result**: We prove that sharing backbone parameters across ensemble members does NOT break the deterministic certification guarantee. At test time, each perturbation affects exactly one subgraph's input to the frozen backbone, leaving all other votes unchanged. This insight is general and applicable to any partition-based certified defense.

## Method Overview

### PGNNCert (Baseline)
```
S independent GNNs: [GNN_1, GNN_2, ..., GNN_S]
Each GNN_i = [3 Conv layers + Linear]
Parameters: S x (Conv layers + Linear) = O(S * d^2 * L)
```

### SECert (Ours)
```
1 shared backbone:  B = [3 Conv layers]           (shared across all S classifiers)
S lightweight heads: [Head_1, Head_2, ..., Head_S]  (per-subgraph adaptation)
Parameters: O(d^2 * L + S * d * C)  where C << d^2
```

### Why Certification Still Holds

**Theorem (Informal)**: At test time with frozen parameters, a perturbation to subgraph j only changes the input fed to backbone B when processing subgraph j. All other subgraphs' inputs are unchanged, so their outputs through B and their respective heads are unchanged. Therefore each perturbation changes at most 1 vote, and the same certification bound as PGNNCert applies.

### Architecture Diagram

```
                    Subgraph 1 ──> [Shared Backbone B] ──> [Head 1] ──> Vote 1
                    Subgraph 2 ──> [Shared Backbone B] ──> [Head 2] ──> Vote 2
Graph G ──> Hash ── ...
  Partition         Subgraph S ──> [Shared Backbone B] ──> [Head S] ──> Vote S
                                                                          │
                                                              Majority Vote ──> Prediction
                                                              Vote Margin  ──> Certified Radius
```

## Installation

### Requirements
- Python >= 3.8
- PyTorch (CUDA build recommended for GPU runs)
- PyTorch Geometric
- NumPy, SciPy, scikit-learn

### Setup
```bash
# Clone repository
git clone https://github.com/60akramuddoula/pgn.git
cd pgn

# Install dependencies
pip install torch torchvision torchaudio
pip install torch-geometric scipy scikit-learn

# Verify installation
python verify_code.py
```

### Pre-download Paper Datasets
```bash
# Download every paper dataset into ./datasets/<dataset-name>/
python download_dataset.py

# Download only selected datasets
python download_dataset.py --dataset Cora-ML CiteSeer PubMed Amazon-C
```

### GTX 1650 (Windows, 4GB VRAM) Recommended Setup

If your `nvidia-smi` shows an older driver branch (for example 457.xx, CUDA 11.1),
use the CUDA 11.1 wheel set below:

```powershell
# From repository root
py -3.9 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel

# PyTorch (CUDA 11.1)
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

# PyG extensions matching torch 1.10/cu111
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
pip install torch-geometric==2.0.4 scipy scikit-learn numpy

# GPU check
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

If you update to a newer NVIDIA driver, you can use newer PyTorch/CUDA wheels
(for example cu118/cu121) and the matching `data.pyg.org` wheel index.

### Low-VRAM Commands (GTX 1650 Safe Start)

Start with smaller `T` to reduce memory/time:

```powershell
# Quick functional check
python verify_code.py

# Node classification (safer on 4GB VRAM)
python run_node_experiment.py --dataset CiteSeer --method secert --variant E --gnn GCN --T 20 --epochs 50

# Graph classification (safer on 4GB VRAM)
python run_graph_experiment.py --dataset AIDS --method secert --variant E --gnn GCN --T 15 --epochs 50
```

Then increase `--T` and `--epochs` gradually if memory remains stable.

## Project Structure

```
.
├── gnn.py                     # GNN architectures (original + shared backbone)
├── utils.py                   # Utility functions (evaluation, checkpointing)
├── datasets/
│   ├── __init__.py
│   ├── dataset_loader.py      # Data loading for node/graph classification
│   └── utils.py               # Data preprocessing utilities
├── edge_hash.py               # PGNNCert-E (baseline, edge-centric)
├── node_hash.py               # PGNNCert-N (baseline, node-centric)
├── secert_edge_hash.py        # SECert-E (ours, edge-centric)
├── secert_node_hash.py        # SECert-N (ours, node-centric)
├── run_node_experiment.py     # Node classification experiment runner
├── run_graph_experiment.py    # Graph classification experiment runner
├── run_all_experiments.py     # Automated runner for all tables
├── verify_code.py             # Code verification (synthetic data, 5 epochs)
├── create_synthetic_data.py   # Synthetic dataset generator
└── README.md
```

## Quick Start

### Verify Code Works
```bash
python verify_code.py
```
This runs 5 tests (PGNNCert-E node, SECert-E node, PGNNCert-E graph, SECert-E graph, SECert-N node) on synthetic data for 5 epochs each.

### Run a Single Experiment
```bash
# Node classification: full Table 1-style comparison on CiteSeer with GCN
python run_node_experiment.py --dataset CiteSeer --method all --variant both --gnn GCN --T 60 --epochs 200

# Node classification: SECert-E on CiteSeer with GCN
python run_node_experiment.py --dataset CiteSeer --method secert --variant E --gnn GCN --T 60 --epochs 200

# Graph classification: Both methods on AIDS
python run_graph_experiment.py --dataset AIDS --method both --variant both --gnn GCN --T 50 --epochs 200
```

If you already have checkpoints from earlier runs and want a fair fresh comparison, add `--retrain` to overwrite the saved model for the requested configuration.

## Reproducing Paper Tables

### Table 1: Node/Graph Accuracy Comparison

Node classification datasets: Cora-ML, CiteSeer, PubMed, Amazon-C
Graph classification datasets: AIDS, MUTAG, PROTEINS, DD
GNN backbones: GCN, GSAGE, GAT

```bash
# Run all Table 1 experiments
python run_all_experiments.py --table 1 --epochs 200

# Or run individual experiments:
# Node classification
for dataset in Cora-ML CiteSeer PubMed Amazon-C; do
  for gnn in GCN GSAGE GAT; do
    python run_node_experiment.py --dataset $dataset --method all --variant both --gnn $gnn --T 60 --epochs 200
  done
done

# Graph classification
for dataset in AIDS MUTAG PROTEINS DD; do
  for gnn in GCN GSAGE GAT; do
    python run_graph_experiment.py --dataset $dataset --method all --variant both --gnn $gnn --T 50 --epochs 200
  done
done
```

**Expected output format** (results saved to `results/` directory):
```
Method          | Acc    | p=0    | p=5    | p=10   | Time
PGNNCert-E      | 0.6800 | 0.6700 | 0.4500 | 0.2100 | 120.5s
SECert-E        | 0.7100 | 0.7000 | 0.4900 | 0.2500 | 45.2s
PGNNCert-N      | 0.6700 | 0.6600 | 0.4300 | 0.1900 | 118.3s
SECert-N        | 0.6900 | 0.6800 | 0.4700 | 0.2300 | 43.1s
```

### Table 2: Certified Graph Accuracy vs Edge Perturbations

Datasets: PROTEINS, DD
Perturbation edges: 0, 5, 10, 15

```bash
python run_all_experiments.py --table 2 --epochs 200

# Or individually:
python run_graph_experiment.py --dataset PROTEINS --method both --variant both --gnn GCN --T 50 --epochs 200
python run_graph_experiment.py --dataset DD --method both --variant both --gnn GCN --T 50 --epochs 200
```

The certified accuracy at each perturbation size p is computed as:
```
CertifiedAcc(p) = fraction of test samples where (prediction correct) AND (vote margin >= 2p)
```

### Table 3: CIFAR10 Superpixel Results

```bash
python run_all_experiments.py --table 3 --epochs 200

# Equivalent direct command:
python run_graph_experiment.py --dataset CIFAR10 --method pgnncert --variant both --gnn GCN --T 50 --epochs 200
```

### Collect All Results
```bash
python run_all_experiments.py --collect
```

## Hyperparameters

| Parameter | Node Classification | Graph Classification |
|-----------|-------------------|---------------------|
| S (subgraphs) | 60 | 50 |
| Hash function | MD5 | MD5 |
| Learning rate | 0.002 | 0.002 |
| Epochs | 200 | 200 |
| Gradient clip | 2.0 | 2.0 |
| Early stopping | 100 epochs | 100 epochs |
| Hidden size | 20 | 32 |
| GNN layers | 3 | 3 |
| Lambda (margin) | 0.1 | 0.1 |

## Key Differences from PGNNCert

| Aspect | PGNNCert | SECert (Ours) |
|--------|----------|---------------|
| Architecture | S independent GNNs | 1 shared backbone + S heads |
| Parameters | O(S * model_size) | O(model_size + S * head_size) |
| Training signal | Each GNN sees 1 subgraph | Backbone sees ALL subgraphs |
| Certification | Deterministic | Deterministic (same guarantee) |
| Loss function | Cross-entropy only | Cross-entropy + margin hinge |
| Speed | 1x | ~S/2 x faster |

## Datasets

### Node Classification
- **Cora-ML**: 2,995 nodes, 8,158 edges, 7 classes, 2,879 features
- **CiteSeer**: 3,327 nodes, 4,552 edges, 6 classes, 3,703 features
- **PubMed**: 19,717 nodes, 44,324 edges, 3 classes, 500 features
- **Amazon-C**: 13,752 nodes, 245,861 edges, 10 classes, 767 features

### Graph Classification
- **AIDS**: 2,000 graphs, 2 classes
- **MUTAG**: 188 graphs, 2 classes
- **PROTEINS**: 1,113 graphs, 2 classes
- **DD**: 1,178 graphs, 2 classes
- **CIFAR10 (superpixel)**: benchmark graph classification dataset used in Table 3

### Data Splits
- Node: 30% train / 10% val / 30% test (stratified by class)
- Graph: 50% train / 20% val / 30% test (stratified by class)

## Theoretical Contributions

### Theorem 1 (Certification Preservation)
Let B be a shared backbone network and {h_1, ..., h_S} be S classification heads. Given a graph G partitioned into S subgraphs via a deterministic hash function, define the voting classifier f(G) = argmax_y sum_i I[h_i(B(G_i)) = y]. Under arbitrary perturbation with budget P (affecting at most P subgraphs), if the vote margin exceeds 2P, then f(G) = f(G') for any perturbed graph G' within budget P.

### Theorem 2 (Efficiency)
SECert reduces parameters from O(S * d^2 * L) to O(d^2 * L + S * d * C), where d is hidden dimension, L is number of layers, C is number of classes. For typical settings (S=50, L=3, d=20, C=7), this yields a ~6x reduction.

### Proposition (Gradient Benefit)
During training, the shared backbone receives gradients from all S classification tasks simultaneously, providing an S-fold increase in gradient signal compared to PGNNCert's independent classifiers.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{secert2026,
  title={SECert: Breaking the Accuracy-Robustness Tradeoff in Certified Graph Defense via Shared-Encoder Ensemble Architectures},
  author={Anonymous},
  booktitle={NeurIPS},
  year={2026}
}
```

Also cite the PGNNCert baseline:

```bibtex
@inproceedings{lee2025pgnncert,
  title={PGNNCert: Deterministic Certification of Graph Neural Networks Against Graph Poisoning Attacks with Arbitrary Perturbations},
  author={Lee, Richard and Wang, Binghui},
  booktitle={CVPR},
  year={2025}
}
```

## License

This project is for research purposes. Please refer to the original PGNNCert repository for licensing details of the baseline code.







cd /c/PGN
source venv/Scripts/activate
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
python verify_code.py

for ds in Cora-ML CiteSeer PubMed Amazon-C; do
  python run_node_experiment.py --dataset "$ds" --method both --variant both --gnn GCN --T 15 --epochs 20
done

for ds in AIDS MUTAG PROTEINS DD; do
  python run_graph_experiment.py --dataset "$ds" --method both --variant both --gnn GCN --T 12 --epochs 20
done

for ds in PROTEINS DD; do
  python run_graph_experiment.py --dataset "$ds" --method both --variant both --gnn GCN --T 15 --epochs 20
done

python run_graph_experiment.py --dataset CIFAR10 --method pgnncert --variant both --gnn GCN --T 8 --epochs 5
python run_all_experiments.py --collect
# pgnncert






# =========================
# 1) Open project
# =========================
cd /c/PGN

# (optional) check python version; 3.10/3.11 is safest for this setup
python --version

# =========================
# 2) Create + activate venv
# =========================
python -m venv venv
source venv/Scripts/activate

# =========================
# 3) Clean old/conflicting packages
# =========================
python -m pip install --upgrade pip setuptools wheel
pip uninstall -y torch torchvision torchaudio torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric numpy

# =========================
# 4) Install GPU stack
# =========================
pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.26.4 scipy scikit-learn
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch-geometric==2.3.1

# =========================
# 5) Verify CUDA + packages
# =========================
python -c "import torch, numpy, torch_geometric; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cuda_available', torch.cuda.is_available(), 'gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('numpy', numpy.__version__); print('pyg', torch_geometric.__version__)"

# =========================
# 6) Verify repo code
# =========================
python verify_code.py

# =========================
# 7) Run experiments (GPU auto-used)
# =========================
python run_node_experiment.py --dataset CiteSeer --method secert --variant E --gnn GCN --T 20 --epochs 50
python run_node_experiment.py --dataset CiteSeer --method pgnncert --variant E --gnn GCN --T 20 --epochs 50
python run_graph_experiment.py --dataset AIDS --method secert --variant E --gnn GCN --T 15 --epochs 50

# Full table run (very long):
# python run_all_experiments.py --table 1 --epochs 200


cd /c/PGN
source venv/Scripts/activate

# all paper datasets
python download_dataset.py

# only some datasets
python download_dataset.py --dataset Cora-ML CiteSeer PubMed Amazon-C

# list supported names
python download_dataset.py --list

# re-download only Cora-ML if needed
python download_dataset.py --dataset Cora-ML --force-cora-ml
