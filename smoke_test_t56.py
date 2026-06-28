# -*- coding: utf-8 -*-
"""
Smoke test: DIR vs Aethelred on SPMotif — 1 seed, 1 bias, fast settings.
Verifies both pipelines run end-to-end and prints Precision@K side by side.

Run:
    python smoke_test_t56.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

# ── project imports ──────────────────────────────────────────────────────────
from datasets.spmotif import generate_spmotif, split_spmotif
from baselines.dir_gnn import train_dir, dir_get_explanation

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── settings ──────────────────────────────────────────────────────────────────
SEED       = 42
BIAS       = 0.33        # Balance setting
EPOCHS     = 50          # fast smoke
N_EVAL     = 50          # graphs to evaluate
N_GRAPHS   = 600         # small dataset (200 per class)
NC         = 3

# ── data ──────────────────────────────────────────────────────────────────────
print(f"\nGenerating SPMotif  bias={BIAS}  n_graphs={N_GRAPHS}  random_features=True ...")
tr_src = generate_spmotif(n_graphs=N_GRAPHS, bias=BIAS, seed=SEED, random_features=True)
cnt_src = generate_spmotif(n_graphs=300, bias=0.33, seed=SEED+333, random_features=True)
train_g, val_g, test_g = split_spmotif(tr_src,  seed=SEED)
cnt_g, _, _            = split_spmotif(cnt_src, seed=SEED)
NF = train_g[0].x.size(1)
print(f"  Train={len(train_g)}  Val={len(val_g)}  Test={len(test_g)}  nf={NF}")

# ── Precision@K helper ────────────────────────────────────────────────────────
def prec_at_k(mask, gt):
    """Adaptive K = GT count. Both tensors on CPU."""
    mask = mask.cpu().float()
    gt   = gt.cpu().float()
    k    = max(1, int(gt.sum().item()))
    if k >= mask.numel():
        pred_bin = torch.ones_like(mask)
    else:
        _, idx   = torch.topk(mask, k)
        pred_bin = torch.zeros_like(mask)
        pred_bin[idx] = 1.0
    tp = (pred_bin * gt).sum().item()
    return tp / (k + 1e-8)

# ── DIR ───────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  [DIR]  epochs={EPOCHS}  seed={SEED}")
print(f"{'='*60}")
torch.manual_seed(SEED)
dir_model, dir_acc = train_dir(
    [g.cpu() for g in train_g],
    [g.cpu() for g in val_g],
    [g.cpu() for g in test_g],
    NF, NC,
    device=device,
    hidden_dim=64,
    epochs=EPOCHS,
    lr=0.001,
    n_envs=5,
    irm_lambda=1.0,
    seed=SEED,
)
dir_model.eval()

dir_precs = []
for g in test_g[:N_EVAL]:
    mask = dir_get_explanation(dir_model, g.cpu(), device)
    dir_precs.append(prec_at_k(mask, g.ground_truth_mask))

print(f"  DIR   Test Acc={dir_acc:.4f}   Precision@K={np.mean(dir_precs):.4f}")

# ── Aethelred ─────────────────────────────────────────────────────────────────
# Import training helpers from the main comparison script
print(f"\n{'='*60}")
print(f"  [Aethelred]  epochs={EPOCHS}  seed={SEED}")
print(f"{'='*60}")

# Pull internal helpers from run_aethelred_comparison
import importlib, types
_mod = importlib.import_module("run_aethelred_comparison")
_train_aethelred_expl = _mod._train_aethelred_expl

aeth_mdl, aeth_acc = _train_aethelred_expl(
    [g.cpu() for g in train_g],
    [g.cpu() for g in val_g],
    [g.cpu() for g in test_g],
    NF, NC,
    epochs=EPOCHS,
    seed=SEED,
    ood_tr_g=[g.cpu() for g in cnt_g],
    mask_budget=0.25,
    spar_w=0.30,
    ent_w=0.15,
    ctx_w=1.0,
    adv_w=1.0,
    irm_w=1.0,
    cert_w=0.50,
    eps_ibp=0.10,
)
aeth_mdl.eval()

aeth_precs = []
for g in test_g[:N_EVAL]:
    g_dev = g.clone().to(device)
    with torch.no_grad():
        mask = aeth_mdl.causal_core(g_dev.x.float(), g_dev.edge_index)
    aeth_precs.append(prec_at_k(mask.cpu(), g.ground_truth_mask))

print(f"  Aeth  Test Acc={aeth_acc:.4f}   Precision@K={np.mean(aeth_precs):.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  SMOKE TEST SUMMARY  bias={BIAS}  seed={SEED}  epochs={EPOCHS}")
print(f"{'='*60}")
print(f"  {'Model':<20} {'Test Acc':>10} {'Prec@K':>10}")
print(f"  {'-'*42}")
print(f"  {'DIR':<20} {dir_acc:>10.4f} {np.mean(dir_precs):>10.4f}")
print(f"  {'Aethelred (Ours)':<20} {aeth_acc:>10.4f} {np.mean(aeth_precs):>10.4f}")
delta = np.mean(aeth_precs) - np.mean(dir_precs)
verdict = "Aethelred WINS ✓" if delta > 0 else "Aethelred LOSES ✗"
print(f"\n  Δ Prec@K = {delta:+.4f}   →  {verdict}")
print(f"{'='*60}")
