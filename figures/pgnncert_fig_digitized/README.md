# PGNNCert — Plot Extraction & Digitization

Source: *Deterministic Certification of GNNs against Graph Poisoning Attacks with Arbitrary Perturbations* (PGNNCert, arXiv:2503.18503).

## 1. Plot inventory (what was extracted)

All data-bearing plots are **certified-accuracy vs. perturbation-size line charts** (matplotlib raster images embedded in the PDF). 13 "master" figures × 4 subfigures each = **52 charts**, extracted at native 864×576 resolution from PDF pages 7, 15, 16.

| Fig | Type | Curves (legend) | Datasets |
|---|---|---|---|
| 3 | PGNNCert-E, node acc | S=40/60/80 | Cora-ML, Citeseer, Pubmed, Amazon-C |
| 4 | PGNNCert-N, node acc | S=40/60/80 | (same) |
| 5 | PGNNCert-E, graph acc | S=30/50/70 | AIDS, MUTAG, PROTEINS, DD |
| 6 | PGNNCert-N, graph acc | S=30/50/70 | (same) |
| **7** | **PGNNCert vs SOTA** | PGNNCert-E/N, Bi-RS-Inc/Exc, RS, Bagging | Cora-ML, Citeseer, Pubmed, Amazon-C |
| 10–13 | E/N with GSAGE & GAT backbones | S=40/60/80 | node datasets |
| 14–17 | Hash-function ablation | h=md5/sha1/sha256 | node + graph datasets |

## 2. Which figures are directly comparable to a new SOTA model

**Only Figure 7.** It is the single figure that plots *external competing defenses* as named, separately-colored curves on a head-to-head axis (certified node accuracy vs. number of injected nodes, under node-injection poisoning). Methods present: **PGNNCert-E, PGNNCert-N, Bi-RS-Include, Bi-RS-Exclude, RS, Bagging**. A new SOTA model can be benchmarked apples-to-apples on this axis.

**Figures 3–6 and 10–17 are self-ablations** — every curve is a *variant of PGNNCert itself* (different S, GNN backbone, or hash). They contain no competing methods, so they are not direct SOTA comparisons (you would only be comparing against PGNNCert's own settings). They remain useful as reference curves if you overlay your model on the same axes.

Additional already-tabulated comparisons live in the paper text: **Table 2** (vs Bagging, graph classification, edge perturbations), Table 3 (CIFAR10), Table 4 (vs Metattack). These are numeric in the PDF and need no image extraction.

## 3. Extraction method & validation

Per chart: detected the shared plot box (px L=108,R=778,T=69,B=504), calibrated pixel→data using each figure's x-max (node=48, SOTA=50, graph: AIDS/MUTAG=24, PROTEINS=34, DD=40/30) and y∈[0,1]; matched each curve by reference color; masked legend regions; took the densest per-column pixel cluster as the line. **Every one of the 52 charts was validated by re-plotting the extracted points back onto the original image** — all points sit on their curves.

## 4. Files

- `pgnncert_all_digitized.csv` — long format: figure, dataset, x_axis, curve, x, certified_accuracy (3,757 rows, all 52 charts).
- `pgnncert_all_digitized.json` — same data, nested by chart.
- `fig7_sota_comparison.json` — just the SOTA-comparable Figure 7.

## 5. Figure 7 values at key injection counts (the SOTA comparison)

### Fig 7 — Cora-ML (x = Injected Nodes)

| Method | x=0 | x=2 | x=5 | x=10 | x=15 | x=20 | x=25 | x=30 | x=35 | x=40 |
|---|---|---|---|---|---|---|---|---|---|---|
| PGNNCert-N | 0.69 | 0.68 | 0.67 | 0.65 | 0.64 | 0.62 | 0.58 |  |  |  |
| PGNNCert-E |  |  | 0.50 |  |  |  |  |  |  |  |
| Bi-RS-Exclude | 0.76 | 0.65 | 0.56 | 0.38 | 0.16 | 0.01 | 0.01 |  |  |  |
| Bi-RS-Include | 0.67 | 0.25 | 0.06 | 0.01 | 0.01 |  |  |  |  |  |
| Bagging | 0.70 | 0.22 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |  |  |  |
| RS | 0.72 | 0.01 |  |  |  |  |  |  |  |  |

### Fig 7 — Citeseer (x = Injected Nodes)

| Method | x=0 | x=2 | x=5 | x=10 | x=15 | x=20 | x=25 | x=30 | x=35 | x=40 |
|---|---|---|---|---|---|---|---|---|---|---|
| PGNNCert-N | 0.67 | 0.65 | 0.65 | 0.63 | 0.60 | 0.57 | 0.53 |  |  |  |
| PGNNCert-E |  | 0.63 | 0.48 |  |  |  |  |  |  |  |
| Bi-RS-Exclude | 0.75 | 0.67 | 0.58 | 0.41 | 0.18 | 0.01 | 0.01 |  |  |  |
| Bi-RS-Include |  | 0.26 | 0.07 | 0.02 | 0.01 |  |  |  |  |  |
| Bagging | 0.73 | 0.22 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |  |  |  |
| RS | 0.69 | 0.01 |  |  |  |  |  |  |  |  |

### Fig 7 — Pubmed (x = Injected Nodes)

| Method | x=0 | x=2 | x=5 | x=10 | x=15 | x=20 | x=25 | x=30 | x=35 | x=40 |
|---|---|---|---|---|---|---|---|---|---|---|
| PGNNCert-N | 0.86 | 0.85 | 0.85 | 0.84 | 0.83 | 0.82 | 0.79 |  |  |  |
| PGNNCert-E |  | 0.82 | 0.69 | 0.01 |  |  |  |  |  |  |
| Bi-RS-Exclude | 0.83 | 0.80 | 0.78 | 0.71 | 0.59 | 0.02 | 0.01 |  |  |  |
| Bi-RS-Include | 0.85 | 0.34 | 0.16 | 0.06 | 0.02 |  |  |  |  |  |
| Bagging | 0.84 | 0.46 | 0.10 | 0.01 | 0.01 | 0.01 | 0.01 |  |  |  |
| RS |  | 0.37 | 0.01 |  |  |  |  |  |  |  |

### Fig 7 — Amazon-C (x = Injected Nodes)

| Method | x=0 | x=2 | x=5 | x=10 | x=15 | x=20 | x=25 | x=30 | x=35 | x=40 |
|---|---|---|---|---|---|---|---|---|---|---|
| PGNNCert-N | 0.82 | 0.81 | 0.79 | 0.77 | 0.74 | 0.71 | 0.66 |  |  |  |
| PGNNCert-E |  | 0.78 | 0.56 | 0.01 | 0.01 |  |  |  |  |  |
| Bi-RS-Exclude | 0.90 | 0.84 | 0.81 | 0.73 | 0.59 | 0.02 | 0.01 |  |  |  |
| Bi-RS-Include |  | 0.76 | 0.62 | 0.41 | 0.23 |  |  |  |  |  |
| Bagging | 0.86 | 0.57 | 0.35 | 0.07 | 0.01 | 0.01 | 0.01 |  |  |  |
| RS | 0.88 | 0.26 | 0.03 | 0.01 |  |  |  |  |  |  |

> Note: PGNNCert-E in Fig 7 has a near-vertical cliff, so it has markers at only a few integer x; blank cells = no marker at that exact x (curve still captured in the CSV where present). Legend labels in Figs 3–6/10–13 print "T=" but denote the number of subgraphs S, per the paper text.
