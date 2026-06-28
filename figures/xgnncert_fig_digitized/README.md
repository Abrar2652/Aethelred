# XGNNCert — Plot Extraction & Digitization

Source: *Provably Robust Explainable Graph Neural Networks against Graph Perturbation Attacks* (XGNNCert, ICLR 2025).

## 1. Plot inventory (what was extracted)

All data plots are **certified perturbation size M_λ vs. λ** line charts (one data point per integer λ). 11 figures, **30 subfigures**, extracted at native 576×324 from pages 8, 18, 19.

| Fig | Page | Curves vary | Subfigures | Base explainer |
|---|---|---|---|---|
| 3 | 8 | T=30/50/70/90 | SG+House, SG+Diamond, Benzene | PGExplainer |
| 4 | 8 | p=0/0.2/0.3/0.4 | SG+House, SG+Diamond, Benzene | PGExplainer |
| 5 | 8 | γ=0.2/0.3/0.4 | SG+House, SG+Diamond, Benzene | PGExplainer |
| 6 | 18 | T=30/50/70/90 | SG+Wheel, FC | PGExplainer |
| 7 | 18 | T=30/50/70/90 | House, Diamond, Wheel, Benzene, FC | ReFine |
| 8 | 18 | T=30/50/70/90 | House, Diamond, Wheel, Benzene, FC | GSAT |
| 9 | 19 | p=0/0.2/0.3/0.4 | SG+Wheel, FC | PGExplainer |
| 10 | 19 | γ=0.2/0.3/0.4 | SG+Wheel, FC | PGExplainer |
| 11 | 19 | h=MD5/SHA1/SHA256 | House, Diamond, Wheel, Benzene, FC | PGExplainer |

## 2. Comparability to a new SOTA model

**No figure in this paper contains an external competing method as a plotted curve.** XGNNCert is presented as *the first* certifiably robust XGNN, so every curve in every figure is a **self-ablation** of XGNNCert's own hyperparameters (T, p, γ, hash h) or its choice of base explainer (PGExplainer/ReFine/GSAT). There is therefore no direct head-to-head plot to benchmark against, unlike Fig 7 of the PGNNCert paper.

The only cross-method comparison in the paper is **Table 4** (XGNNCert vs. the empirical defense V-InfoR, on explanation accuracy and edge-change fraction under the Li et al. 2024 attack) — already numeric, no extraction needed.

**What to compare a new model against:** the certified frontier M_λ-vs-λ of XGNNCert itself, at its stated **default setting (T=70, p=0.3, γ=0.3, MD5)**. These default curves, per dataset × base explainer, are in `xgnncert_default_T70_reference.json`. If your new SOTA model produces M_λ-vs-λ curves on the same datasets, those are the directly comparable reference points.

## 3. Extraction method & validation

Constant plot box (px L=72, T=39, B=284). Per chart: x-axis (λ=1..k) calibrated from detected x-tick marks (k = #groundtruth edges: House=6, Diamond=5, Wheel=8, Benzene=6, FC=5); y-axis (M_λ) calibrated from the per-chart y-max read from tick labels (varies: 4.0, 5, 7.5, or 10). Curves matched by color; legend swatches masked via tight bounding-box detection; per-λ value chosen by neighbor-consistency to avoid legend/crossover contamination. **All 30 charts validated by re-plotting extracted points onto the originals** (see validation_*.png).

## 4. A note on inferred points

63 of 660 points (≈10%) are marked `source=inferred` in the CSV. These occur where curves perfectly overlap — either converging to M_λ=0 at high λ (the dominant case, per the paper's monotone-decreasing M_λ), or coinciding at a crossover. Inferred = 0 at floor convergence, or the median of the overlapping curves otherwise. All measured points are direct pixel reads. The `source` column lets you filter these out if you need measured-only data.

## 5. Files

- `xgnncert_all_digitized.csv` — long format: figure, dataset, curve, lambda, M_lambda, y_max, source (660 rows).
- `xgnncert_all_digitized.json` — nested by chart, includes detected legend boxes.
- `xgnncert_default_T70_reference.json` — the default-setting (T=70) reference curves per dataset×explainer.
- `validation_Figs3-5/6-8/9-11.png` — extracted points overlaid on every original chart.

## 6. Default-setting reference (T=70) — M_λ at each λ


### PGExplainer (T=70 default)

| Dataset | λ=1 | λ=2 | λ=3 | λ=4 | λ=5 | λ=6 | λ=7 | λ=8 |
|---|---|---|---|---|---|---|---|---|
| SG+House | 4.57 | 2.37 | 1.59 | 0.91 | 0.22 | 0.00 |  |  |
| SG+Diamond | 3.00 | 1.78 | 0.98 | 0.00 | 0.11 |  |  |  |
| Benzene | 2.42 | 1.52 | 0.90 | 0.34 | 0.09 | 0.00 |  |  |
| SG+Wheel | 5.48 | 4.13 | 3.15 | 1.87 | 1.35 | 0.67 | 0.00 | 0.00 |
| FC | 2.87 | 1.76 | 0.83 | 0.00 | 0.09 |  |  |  |

### ReFine (T=70 default)

| Dataset | λ=1 | λ=2 | λ=3 | λ=4 | λ=5 | λ=6 | λ=7 | λ=8 |
|---|---|---|---|---|---|---|---|---|
| SG+House | 7.06 | 4.94 | 3.14 | 1.02 | 0.00 | 0.00 |  |  |
| SG+Diamond | 3.73 | 2.51 | 1.22 | 0.00 | 0.11 |  |  |  |
| SG+Wheel | 5.91 | 4.87 | 3.83 | 2.97 | 2.11 | 0.95 | 0.00 | 0.00 |
| Benzene | 3.35 | 2.37 | 1.53 | 0.52 | 0.00 | 0.00 |  |  |
| FC | 2.92 | 1.81 | 0.82 | 0.00 | 0.09 |  |  |  |

### GSAT (T=70 default)

| Dataset | λ=1 | λ=2 | λ=3 | λ=4 | λ=5 | λ=6 | λ=7 | λ=8 |
|---|---|---|---|---|---|---|---|---|
| SG+House | 4.69 | 2.57 | 1.67 | 0.88 | 0.22 | 0.00 |  |  |
| SG+Diamond | 3.98 | 2.67 | 0.90 | 0.00 | 0.11 |  |  |  |
| SG+Wheel | 5.26 | 3.64 | 3.32 | 1.99 | 0.89 | 0.18 | 0.00 | 0.00 |
| Benzene | 3.46 | 2.33 | 1.39 | 0.61 | 0.00 | 0.00 |  |  |
| FC | 2.76 | 1.96 | 1.01 | 0.00 | 0.09 |  |  |  |