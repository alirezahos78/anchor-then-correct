# Anchor, Then Correct: When Deep Residuals Improve Volatility Forecasts

This repository reproduces every number reported in the paper for a two-stage volatility
forecasting model. **Stage 1** is a frozen, closed-form econometric core — an OLS combination of
HAR-RV components, implied volatility (VIX/VXN), and a rolling GARCH(1,1) forecast — fit once on
the training split and never updated. **Stage 2** is a deep sequence corrector (default backbone:
a bidirectional selective-state-space network, "BI-Mamba") whose output head is **zero-initialized**
so that at the start of training the hybrid model is numerically identical to the core alone; the
corrector can only ever add to the core's forecast, and gradient descent decides how much of a
correction, if any, is worth making. This design means the deep model can never do worse than its
own econometric baseline at initialization, and any measured improvement is attributable to the
residual the core does not already explain. All backbones (BI-Mamba, LSTM, GRU, PatchTST-style,
TSMixer-style, a windowed MLP, xLSTM/sLSTM-style, iTransformer-style) are evaluated in this same
hybrid harness and are additionally parameter-matched to BI-Mamba via binary search over their
width, so that comparisons across correctors are not confounded by capacity.

## Reproduction map

All notebooks share an identical foundational setup (cells "0 — Setup" through "5 — Train ·
multi-seed · ablation harness": data pipeline with embargoed splits, model classes, metrics,
training harness) and differ only in which experiment cells follow. Runtimes are wall-clock on a
single consumer GPU (RTX-4090-class); CPU-only runs are 5-15x slower depending on backbone.

| Paper item | Notebook | Section(s) | Approx. GPU runtime |
|---|---|---|---|
| Table I (fixed-split metrics, all baselines + Hybrid) | `notebooks/01_main_ablation_tables_I_II_IV.ipynb` | "Table I / Table II / Table IV — main composition ablation" | ~15 min |
| Table II (DM: Hybrid vs. 5 baselines) | `notebooks/01_main_ablation_tables_I_II_IV.ipynb` | "DM (purged) — Table II pairs ... + Table IV pairs" | (included above) |
| Table IV (corrector-backbone ablation + DM vs. Hybrid(BI-Mamba)) | `notebooks/01_main_ablation_tables_I_II_IV.ipynb` | same DM section, plus "Table IV extension — xLSTM-style and iTransformer-style correctors" | ~45-60 min total for the whole notebook (7 correctors x 3 indices x 3 seeds, plus the 2 extension backbones) |
| Supplementary: LSTM/GRU vs. core OLS (full), 3 indices | `notebooks/01_main_ablation_tables_I_II_IV.ipynb` | "Supplementary — LSTM / GRU vs core OLS (full), all 3 indices" | ~15 min |
| Supplementary: PatchTST-style/TSMixer-style vs. core OLS (full), IXIC + reproduction check | `notebooks/01_main_ablation_tables_I_II_IV.ipynb` | "Supplementary — PatchTST-style / TSMixer-style vs core OLS (full), IXIC" | ~5 min |
| Supplementary: raw (unanchored) PatchTF/TSMixer vs. VIX-only | `notebooks/01_main_ablation_tables_I_II_IV.ipynb` | "Supplementary — raw (unanchored) corrector baselines" | ~10 min |
| Supplementary: seed-stability of the iTransformer exception (IXIC, seeds 1-6) | `notebooks/01_main_ablation_tables_I_II_IV.ipynb` | "Supplementary — seed-stability check (iTransformer-style vs. BI-Mamba, IXIC)" | ~10 min |
| Mechanism §IV-B(i): core-composition ablation (IXIC only) | `notebooks/02_core_composition_mechanism.ipynb` | "Core-composition ablation and DM vs. own core OLS" | ~15 min |
| Design ablations: freeze vs. joint core, no-anchor control, parameter-matched uni- vs. bidirectional scan (IXIC only) | `notebooks/02_core_composition_mechanism.ipynb` | "Design ablations — freeze vs. joint core, no-anchor control, parameter-matched uni- vs bidirectional scan" | ~15 min |
| Table III (rolling-origin, K=3 folds) | `notebooks/03_rolling_origin_table_III.ipynb` | "Rolling-origin folds, pooled DM, and best-or-tied-best summary" | ~25 min |
| Table V (horizon sensitivity, h in {1,5,10,21}) | `notebooks/04_horizon_sensitivity_table_V.ipynb` | "Horizon sweep, per-horizon DM, and compact tables" | ~35 min |
| VXN section (is the NASDAQ win driven by VIX or by NASDAQ's own implied vol?) | `notebooks/05_vxn_robustness.ipynb` | "VIX-core vs. VXN-core, and the decisive DM pairs" | ~10 min |
| Fig. 1 (architecture diagram) | *drawn directly in TikZ in the paper source* — no notebook produces it | — | — |
| Fig. 2 (proxy-invariance bar chart: VIX-only / VXN-only / Hyb(VIX) / Hyb(VXN) QLIKE) | *drawn directly in pgfplots in the paper source*, plotting values computed in `notebooks/05_vxn_robustness.ipynb` | "VIX-core vs. VXN-core, and the decisive DM pairs" (same section as the VXN section row above; values also in `results/vxn_section_qlike.csv`) | ~10 min (same run as the VXN section) |

`results/` contains one CSV per table above with the exact numbers the paper reports (see
"results/ contents" below) so a reproduction can be diffed without reading notebook outputs.

## Setup

```bash
git clone <this-repository-url> anchor-then-correct && cd anchor-then-correct
python -m venv venv && source venv/bin/activate      # Python 3.10+ recommended
pip install -r requirements.txt
```

`requirements.txt` installs a CPU-only `torch` wheel by default; see the comment at the top of the
file for the CUDA install command if you have a GPU.

### Building the dataset

```bash
python data/build_vol_dataset.py
```

This downloads daily OHLCV for `^IXIC` (NASDAQ Composite), `^NYA` (NYSE Composite), `^DJI` (Dow
Jones Industrial Average), and exogenous `^VIX`, `^VXN`, `^TNX` from Yahoo Finance via `yfinance`,
starting `2010-01-01`, builds strictly causal technical/volatility features plus an h-day-ahead
(default h=5) realized-volatility target, and runs a leak self-check. It writes `IXIC_vol.csv`,
`NYA_vol.csv`, `DJI_vol.csv` into `./vol_dataset/` (its working-directory-relative default output
folder) **and**, repo-relatively (resolved from the script's own file location, regardless of the
directory you run it from), directly into `notebooks/` and `extras/` — so every notebook's first
data-loading cell finds `<INDEX>_vol.csv` sitting right next to it with no manual copying.

## Evaluation protocol

- **Split**: chronological 70% / 15% / 15% (train / val / test) per index, no shuffling.
- **Embargo**: an h-day purge is applied at both split boundaries (end of train, end of val) —
  exactly `h` boundary-adjacent samples are dropped from train and from val (never from test), so
  no training or validation target window overlaps a later split's feature window. Verified at
  runtime by a zero-overlap assertion (`train target max + h < val target min`, and similarly at
  the val/test boundary).
- **Seeds**: {1, 2, 3} for all main-paper results; seeds {4, 5, 6} additionally used only for the
  iTransformer seed-stability supplementary check.
- **Ensembling**: metrics are computed on the **seed-ensemble prediction** (the mean of the raw
  per-seed predictions, then inverse-transformed), not the mean of per-seed metrics — these differ
  because QLIKE is convex (Jensen's gap).
- **Metrics**: RMSE, MAE, QLIKE (`vp,vt = exp(2*pred), exp(2*true)`; `qlike = (vt/vp - log(vt/vp) -
  1).mean()`), R², and Pearson correlation, all computed on the un-standardized realized-vol scale.
- **Significance**: Diebold-Mariano test on the QLIKE loss differential, with HAC(h-1) autocovariance
  correction and the Harvey-Leybourne-Newbold (HLN) small-sample adjustment; two-sided p-values.
- **Table III** additionally uses K=3 rolling-origin (expanding-window) folds: train grows
  55%/70%/85% of the data, val is a fixed 5%, test a fixed 10%, per fold; DM p-values in
  `table_III_pooled_dm.csv` pool the loss differential across all three folds' test windows.

## Determinism note

All results in `results/` were produced with fixed seeds {1, 2, 3} (plus {4, 5, 6} for the one
supplementary check noted above) and were verified to reproduce exactly under repeated re-runs on
the same hardware/software stack pinned in `requirements.txt`. This has been checked in practice, not
just assumed: the LSTM/GRU and PatchTST-style/TSMixer-style backbones were independently retrained
from scratch (fresh kernel, later session) to compute their vs-core-OLS DM p-values, and every
ensemble QLIKE reproduced to 4 decimal places against the original composition-ablation run (e.g.
IXIC LSTM 0.4104, GRU 0.3933, PatchTF 0.4036, TSMixer 0.4120 — identical both times). The design
ablations (freeze/joint/no-anchor/uni-scan, IXIC) were run twice independently for the same reason
and reproduced exactly both times. Exact bit-for-bit reproduction across different hardware (CPU
vs. GPU, different GPU models) or different library versions is not guaranteed — floating-point
non-associativity in cuDNN/BLAS kernels can shift results at the last 1-2 decimal places even with
identical seeds. The qualitative conclusions (which comparisons are significant at p<0.05) were
confirmed stable across all these re-runs.

## Known limitations

- **QLIKE has no Jensen/bias correction.** The QLIKE implementation used throughout (and in
  `results/`) is the standard naive plug-in `E[vt/vp - log(vt/vp) - 1]` on point forecasts, with no
  correction for the convexity gap between forecasting the log-variance and forecasting the
  variance itself. This is standard practice in the volatility-forecasting literature but does
  slightly favor forecasts that are conservative in log-space.
- **The corrector-backbone comparisons in Table IV are not uniformly "all equivalent."** One
  exception is confirmed and reproducible: Hybrid (iTransformer-style) is significantly worse than
  Hybrid (BI-Mamba) on IXIC (original seeds {1,2,3}: p=0.006; independent seeds {4,5,6}: p=0.0019;
  pooled seeds {1..6}: p=0.0036 — see `results/supplementary_seed_stability_itransformer.csv`). No
  other corrector-vs-BI-Mamba pair reaches significance on any index.
- **Rolling-origin (Table III) significance is index-dependent.** The Hybrid model is significantly
  better than both baselines pooled across folds only on IXIC; on DJI and NYA the pooled DM tests
  do not reach significance, though Hybrid is best-or-tied-best in most individual folds (see
  `results/table_III_pooled_dm.csv`).
- **The no-anchor control does not clearly collapse.** Removing the OLS anchor entirely (core
  randomly initialized, trained jointly, IXIC) gives ensemble QLIKE 0.4227 vs. 0.3965 for the
  anchored default — numerically worse, but not statistically distinguishable from it (DM p(QLIKE)
  = 0.3925), and nowhere near the raw core-OLS floor of 0.6092. This is a milder result than a claim
  that the anchor is necessary to avoid failure; see `results/design_ablations_dm.csv`.
- **The parameter-matched uni- vs. bidirectional-scan comparison reverses the paper's claim.** An
  earlier version of this ablation used the same `d_model` for both scan directions, which gives the
  unidirectional variant only ~54% of the bidirectional backbone's parameters — not a fair
  comparison. Properly parameter-matched (d_model 32→55, 99.1% of target params), the
  **unidirectional** scan has significantly **lower** (better) squared-error loss than bidirectional
  (DM(SE)=2.27, p(SE)=0.0238); QLIKE is not significant either way (p=0.2984). This is the opposite
  of "bidirectionality helps SE specifically" — see `results/design_ablations_dm.csv` and
  `notebooks/02_core_composition_mechanism.ipynb`, "Design ablations" section.
- **No exploitable cross-market graph signal was found.** Two independent graph-based extensions
  (a static learned adjacency and a redesigned dynamic graph-attention variant, both evaluated with
  the same embargoed protocol) both converged to near-uniform/near-off graph weights on all three
  indices and showed no significant improvement. See `extras/README.md`.
- **HAR-SSM (a learnable-half-life corrector) was evaluated and not adopted** after a mis-initialization
  control showed its learned half-lives do not reliably migrate back toward the HAR time-scales when
  planted elsewhere — the "recovers interpretable time-scales from data" claim did not survive the
  control. See `extras/README.md` and `extras/har_ssm_corrector.ipynb`.
- **A pre-existing feature-dimension graph module could not be recovered.** Vestigial `Config`
  fields (`cheb_k`, `embed`, `hid`) and a code comment (`# Pure bidirectional Mamba (graph removed)`)
  indicate an earlier version of the backbone had some Chebyshev/adaptive-graph-conv-style module;
  no implementation of it survives in any notebook or backup available to us. Reported here rather
  than reconstructed. See `extras/README.md`, "A gap we could not fill."
- **Register tokens are dormant, untested code**, not an evaluated-and-rejected ablation — see
  `extras/register_tokens.py`.

## `results/` contents

One file per table/section, with the exact metric values and DM p-values the paper reports:

| File | Contents |
|---|---|
| `table_I_fixed_split_metrics.csv` | RMSE/MAE/QLIKE/R²/Corr, all baselines + Hybrid(BI-Mamba), 3 indices |
| `table_II_dm_pvalues.csv` | DM statistic + p-value, Hybrid vs. 5 baselines, 3 indices |
| `table_III_rolling_origin_qlike.csv` | Per-fold QLIKE, all rows, 3 indices |
| `table_III_pooled_dm.csv` | Pooled-across-folds DM p-values + best-or-tied-best fold count, 3 indices |
| `table_IV_composition_ablation_qlike.csv` | Ensemble QLIKE per corrector backbone (incl. xLSTM/iTransformer/core OLS), 3 indices |
| `table_IV_dm_vs_bimamba.csv` | DM p-values, each corrector vs. Hybrid(BI-Mamba), 3 indices |
| `table_IV_vs_core_completeness.csv` | DM p-values, every corrector backbone (BI-Mamba/LSTM/GRU/PatchTF/TSMixer/xLSTM/iTransformer) vs. core OLS (full), all indices actually tested |
| `table_V_horizon_sensitivity.csv` | QLIKE (VIX-only / core / Hybrid) + DM p-values at h in {1,5,10,21}, 3 indices |
| `vxn_section_qlike.csv`, `vxn_section_dm.csv` | VIX-core vs. VXN-core QLIKE and DM p-values, IXIC |
| `mechanism_core_composition.csv` | Core-composition ablation (§IV-B(i)) QLIKE + DM vs. own core, IXIC |
| `design_ablations_freeze_joint_noanchor_uniscan.csv`, `design_ablations_dm.csv` | Freeze vs. joint core, no-anchor control, parameter-matched uni- vs. bidirectional scan: ensemble QLIKE + DM p-values (QLIKE and SE), IXIC |
| `supplementary_raw_correctors.csv` | Raw (unanchored) PatchTF/TSMixer QLIKE/R² + DM vs. VIX-only, 3 indices |
| `supplementary_seed_stability_itransformer.csv` | iTransformer-vs-BI-Mamba QLIKE + DM across seed sets {1,2,3}/{4,5,6}/{1..6}, IXIC |

## License

MIT — see `LICENSE`.
