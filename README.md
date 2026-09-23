# Anchor, Then Correct

Research code for **ANCHOR, THEN CORRECT: MODEL-AGNOSTIC ECONOMETRIC–NEURAL VOLATILITY FORECASTING**.

**Authors:** Alireza Hoseinzade, Ehsan Hoseinzade, and Ali Rajaei.

The experiments compare direct sequence prediction with a frozen statistical core plus a learned residual. Paired runs share encoder inputs, trainable initialization, training budgets, and validation checkpoint selection. The residual formulation adds the fitted conditional core path.

## Quick start
 
Use your existing Python environment. From the repository root:

```bash
python run_matched.py
```

The runner reuses a valid dataset in `data/vol_dataset` or `vol_dataset`. Otherwise it builds the dataset automatically. It never installs or upgrades packages. Dependency specifications are in `requirements.txt` and `requirements_data_builder_optional.txt`; data creation requires `yfinance`.

The full default suite comprises seven markets (SPX, NDX, RUT, DIA, GLD, USO, EEM), seven backbones, direct and residual modes, and seeds 1, 2, and 3: **294 training runs**, at a five-day prediction horizon. The data cutoff is 2026-08-16, with an exclusive download end of 2026-08-17. See `configs/matched_control.json` and `configs/market_universe.json`.

For a short execution check:

```bash
python run_matched.py --smoke
```

Completed seeds are cached in `artifacts_matched_control`. Repeating the full command reuses compatible completed runs; an interrupted seed restarts. Smoke outputs are separate from the research results.

## Analysis after training

```bash
python run_posthoc_controls.py
```

This standalone script reads saved predictions and checkpoints' metadata without training or downloading data. For results stored elsewhere:

```bash
python run_posthoc_controls.py --artifacts /path/to/artifacts_matched_control
```

It evaluates the original saved forecasts and exports DM tests with multiplicity correction, per-seed metrics, correction magnitudes, and epoch-zero retention summaries. The comparisons are residual versus direct, core OLS, GARCH, and HAR. In `primary_dm.csv`, select `method_a=residual` and `method_b=garch` for the residual-versus-GARCH comparison. Forecasts and checkpoints are reused as saved. Keep the complete output to preserve all markets and comparison results.

The posthoc command creates a timestamped result folder and ZIP under `posthoc_controls`, next to the supplied training output directory. Run it after the training suite has completed.

## Hybrid baselines

Three published hybrid mechanisms are reimplemented, each keeping its own
mechanism while sharing this study's horizon, QLIKE objective, window,
channels, parameter budget, optimizer, epoch budget, seeds, and seed-averaged
reporting:

| Placement of the econometric forecast | Method | Command |
| --- | --- | --- |
| Encoder input | Hybrid LSTM (Kim and Won, 2018) | `run_hybrid_baselines.py` |
| Training loss | GINN (Xu et al., 2024) | `run_hybrid_baselines.py` |
| Post-hoc fusion | Stacked GARCH-LSTM (Peter et al., 2026) | `run_stacking_baseline.py` |
| Output (this work) | Anchor, then correct | `run_matched.py` |

Run after the training suite has completed:

```bash
python run_hybrid_baselines.py --stage verify   # seconds; checks the GARCH reference path
python run_hybrid_baselines.py                  # Hybrid LSTM and GINN, 42 trainings
python run_stacking_baseline.py                 # seconds; reuses saved forecasts
python run_table1_dm.py                         # DM tests with Holm correction
```

Defaults reproduce the paper: an LSTM encoder as in the original
implementations, GINN blend weight `lambda = 0.1`, and a linear stacking
meta-learner fitted on validation. The zero-initialized readout and the
epoch-zero checkpoint candidate belong to the anchored design and are not
given to the baselines. Numbers are reimplementations under a common protocol
and do not match the original papers, whose targets and objectives differ.

The GINN weight was selected on validation QLIKE. To reproduce the selection:

```bash
for L in 0.01 0.1 0.3 0.5 0.7; do
  python run_hybrid_baselines.py --kinds ginn --lam $L --tag lam$L
done
```

and compare `ensemble_val_QLIKE` in each `hybrid_baselines_lam*/hybrid_baselines.csv`.

`run_table1_dm.py` complements `run_posthoc_controls.py`: besides the residual
comparisons, it tests every hybrid baseline and direct learning against GARCH.
To test a tagged GINN run, pass `--baseline-dir hybrid_baselines_<tag>`.

## Repository contents

| Path | Purpose |
|---|---|
| `run_matched.py` | Main data and training entry point |
| `run_posthoc_controls.py` | Analysis of saved forecasts, including GARCH DM tests |
| `run_hybrid_baselines.py` | Input-level and loss-level hybrid baselines |
| `run_stacking_baseline.py` | Stacking hybrid baseline from saved forecasts |
| `run_table1_dm.py` | DM tests for hybrid baselines and direct learning versus GARCH |
| `volatility_pipeline/` | Data processing, models, training, statistics, and reporting |
| `configs/` | Frozen experiment settings and market definitions |
| `data/build_market_dataset.py` | Dataset builder |
| `tests/` | Existing implementation checks |

`run_deep_baselines.py` is retained as the implementation behind the preferred `run_matched.py` entry point. Historical configuration fields are preserved for run compatibility; the executed suite is documented in the runner and saved suite definition.

## Main training outputs

Outputs are written under `artifacts_matched_control`:

| File | Contents |
|---|---|
| `tables/deep_backbone_seed_metrics.csv` | Individual market, backbone, mode, and seed results |
| `tables/deep_backbone_equal_weight_by_seed.csv` | Equal-weight market average for each seed |
| `tables/deep_backbone_paired_seed_mean_std.csv` | Paired direct/residual means and sample standard deviations |
| `tables/matched_checkpoint_audit.csv` | Initial and selected validation scores, selected epoch, and epoch-zero selection |
| `tables/paper_primary_qlike_with_har.csv` | Core OLS, HAR, GARCH, direct SSM, and residual SSM ensemble scores |
| `paired_feature_manifest.json` | Shared feature names and counts |
| `suite_definition.json` | Executed experiment definition |
| `completion.json` | Suite completion record |

Each completed seed also saves predictions, metrics, training history, and model weights in its case directory. Sample standard deviations use `ddof=1` after averaging markets within each seed.

## Preserve a run

Retain the exact dataset and `data_manifest.json`, the complete `artifacts_matched_control` directory, and the posthoc analysis output. Data hashes and split dates identify the input snapshot used for the reported numbers. This repository contains code, configuration files, tests, and documentation. Datasets and result files are produced by the execution commands above.

Large datasets, checkpoints, and generated run directories are excluded from Git by `.gitignore`. Archive them separately when preparing a reproducibility release. Per-seed mean and sample SD are distinct from scores of ensemble forecasts; the reporting code exports both.

## Checks

```bash
python -m unittest discover -s tests -v
python run_posthoc_controls.py --self-test
```


## License

The existing project code license is retained in `LICENSE`. Dataset redistribution is separate from the code license.
