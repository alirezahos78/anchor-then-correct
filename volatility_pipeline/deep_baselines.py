from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy
import torch

from .config import ExperimentConfig, save_config
from .reporting import (BACKBONE_LABELS, REPORTED_METRICS, aggregate_seed_metrics,
                        primary_qlike_with_har, _latex_primary_qlike_with_har)
from .artifacts import write_json_atomic
from .data import derive_fixed_split_dates, prepare_data
from .experiments import CaseResult, run_case
from .models import AnchoredCorrector, n_parameters
from .statistics import adjust_pvalues, daily_qlike, dm_test





def _load_universe(path: str | Path) -> dict[str, dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {market["name"]: market for market in raw["markets"]}


def _model_counts(cfg: ExperimentConfig, case: CaseResult, backbone: str) -> dict[str, int]:
    data = case.metadata["data"]
    n_features = len(data["features"])
    n_core = len(data["core"])
    model = AnchoredCorrector(
        cfg,
        n_features,
        n_core,
        (np.zeros(n_core, dtype="float32"), 0.0),
        corrector=backbone,
        freeze_core=True,
        anchor=False,
        use_core=True,
    )
    counts = {
        "backbone_parameters": n_parameters(model.backbone),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": n_parameters(model),
    }
    del model
    return counts


def _check_alignment(residual: CaseResult, direct: CaseResult, market: str, backbone: str) -> None:
    checks = (
        (residual.validation_dates, direct.validation_dates, "validation dates"),
        (residual.test_dates, direct.test_dates, "test dates"),
        (residual.validation_true, direct.validation_true, "validation targets"),
        (residual.test_true, direct.test_true, "test targets"),
    )
    for left, right, label in checks:
        if not np.array_equal(left, right):
            raise AssertionError(f"{market}/{backbone}: paired {label} differ")
    residual_data = residual.metadata["data"]
    direct_data = direct.metadata["data"]
    for key in ("features", "sizes", "date_ranges", "corrector_iv_mode", "input_sha256", "target_standardizer"):
        if residual_data[key] != direct_data[key]:
            raise AssertionError(f"{market}/{backbone}: paired data field {key!r} differs")


def _dm_row(
    *,
    backbone: str,
    market: str,
    horizon: int,
    comparison: str,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    true: np.ndarray,
) -> dict:
    result = dm_test(daily_qlike(prediction_a, true), daily_qlike(prediction_b, true), horizon)
    method_b = "direct" if comparison == "residual_vs_direct" else "core"
    return {
        "family": f"{backbone}:{comparison}",
        "backbone": backbone,
        "backbone_label": BACKBONE_LABELS[backbone],
        "market": market,
        "comparison": comparison,
        "method_a": "residual",
        "method_b": method_b,
        "mean_loss_a_minus_b": result.mean_difference,
        "dm": result.statistic,
        "p_raw": result.pvalue,
        "n": result.n,
        "nw_lag": result.lag,
    }


def _adjust_dm(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["p_holm"] = np.nan
    frame["p_bh"] = np.nan
    for _, labels in frame.groupby("family", sort=False).groups.items():
        labels = list(labels)
        pvalues = frame.loc[labels, "p_raw"].tolist()
        frame.loc[labels, "p_holm"] = adjust_pvalues(pvalues, "holm")
        frame.loc[labels, "p_bh"] = adjust_pvalues(pvalues, "bh")
    return frame


def summarize_metrics(metrics: pd.DataFrame, dm: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for backbone, group in metrics.groupby("backbone", sort=False):
        q_direct = group["QLIKE_direct"].to_numpy(float)
        q_residual = group["QLIKE_residual"].to_numpy(float)
        q_core = group["QLIKE_core"].to_numpy(float)
        direct_mean = float(q_direct.mean())
        residual_mean = float(q_residual.mean())
        core_mean = float(q_core.mean())
        direct_tests = dm.loc[
            (dm["backbone"] == backbone) & (dm["comparison"] == "residual_vs_direct")
        ]
        core_tests = dm.loc[
            (dm["backbone"] == backbone) & (dm["comparison"] == "residual_vs_core")
        ]
        rows.append(
            {
                "backbone": backbone,
                "backbone_label": BACKBONE_LABELS[backbone],
                "n_markets": len(group),
                "parameters_residual_median": int(group["trainable_parameters_residual"].median()),
                "parameters_direct_median": int(group["trainable_parameters_direct"].median()),
                "QLIKE_core_equal_weight": core_mean,
                "QLIKE_direct_equal_weight": direct_mean,
                "QLIKE_residual_equal_weight": residual_mean,
                "direct_minus_residual_QLIKE": direct_mean - residual_mean,
                "relative_gain_vs_direct_percent": (
                    100.0 * (direct_mean - residual_mean) / direct_mean
                    if direct_mean != 0
                    else float("nan")
                ),
                "core_minus_residual_QLIKE": core_mean - residual_mean,
                "residual_wins_vs_direct": int((q_residual < q_direct).sum()),
                "residual_wins_vs_core": int((q_residual < q_core).sum()),
                "holm_significant_wins_vs_direct": int(
                    ((direct_tests["mean_loss_a_minus_b"] < 0) & (direct_tests["p_holm"] < 0.05)).sum()
                ),
                "holm_significant_wins_vs_core": int(
                    ((core_tests["mean_loss_a_minus_b"] < 0) & (core_tests["p_holm"] < 0.05)).sum()
                ),
                "mean_seed_std_QLIKE_residual": float(group["seed_std_QLIKE_residual"].mean()),
                "mean_seed_std_QLIKE_direct": float(group["seed_std_QLIKE_direct"].mean()),
            }
        )
    return pd.DataFrame(rows)


def leave_one_market_out(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for backbone, group in metrics.groupby("backbone", sort=False):
        for omitted in group["market"]:
            kept = group.loc[group["market"] != omitted]
            direct = float(kept["QLIKE_direct"].mean())
            residual = float(kept["QLIKE_residual"].mean())
            core = float(kept["QLIKE_core"].mean())
            rows.append(
                {
                    "backbone": backbone,
                    "backbone_label": BACKBONE_LABELS[backbone],
                    "omitted_market": omitted,
                    "n_markets": len(kept),
                    "QLIKE_core": core,
                    "QLIKE_direct": direct,
                    "QLIKE_residual": residual,
                    "direct_minus_residual_QLIKE": direct - residual,
                    "core_minus_residual_QLIKE": core - residual,
                    "residual_beats_direct": residual < direct,
                    "residual_beats_core": residual < core,
                }
            )
    return pd.DataFrame(rows)




def _latex_table(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Backbone & Params (K) & Direct & Residual & Gain (\%) & W/D & W/C \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        label = str(row["backbone_label"]).replace("_", r"\_")
        lines.append(
            f"{label} & {row['parameters_residual_median'] / 1000:.1f} & "
            f"{row['QLIKE_direct_equal_weight']:.4f} & {row['QLIKE_residual_equal_weight']:.4f} & "
            f"{row['relative_gain_vs_direct_percent']:.1f} & "
            f"{int(row['residual_wins_vs_direct'])}/{int(row['n_markets'])} & "
            f"{int(row['residual_wins_vs_core'])}/{int(row['n_markets'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _latex_seed_mean_std_table(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Backbone & Params (K) & Direct QLIKE & Residual QLIKE & Gain (\%) \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        label = str(row["backbone_label"]).replace("_", r"\_")
        lines.append(
            f"{label} & {row['parameters_residual_median'] / 1000:.1f} & "
            f"{row['QLIKE_direct_mean']:.4f} $\\pm$ {row['QLIKE_direct_std_sample']:.4f} & "
            f"{row['QLIKE_residual_mean']:.4f} $\\pm$ {row['QLIKE_residual_std_sample']:.4f} & "
            f"{row['relative_gain_vs_direct_percent']:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)






def _source_manifest(project_root: Path, paths: Iterable[Path]) -> dict[str, str]:
    return {
        str(path.relative_to(project_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.exists()
    }


def run_deep_backbone_suite(
    cfg: ExperimentConfig,
    *,
    markets: list[str] | None = None,
    backbones: list[str] | None = None,
) -> dict[str, Path]:
    """Run every declared backbone in paired direct and anchored-residual form."""
    markets = list(markets or cfg.confirmatory_markets)
    backbones = list(backbones or cfg.correctors)
    cfg.validate()
    if cfg.train_loss != "qlike" or cfg.selection_metric != "QLIKE":
        raise ValueError("this matched protocol requires QLIKE training and validation selection")
    if not cfg.seeds or len(cfg.seeds) != len(set(cfg.seeds)):
        raise ValueError("seeds must be nonempty and distinct")
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in cfg.seeds):
        raise ValueError("seeds must be integers in [0, 2**32)")
    unknown = [name for name in backbones if name not in BACKBONE_LABELS]
    if unknown:
        raise ValueError(f"unknown deep backbones: {unknown}")
    if len(markets) != len(set(markets)) or len(backbones) != len(set(backbones)):
        raise ValueError("markets and backbones must not contain duplicates")

    universe = _load_universe(cfg.market_universe)
    missing = [market for market in markets if market not in universe]
    if missing:
        raise ValueError(f"markets missing from the frozen universe: {missing}")

    output = Path(cfg.output_dir)
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output / "completion.json", {
        "complete": False, "status": "running",
        "markets": markets, "backbones": backbones, "seeds": cfg.seeds,
    })
    save_config(cfg, output / "resolved_config.json")
    (output / "suite_definition.json").write_text(
        json.dumps(
            {
                "status": "matched_control_backbone_robustness",
                "primary_metric": "QLIKE",
                "horizon": cfg.primary_horizon,
                "markets": markets,
                "backbones": backbones,
                "seeds": cfg.seeds,
                "paired_design": {
                    "shared_encoder_features": cfg.paired_corrector_iv_mode,
                    "expected_feature_count": cfg.expected_paired_features,
                    "residual": "shared encoder input plus frozen full OLS core",
                    "direct": "same encoder input and trainable architecture, without the core path",
                    "zero_readout_initialization": "enabled identically for direct and residual",
                    "guard": False,
                    "selection": "identical validation QLIKE rule; epoch zero is eligible for both modes",
                },
                "multiplicity": "Holm and BH within each backbone/comparison family across markets",
                "reporting_rule": "all configured backbones and markets are retained",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    project_root = Path(__file__).resolve().parents[1]
    source_paths = sorted((project_root / "volatility_pipeline").glob("*.py")) + [
        project_root / "run_matched.py",
        project_root / "run_deep_baselines.py",
        project_root / "data" / "build_market_dataset.py",
        project_root / "configs" / "matched_control.json",
        project_root / "configs" / "deep_baselines.json",
        project_root / "configs" / "market_universe.json",
    ]
    (output / "source_manifest.json").write_text(
        json.dumps(_source_manifest(project_root, source_paths), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest_path = Path(cfg.data_dir) / "data_manifest.json"
    if manifest_path.exists():
        (output / "data_manifest.json").write_text(
            manifest_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

    metric_rows: list[dict] = []
    seed_rows: list[dict] = []
    dm_rows: list[dict] = []
    split_dates: dict[str, dict] = {}
    feature_manifest: dict[str, dict] = {}
    split_objects = {}

    try:
        from tqdm.auto import tqdm
    except ImportError:  # dependency preflight normally prevents this branch
        tqdm = lambda values, **_: values  # type: ignore[assignment]

    # Validate every frozen market and the shared feature contract before the
    # first GPU training starts.
    for market in tqdm(markets, desc="preflight", unit="market"):
        split = derive_fixed_split_dates(cfg, market, max(cfg.horizons), "ANCHOR")
        split_objects[market] = split
        split_dates[market] = split.as_dict()
        probe = prepare_data(
            cfg,
            market,
            cfg.primary_horizon,
            split,
            core_set="full",
            iv_column="ANCHOR",
            corrector_iv_mode=cfg.paired_corrector_iv_mode,
        )
        if cfg.expected_paired_features is not None and probe.n_features != cfg.expected_paired_features:
            raise AssertionError(
                f"{market}: expected {cfg.expected_paired_features} paired features, "
                f"got {probe.n_features}"
            )
        feature_manifest[market] = {
            "n_features": probe.n_features,
            "feature_names": probe.feature_names,
            "corrector_iv_mode": cfg.paired_corrector_iv_mode,
        }
        print(f"[features] {market}: {probe.n_features} shared encoder channels")
        del probe

    write_json_atomic(output / "paired_feature_manifest.json", feature_manifest)
    write_json_atomic(output / "split_dates.json", split_dates)

    for market in tqdm(markets, desc="markets", unit="market"):
        split = split_objects[market]
        metadata = universe[market]
        for backbone in tqdm(backbones, desc=market, unit="backbone", leave=False):
            residual = run_case(
                cfg,
                f"matched_pair_{market}_{backbone}_residual",
                market,
                cfg.primary_horizon,
                split,
                core_set="full",
                iv_column="ANCHOR",
                corrector_iv_mode=cfg.paired_corrector_iv_mode,
                corrector=backbone,
                freeze_core=True,
                anchor=True,
                use_core=True,
                zero_init_readout=cfg.matched_zero_init_readout,
                use_guard=False,
            )
            direct = run_case(
                cfg,
                f"matched_pair_{market}_{backbone}_direct",
                market,
                cfg.primary_horizon,
                split,
                core_set="full",
                iv_column="ANCHOR",
                corrector_iv_mode=cfg.paired_corrector_iv_mode,
                corrector=backbone,
                freeze_core=True,
                anchor=False,
                use_core=False,
                zero_init_readout=cfg.matched_zero_init_readout,
                use_guard=False,
            )
            _check_alignment(residual, direct, market, backbone)
            residual_hashes = {
                row["seed"]: row["initial_trainable_hash"] for row in residual.seed_metrics
            }
            direct_hashes = {
                row["seed"]: row["initial_trainable_hash"] for row in direct.seed_metrics
            }
            if residual_hashes != direct_hashes:
                raise AssertionError(
                    f"{market}/{backbone}: paired trainable initializations differ"
                )
            if set(residual_hashes) != set(cfg.seeds):
                raise AssertionError(f"{market}/{backbone}: missing configured seed results")
            features = residual.metadata["data"]["features"]
            if cfg.expected_paired_features is not None and len(features) != cfg.expected_paired_features:
                raise AssertionError(
                    f"{market}: expected {cfg.expected_paired_features} paired features, got {len(features)}"
                )
            if features != feature_manifest[market]["feature_names"]:
                raise AssertionError(f"{market}/{backbone}: features changed after preflight")
            residual_counts = _model_counts(cfg, residual, backbone)
            direct_counts = _model_counts(cfg, direct, backbone)
            for count_name in ("backbone_parameters", "trainable_parameters", "total_parameters"):
                if residual_counts[count_name] != direct_counts[count_name]:
                    raise AssertionError(
                        f"{market}/{backbone}: paired {count_name} differs "
                        f"({residual_counts[count_name]} vs {direct_counts[count_name]})"
                    )
            target_residual = _model_counts(cfg, residual, "bimamba")["backbone_parameters"]
            target_direct = _model_counts(cfg, direct, "bimamba")["backbone_parameters"]

            row = {
                "market": market,
                "role": metadata.get("role"),
                "family": metadata.get("family"),
                "anchor": metadata.get("anchor_name"),
                "backbone": backbone,
                "backbone_label": BACKBONE_LABELS[backbone],
                "horizon": cfg.primary_horizon,
                "n_test": len(residual.test_true),
                **{f"{key}_core": value for key, value in residual.test_metrics["core_ols"].items()},
                **{f"{key}_har": value for key, value in residual.test_metrics["har"].items()},
                **{f"{key}_garch": value for key, value in residual.test_metrics["garch"].items()},
                **{f"{key}_direct": value for key, value in direct.test_metrics["hybrid"].items()},
                **{f"{key}_residual": value for key, value in residual.test_metrics["hybrid"].items()},
                **{f"seed_mean_{key}_direct": value for key, value in direct.seed_mean.items()},
                **{f"seed_std_{key}_direct": value for key, value in direct.seed_std.items()},
                **{f"seed_mean_{key}_residual": value for key, value in residual.seed_mean.items()},
                **{f"seed_std_{key}_residual": value for key, value in residual.seed_std.items()},
                **{f"{key}_residual": value for key, value in residual_counts.items()},
                **{f"{key}_direct": value for key, value in direct_counts.items()},
                "bimamba_target_parameters_residual": target_residual,
                "bimamba_target_parameters_direct": target_direct,
                "parameter_ratio_residual": residual_counts["backbone_parameters"] / target_residual,
                "parameter_ratio_direct": direct_counts["backbone_parameters"] / target_direct,
            }
            row["direct_minus_residual_QLIKE"] = row["QLIKE_direct"] - row["QLIKE_residual"]
            row["core_minus_residual_QLIKE"] = row["QLIKE_core"] - row["QLIKE_residual"]
            metric_rows.append(row)

            for mode, case, counts in (
                ("residual", residual, residual_counts),
                ("direct", direct, direct_counts),
            ):
                for seed_metric in case.seed_metrics:
                    seed_rows.append(
                        {
                            "market": market,
                            "role": metadata.get("role"),
                            "family": metadata.get("family"),
                            "anchor": metadata.get("anchor_name"),
                            "backbone": backbone,
                            "backbone_label": BACKBONE_LABELS[backbone],
                            "mode": mode,
                            "horizon": cfg.primary_horizon,
                            "n_test": len(case.test_true),
                            "n_features": len(case.metadata["data"]["features"]),
                            "corrector_iv_mode": case.metadata["data"]["corrector_iv_mode"],
                            "zero_init_readout": bool(case.metadata["zero_init_readout"]),
                            **counts,
                            **seed_metric,
                        }
                    )

            dm_rows.extend(
                [
                    _dm_row(
                        backbone=backbone,
                        market=market,
                        horizon=cfg.primary_horizon,
                        comparison="residual_vs_direct",
                        prediction_a=residual.test_predictions["hybrid"],
                        prediction_b=direct.test_predictions["hybrid"],
                        true=residual.test_true,
                    ),
                    _dm_row(
                        backbone=backbone,
                        market=market,
                        horizon=cfg.primary_horizon,
                        comparison="residual_vs_core",
                        prediction_a=residual.test_predictions["hybrid"],
                        prediction_b=residual.test_predictions["core_ols"],
                        true=residual.test_true,
                    ),
                ]
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    metrics = pd.DataFrame(metric_rows)
    dm = _adjust_dm(dm_rows)
    summary = summarize_metrics(metrics, dm)
    lomo = leave_one_market_out(metrics)
    seeds = pd.DataFrame(seed_rows)
    expected_runs = len(markets) * len(backbones) * 2 * len(cfg.seeds)
    if len(seeds) != expected_runs or seeds.duplicated(["market", "backbone", "mode", "seed"]).any():
        raise AssertionError("seed-level reporting grid is incomplete or duplicated")
    if not np.isfinite(seeds["test_QLIKE"].to_numpy()).all():
        raise AssertionError("non-finite QLIKE in the seed-level output")
    seed_aggregates = aggregate_seed_metrics(seeds, summary)
    primary_backbone = cfg.confirmatory_corrector if cfg.confirmatory_corrector in backbones else backbones[0]
    primary_table = primary_qlike_with_har(metrics, primary_backbone, markets)
    checkpoint_audit = seeds[
        [
            "market",
            "backbone",
            "mode",
            "seed",
            "n_features",
            "zero_init_readout",
            "initial_trainable_hash",
            "initial_validation_QLIKE",
            "validation_QLIKE",
            "best_epoch",
            "selected_epoch_zero",
        ]
    ].copy()

    paths = {
        "metrics": tables / "deep_backbone_market_metrics.csv",
        "seed_metrics": tables / "deep_backbone_seed_metrics.csv",
        "dm_tests": tables / "deep_backbone_dm_tests.csv",
        "summary": tables / "deep_backbone_summary.csv",
        "leave_one_out": tables / "deep_backbone_leave_one_market_out.csv",
        "latex": tables / "paper_deep_backbones.tex",
        "market_seed_mean_std": tables / "deep_backbone_market_seed_mean_std.csv",
        "equal_weight_by_seed": tables / "deep_backbone_equal_weight_by_seed.csv",
        "equal_weight_mean_std": tables / "deep_backbone_equal_weight_mean_std.csv",
        "paired_seed_mean_std": tables / "deep_backbone_paired_seed_mean_std.csv",
        "latex_seed_mean_std": tables / "paper_deep_backbones_seed_mean_std.tex",
        "primary_qlike_with_har": tables / "paper_primary_qlike_with_har.csv",
        "latex_primary_qlike_with_har": tables / "paper_primary_qlike_with_har.tex",
        "checkpoint_audit": tables / "matched_checkpoint_audit.csv",
    }
    metrics.to_csv(paths["metrics"], index=False)
    seeds.to_csv(paths["seed_metrics"], index=False)
    dm.to_csv(paths["dm_tests"], index=False)
    summary.to_csv(paths["summary"], index=False)
    lomo.to_csv(paths["leave_one_out"], index=False)
    paths["latex"].write_text(_latex_table(summary), encoding="utf-8")
    seed_aggregates["market_mean_std"].to_csv(paths["market_seed_mean_std"], index=False)
    seed_aggregates["equal_weight_by_seed"].to_csv(paths["equal_weight_by_seed"], index=False)
    seed_aggregates["backbone_mode_mean_std"].to_csv(
        paths["equal_weight_mean_std"], index=False
    )
    seed_aggregates["paired_mean_std"].to_csv(paths["paired_seed_mean_std"], index=False)
    paths["latex_seed_mean_std"].write_text(
        _latex_seed_mean_std_table(seed_aggregates["paired_mean_std"]), encoding="utf-8"
    )
    primary_table.to_csv(paths["primary_qlike_with_har"], index=False)
    paths["latex_primary_qlike_with_har"].write_text(
        _latex_primary_qlike_with_har(primary_table), encoding="utf-8"
    )
    checkpoint_audit.to_csv(paths["checkpoint_audit"], index=False)
    (output / "split_dates.json").write_text(json.dumps(split_dates, indent=2), encoding="utf-8")
    (output / "paired_feature_manifest.json").write_text(
        json.dumps(feature_manifest, indent=2), encoding="utf-8"
    )

    completion = {
        "complete": len(metrics) == len(markets) * len(backbones) and len(seeds) == expected_runs,
        "expected_market_backbone_pairs": len(markets) * len(backbones),
        "observed_market_backbone_pairs": len(metrics),
        "expected_training_runs": expected_runs,
        "observed_training_runs": len(seeds),
        "seeds": cfg.seeds,
        "primary_table_backbone": primary_backbone,
        "primary_table_estimator": "QLIKE of mean log-volatility prediction across seeds",
        "all_markets": markets,
        "all_backbones": backbones,
        "matched_feature_count_by_market": {
            market: feature_manifest[market]["n_features"] for market in markets
        },
        "zero_initialized_both_modes": cfg.matched_zero_init_readout,
        "epoch_zero_candidate_both_modes": True,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    write_json_atomic(output / "completion.json", completion)
    if not completion["complete"]:
        raise AssertionError("deep-backbone suite did not produce every configured pair")
    return paths
