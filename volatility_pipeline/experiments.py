from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy
import torch
from scipy import stats

from .config import ExperimentConfig, save_config
from .artifacts import file_sha256, write_json_atomic, write_npz_atomic
from .data import (
    IV_FEATURES,
    SplitDates,
    baseline_predictions,
    derive_fixed_split_dates,
    derive_rolling_splits,
    prepare_data,
    split_summary,
)
from .statistics import adjust_pvalues, daily_qlike, dm_test, fold_aware_dm
from .training import run_ensemble, volatility_metrics


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return str(value)
    raise TypeError(type(value).__name__)


@dataclass
class CaseResult:
    case_id: str
    index_name: str
    horizon: int
    validation_dates: np.ndarray
    test_dates: np.ndarray
    validation_true: np.ndarray
    test_true: np.ndarray
    validation_predictions: dict[str, np.ndarray]
    test_predictions: dict[str, np.ndarray]
    validation_metrics: dict[str, dict[str, float]]
    test_metrics: dict[str, dict[str, float]]
    seed_mean: dict[str, float]
    seed_std: dict[str, float]
    seed_metrics: list[dict]
    guard_applied: bool
    metadata: dict


def _case_signature(cfg: ExperimentConfig, payload: dict) -> str:
    relevant = cfg.to_dict().copy()
    relevant.pop("force", None)
    relevant.pop("output_dir", None)
    source_dir = Path(__file__).resolve().parent
    sources = {path.name: file_sha256(path) for path in sorted(source_dir.glob("*.py"))}
    data_path = Path(cfg.data_dir) / f"{payload['index']}_vol.csv"
    raw = json.dumps({"config": relevant, "case": payload,
                      "data_sha256": file_sha256(data_path), "sources": sources},
                     sort_keys=True, default=_jsonable).encode()
    return hashlib.sha256(raw).hexdigest()


def _save_case(case_dir: Path, signature: str, case: CaseResult) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    methods = sorted(case.test_predictions)
    method_keys = {method: f"m{number:03d}" for number, method in enumerate(methods)}
    arrays = {
        "validation_dates": case.validation_dates.astype("datetime64[ns]"),
        "test_dates": case.test_dates.astype("datetime64[ns]"),
        "validation_true": case.validation_true,
        "test_true": case.test_true,
    }
    for method, key in method_keys.items():
        arrays[f"val_{key}"] = case.validation_predictions[method]
        arrays[f"test_{key}"] = case.test_predictions[method]
    write_npz_atomic(case_dir / "predictions.npz", **arrays)
    payload = {
        "signature": signature,
        "predictions_sha256": file_sha256(case_dir / "predictions.npz"),
        "case_id": case.case_id,
        "index_name": case.index_name,
        "horizon": case.horizon,
        "method_keys": method_keys,
        "validation_metrics": case.validation_metrics,
        "test_metrics": case.test_metrics,
        "seed_mean": case.seed_mean,
        "seed_std": case.seed_std,
        "seed_metrics": case.seed_metrics,
        "guard_applied": case.guard_applied,
        "metadata": case.metadata,
    }
    write_json_atomic(case_dir / "result.json", payload, sort_keys=True, default=_jsonable)


def _load_case(case_dir: Path, signature: str) -> CaseResult | None:
    result_path, prediction_path = case_dir / "result.json", case_dir / "predictions.npz"
    if not result_path.exists() or not prediction_path.exists():
        return None
    try:
        with result_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return None
    if payload.get("signature") != signature:
        return None
    if file_sha256(prediction_path) != payload.get("predictions_sha256"):
        return None
    with np.load(prediction_path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    val_predictions, test_predictions = {}, {}
    for method, key in payload["method_keys"].items():
        val_predictions[method] = arrays[f"val_{key}"]
        test_predictions[method] = arrays[f"test_{key}"]
    return CaseResult(
        case_id=payload["case_id"],
        index_name=payload["index_name"],
        horizon=int(payload["horizon"]),
        validation_dates=arrays["validation_dates"],
        test_dates=arrays["test_dates"],
        validation_true=arrays["validation_true"],
        test_true=arrays["test_true"],
        validation_predictions=val_predictions,
        test_predictions=test_predictions,
        validation_metrics=payload["validation_metrics"],
        test_metrics=payload["test_metrics"],
        seed_mean=payload["seed_mean"],
        seed_std=payload["seed_std"],
        seed_metrics=payload.get("seed_metrics", []),
        guard_applied=bool(payload["guard_applied"]),
        metadata=payload["metadata"],
    )


def run_case(
    cfg: ExperimentConfig,
    case_id: str,
    index_name: str,
    horizon: int,
    split_dates: SplitDates,
    *,
    core_set: str = "full",
    iv_column: str = "VIX",
    corrector_iv_mode: str = "none",
    corrector: str = "bimamba",
    freeze_core: bool = True,
    anchor: bool = True,
    use_core: bool = True,
    n_scans: int | None = None,
    zero_init_readout: bool | None = None,
    use_guard: bool | None = None,
) -> CaseResult:
    payload = {
        "case_id": case_id,
        "index": index_name,
        "horizon": horizon,
        "split": split_dates.as_dict(),
        "core_set": core_set,
        "iv_column": iv_column,
        "corrector_iv_mode": corrector_iv_mode,
        "corrector": corrector,
        "freeze_core": freeze_core,
        "anchor": anchor,
        "use_core": use_core,
        "n_scans": n_scans,
        "zero_init_readout": zero_init_readout,
        "use_guard": use_guard,
    }
    signature = _case_signature(cfg, payload)
    case_dir = Path(cfg.output_dir) / "cases" / _slug(case_id)
    if not cfg.force:
        cached = _load_case(case_dir, signature)
        if cached is not None:
            print(f"[cached] {case_id}")
            return cached

    print(f"[run] {case_id}")
    bundle = prepare_data(
        cfg,
        index_name,
        horizon,
        split_dates,
        core_set=core_set,
        iv_column=iv_column,
        corrector_iv_mode=corrector_iv_mode,
    )
    if corrector_iv_mode == "none" and IV_FEATURES.intersection(bundle.feature_names):
        raise AssertionError("no-IV corrector still contains an implied-volatility feature")
    ensemble = run_ensemble(
        cfg,
        bundle,
        corrector=corrector,
        freeze_core=freeze_core,
        anchor=anchor,
        use_core=use_core,
        n_scans=n_scans,
        zero_init_readout=zero_init_readout,
        artifact_dir=case_dir / "checkpoints",
        cache_signature=signature,
        use_guard=use_guard,
    )

    validation_predictions = {
        "hybrid": ensemble.validation_prediction,
        "core_ols": bundle.core_logrv["val"],
        **baseline_predictions(bundle, "val"),
    }
    test_predictions = {
        "hybrid": ensemble.test_prediction,
        "core_ols": bundle.core_logrv["test"],
        **baseline_predictions(bundle, "test"),
    }
    if not all(len(pred) == len(ensemble.test_true) for pred in test_predictions.values()):
        raise AssertionError("baseline/model test predictions are not aligned")
    case = CaseResult(
        case_id=case_id,
        index_name=index_name,
        horizon=horizon,
        validation_dates=bundle.dates["val"],
        test_dates=bundle.dates["test"],
        validation_true=ensemble.validation_true,
        test_true=ensemble.test_true,
        validation_predictions=validation_predictions,
        test_predictions=test_predictions,
        validation_metrics={
            method: volatility_metrics(prediction, ensemble.validation_true)
            for method, prediction in validation_predictions.items()
        },
        test_metrics={
            method: volatility_metrics(prediction, ensemble.test_true)
            for method, prediction in test_predictions.items()
        },
        seed_mean=ensemble.seed_metrics_mean,
        seed_std=ensemble.seed_metrics_std,
        seed_metrics=[
            {
                "seed": result.seed,
                "initial_trainable_hash": result.initial_trainable_hash,
                "best_epoch": result.best_epoch,
                "selected_epoch_zero": result.best_epoch == 0,
                **{f"initial_validation_{key}": value for key, value in result.initial_validation.items()},
                **{f"validation_{key}": value for key, value in result.validation.items()},
                **{f"test_{key}": value for key, value in result.test.items()},
            }
            for result in ensemble.seed_results
        ],
        guard_applied=ensemble.guard_applied,
        metadata={**payload, "data": split_summary(bundle)},
    )
    _save_case(case_dir, signature, case)
    return case


def _metric_rows(case: CaseResult, section: str, methods: Iterable[str] | None = None) -> list[dict]:
    methods = list(methods or case.test_metrics.keys())
    rows = []
    for method in methods:
        if method not in case.test_metrics:
            continue
        row = {
            "section": section,
            "case_id": case.case_id,
            "index": case.index_name,
            "horizon": case.horizon,
            "method": method,
            **case.test_metrics[method],
        }
        if method == "hybrid":
            row.update({f"seed_mean_{key}": value for key, value in case.seed_mean.items()})
            row.update({f"seed_std_{key}": value for key, value in case.seed_std.items()})
            row["guard_applied"] = case.guard_applied
        rows.append(row)
    return rows


def _dm_row(case: CaseResult, method_a: str, method_b: str, family: str) -> dict:
    qa = daily_qlike(case.test_predictions[method_a], case.test_true)
    qb = daily_qlike(case.test_predictions[method_b], case.test_true)
    result = dm_test(qa, qb, case.horizon)
    return {
        "family": family,
        "case_id": case.case_id,
        "index": case.index_name,
        "horizon": case.horizon,
        "method_a": method_a,
        "method_b": method_b,
        "mean_loss_a_minus_b": result.mean_difference,
        "dm": result.statistic,
        "p_raw": result.pvalue,
        "n": result.n,
        "nw_lag": result.lag,
    }


def _adjust_families(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["p_holm"] = np.nan
    frame["p_bh"] = np.nan
    for _, indices in frame.groupby("family").groups.items():
        positions = list(indices)
        pvalues = frame.loc[positions, "p_raw"].tolist()
        frame.loc[positions, "p_holm"] = adjust_pvalues(pvalues, "holm")
        frame.loc[positions, "p_bh"] = adjust_pvalues(pvalues, "bh")
    return frame


def _common_origin_view(cases: list[CaseResult]) -> list[tuple[CaseResult, np.ndarray, dict[str, np.ndarray]]]:
    common = set(pd.to_datetime(cases[0].test_dates))
    for case in cases[1:]:
        common &= set(pd.to_datetime(case.test_dates))
    common_dates = np.array(sorted(common), dtype="datetime64[ns]")
    if not len(common_dates):
        raise AssertionError("horizon cases have no common forecast origins")
    views = []
    for case in cases:
        lookup = {date: pos for pos, date in enumerate(case.test_dates.astype("datetime64[ns]"))}
        positions = np.array([lookup[date] for date in common_dates])
        predictions = {method: values[positions] for method, values in case.test_predictions.items()}
        views.append((case, case.test_true[positions], predictions))
    return views


def _write_table(rows: list[dict] | pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def run_all_experiments(cfg: ExperimentConfig, sections: set[str] | None = None) -> dict[str, Path]:
    sections = sections or {"main", "core", "architecture", "design", "vxn", "horizons", "rolling"}
    output = Path(cfg.output_dir)
    tables = output / "tables"
    output.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output / "resolved_config.json")
    manifest = Path(cfg.data_dir) / "data_manifest.json"
    if manifest.exists():
        shutil.copy2(manifest, output / "data_manifest.json")
    with (output / "environment.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            },
            fh,
            indent=2,
        )

    fixed_splits = {
        index: derive_fixed_split_dates(cfg, index, max(cfg.horizons), "VIX") for index in cfg.indices
    }
    with (output / "split_dates.json").open("w", encoding="utf-8") as fh:
        json.dump({index: split.as_dict() for index, split in fixed_splits.items()}, fh, indent=2)

    generated: dict[str, Path] = {}
    primary_cases: dict[str, CaseResult] = {}
    all_cases: dict[str, CaseResult] = {}
    mismatch_cases: dict[str, CaseResult] = {}
    dm_rows: list[dict] = []

    if "main" in sections:
        rows = []
        for index in cfg.indices:
            case = run_case(
                cfg,
                f"main_{index}_h{cfg.primary_horizon}_noiv",
                index,
                cfg.primary_horizon,
                fixed_splits[index],
                core_set="full",
                iv_column="VIX",
                corrector_iv_mode="none",
            )
            primary_cases[index] = case
            all_cases[case.case_id] = case
            mismatch_cases[case.case_id] = case
            rows.extend(_metric_rows(case, "main"))
            dm_rows.append(_dm_row(case, "hybrid", "core_ols", "primary"))
            if "vix_only" in case.test_predictions:
                dm_rows.append(_dm_row(case, "hybrid", "vix_only", "primary"))
        path = tables / "main_metrics.csv"
        _write_table(rows, path)
        generated["main"] = path

    if "core" in sections:
        rows = []
        for index in cfg.indices:
            for core_set in cfg.core_sets:
                if core_set == "full" and index in primary_cases:
                    case = primary_cases[index]
                else:
                    case = run_case(
                        cfg,
                        f"core_{index}_{core_set}",
                        index,
                        cfg.primary_horizon,
                        fixed_splits[index],
                        core_set=core_set,
                        corrector_iv_mode="none",
                    )
                all_cases[case.case_id] = case
                mismatch_cases[case.case_id] = case
                rows.extend(_metric_rows(case, "core_composition", ["hybrid", "core_ols"]))
                dm_rows.append(_dm_row(case, "hybrid", "core_ols", "core_composition"))
        path = tables / "core_composition.csv"
        _write_table(rows, path)
        generated["core"] = path

    if "architecture" in sections:
        rows, architecture_cases = [], {}
        for index in cfg.indices:
            for corrector in cfg.correctors:
                if corrector == "bimamba" and index in primary_cases:
                    case = primary_cases[index]
                else:
                    case = run_case(
                        cfg,
                        f"architecture_{index}_{corrector}",
                        index,
                        cfg.primary_horizon,
                        fixed_splits[index],
                        corrector=corrector,
                        corrector_iv_mode="none",
                    )
                architecture_cases[(index, corrector)] = case
                all_cases[case.case_id] = case
                rows.extend(_metric_rows(case, "architecture", ["hybrid"]))
        # Architecture p-values are one explicitly labeled exploratory family.
        for index in cfg.indices:
            reference = architecture_cases[(index, "bimamba")]
            for corrector in cfg.correctors:
                if corrector == "bimamba":
                    continue
                challenger = architecture_cases[(index, corrector)]
                qa = daily_qlike(challenger.test_predictions["hybrid"], challenger.test_true)
                qb = daily_qlike(reference.test_predictions["hybrid"], reference.test_true)
                dm = dm_test(qa, qb, cfg.primary_horizon)
                dm_rows.append(
                    {
                        "family": "architecture_exploratory",
                        "case_id": challenger.case_id,
                        "index": index,
                        "horizon": cfg.primary_horizon,
                        "method_a": corrector,
                        "method_b": "bimamba",
                        "mean_loss_a_minus_b": dm.mean_difference,
                        "dm": dm.statistic,
                        "p_raw": dm.pvalue,
                        "n": dm.n,
                        "nw_lag": dm.lag,
                    }
                )
        path = tables / "architecture_ablation.csv"
        _write_table(rows, path)
        generated["architecture"] = path

    if "design" in sections:
        index = "IXIC" if "IXIC" in cfg.indices else cfg.indices[0]
        cases = {}
        cases["anchored_frozen"] = run_case(
            cfg, "design_anchored_frozen", index, cfg.primary_horizon, fixed_splits[index], use_guard=False
        )
        cases["anchored_joint"] = run_case(
            cfg,
            "design_anchored_joint",
            index,
            cfg.primary_horizon,
            fixed_splits[index],
            freeze_core=False,
            use_guard=False,
        )
        cases["no_anchor"] = run_case(
            cfg,
            "design_no_anchor",
            index,
            cfg.primary_horizon,
            fixed_splits[index],
            anchor=False,
            use_guard=False,
        )
        cases["uni_scan"] = run_case(
            cfg,
            "design_uni_scan",
            index,
            cfg.primary_horizon,
            fixed_splits[index],
            n_scans=1,
            use_guard=False,
        )
        rows = []
        for label, case in cases.items():
            row = _metric_rows(case, "design", ["hybrid"])[0]
            row["design"] = label
            rows.append(row)
            all_cases[case.case_id] = case
        reference = cases["anchored_frozen"]
        for label, case in cases.items():
            if label == "anchored_frozen":
                continue
            qa = daily_qlike(reference.test_predictions["hybrid"], reference.test_true)
            qb = daily_qlike(case.test_predictions["hybrid"], case.test_true)
            result = dm_test(qa, qb, cfg.primary_horizon)
            dm_rows.append(
                {
                    "family": "design_confirmatory",
                    "case_id": case.case_id,
                    "index": index,
                    "horizon": cfg.primary_horizon,
                    "method_a": "anchored_frozen",
                    "method_b": label,
                    "mean_loss_a_minus_b": result.mean_difference,
                    "dm": result.statistic,
                    "p_raw": result.pvalue,
                    "n": result.n,
                    "nw_lag": result.lag,
                }
            )
        path = tables / "design_ablation.csv"
        _write_table(rows, path)
        generated["design"] = path

    if "vxn" in sections:
        index = "IXIC" if "IXIC" in cfg.indices else cfg.indices[0]
        cases = {}
        for mode in ("none", "all"):
            for iv in cfg.iv_columns:
                key = f"{iv}_{mode}"
                if mode == "none" and iv == "VIX" and index in primary_cases:
                    cases[key] = primary_cases[index]
                else:
                    cases[key] = run_case(
                        cfg,
                        f"iv_{index}_{iv.lower()}_{mode}",
                        index,
                        cfg.primary_horizon,
                        fixed_splits[index],
                        iv_column=iv,
                        corrector_iv_mode=mode,
                    )
                all_cases[cases[key].case_id] = cases[key]
        rows = []
        for label, case in cases.items():
            row = _metric_rows(case, "iv_robustness", ["hybrid", "core_ols", f"{case.metadata['iv_column'].lower()}_only"])
            for item in row:
                item["iv_experiment"] = label
            rows.extend(row)
            dm_rows.append(_dm_row(case, "hybrid", "core_ols", "iv_robustness"))
            dm_rows.append(
                _dm_row(case, "hybrid", f"{case.metadata['iv_column'].lower()}_only", "iv_robustness")
            )
        for mode in ("none", "all"):
            vix_case, vxn_case = cases[f"VIX_{mode}"], cases[f"VXN_{mode}"]
            common = np.intersect1d(
                vix_case.test_dates.astype("datetime64[ns]"), vxn_case.test_dates.astype("datetime64[ns]")
            )
            vix_lookup = {date: pos for pos, date in enumerate(vix_case.test_dates.astype("datetime64[ns]"))}
            vxn_lookup = {date: pos for pos, date in enumerate(vxn_case.test_dates.astype("datetime64[ns]"))}
            vix_pos = np.array([vix_lookup[date] for date in common])
            vxn_pos = np.array([vxn_lookup[date] for date in common])
            true = vix_case.test_true[vix_pos]
            if not np.allclose(true, vxn_case.test_true[vxn_pos]):
                raise AssertionError("VIX/VXN common-origin targets differ")
            vix_pred = vix_case.test_predictions["hybrid"][vix_pos]
            vxn_pred = vxn_case.test_predictions["hybrid"][vxn_pos]
            for iv, prediction in (("VIX", vix_pred), ("VXN", vxn_pred)):
                rows.append(
                    {
                        "section": "iv_head_to_head_common_origins",
                        "case_id": f"iv_head_to_head_{mode}",
                        "index": index,
                        "horizon": cfg.primary_horizon,
                        "method": f"hybrid_{iv.lower()}",
                        "iv_experiment": mode,
                        "n_common_origins": len(common),
                        **volatility_metrics(prediction, true),
                    }
                )
            result = dm_test(daily_qlike(vxn_pred, true), daily_qlike(vix_pred, true), cfg.primary_horizon)
            dm_rows.append(
                {
                    "family": "iv_head_to_head",
                    "case_id": f"iv_head_to_head_{mode}",
                    "index": index,
                    "horizon": cfg.primary_horizon,
                    "method_a": f"hybrid_vxn_{mode}",
                    "method_b": f"hybrid_vix_{mode}",
                    "mean_loss_a_minus_b": result.mean_difference,
                    "dm": result.statistic,
                    "p_raw": result.pvalue,
                    "n": result.n,
                    "nw_lag": result.lag,
                }
            )
        path = tables / "iv_robustness.csv"
        _write_table(rows, path)
        generated["vxn"] = path

    horizon_cases: dict[tuple[str, int], CaseResult] = {}
    if "horizons" in sections:
        rows = []
        for index in cfg.indices:
            cases = []
            for horizon in cfg.horizons:
                if horizon == cfg.primary_horizon and index in primary_cases:
                    case = primary_cases[index]
                else:
                    case = run_case(
                        cfg,
                        f"horizon_{index}_h{horizon}",
                        index,
                        horizon,
                        fixed_splits[index],
                        corrector_iv_mode="none",
                    )
                cases.append(case)
                horizon_cases[(index, horizon)] = case
                all_cases[case.case_id] = case
                mismatch_cases[case.case_id] = case
            for case, true, predictions in _common_origin_view(cases):
                for method in ("hybrid", "core_ols", "vix_only"):
                    if method in predictions:
                        rows.append(
                            {
                                "section": "horizon_common_origins",
                                "index": index,
                                "horizon": case.horizon,
                                "method": method,
                                "n_common_origins": len(true),
                                **volatility_metrics(predictions[method], true),
                            }
                        )
        path = tables / "horizon_common_origins.csv"
        _write_table(rows, path)
        generated["horizons"] = path

    if "rolling" in sections:
        rows, pooled_rows = [], []
        for index in cfg.indices:
            hybrid_losses, core_losses, iv_losses = [], [], []
            for fold_number, split in enumerate(
                derive_rolling_splits(cfg, index, cfg.primary_horizon, "VIX"), start=1
            ):
                case = run_case(
                    cfg,
                    f"rolling_{index}_fold{fold_number}",
                    index,
                    cfg.primary_horizon,
                    split,
                    corrector_iv_mode="none",
                )
                all_cases[case.case_id] = case
                for method in ("hybrid", "core_ols", "vix_only"):
                    if method in case.test_metrics:
                        row = {
                            "section": "rolling",
                            "index": index,
                            "fold": fold_number,
                            "method": method,
                            **case.test_metrics[method],
                        }
                        rows.append(row)
                hybrid_losses.append(daily_qlike(case.test_predictions["hybrid"], case.test_true))
                core_losses.append(daily_qlike(case.test_predictions["core_ols"], case.test_true))
                iv_losses.append(daily_qlike(case.test_predictions["vix_only"], case.test_true))
            for comparison, competitor in (("core_ols", core_losses), ("vix_only", iv_losses)):
                result = fold_aware_dm(hybrid_losses, competitor, cfg.primary_horizon)
                pooled_rows.append(
                    {
                        "index": index,
                        "method_a": "hybrid",
                        "method_b": comparison,
                        "dm": result.statistic,
                        "p_raw": result.pvalue,
                        "mean_loss_a_minus_b": result.mean_difference,
                        "n": result.n,
                        "nw_lag": result.lag,
                        "note": "fold-aware NW; no covariance products across fold boundaries",
                    }
                )
        path = tables / "rolling_origin.csv"
        pooled_path = tables / "rolling_origin_fold_aware_dm.csv"
        _write_table(rows, path)
        pooled = pd.DataFrame(pooled_rows)
        pooled["p_holm"] = adjust_pvalues(pooled["p_raw"].tolist(), "holm")
        pooled["p_bh"] = adjust_pvalues(pooled["p_raw"].tolist(), "bh")
        _write_table(pooled, pooled_path)
        generated["rolling"] = path
        generated["rolling_dm"] = pooled_path

    # Core mismatch analysis directly addresses the "when" in the paper title.
    mismatch_rows = []
    for case in mismatch_cases.values():
        if "hybrid" not in case.test_metrics or "core_ols" not in case.test_metrics:
            continue
        residual = case.validation_true - case.validation_predictions["core_ols"]
        acf1 = float(np.corrcoef(residual[1:], residual[:-1])[0, 1]) if len(residual) > 2 else float("nan")
        mismatch_rows.append(
            {
                "case_id": case.case_id,
                "index": case.index_name,
                "horizon": case.horizon,
                "validation_core_qlike": case.validation_metrics["core_ols"]["QLIKE"],
                "validation_core_residual_acf1": acf1,
                "test_core_qlike": case.test_metrics["core_ols"]["QLIKE"],
                "test_hybrid_qlike": case.test_metrics["hybrid"]["QLIKE"],
                "test_hybrid_gain": case.test_metrics["core_ols"]["QLIKE"]
                - case.test_metrics["hybrid"]["QLIKE"],
                "guard_applied": case.guard_applied,
            }
        )
    mismatch = pd.DataFrame(mismatch_rows).drop_duplicates("case_id")
    mismatch_path = tables / "core_mismatch.csv"
    _write_table(mismatch, mismatch_path)
    generated["mismatch"] = mismatch_path
    if len(mismatch) >= 4:
        pearson = stats.pearsonr(mismatch["validation_core_qlike"], mismatch["test_hybrid_gain"])
        spearman = stats.spearmanr(mismatch["validation_core_qlike"], mismatch["test_hybrid_gain"])
        with (tables / "core_mismatch_summary.json").open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "n_settings": int(len(mismatch)),
                    "pearson_r": float(pearson.statistic),
                    "pearson_p": float(pearson.pvalue),
                    "spearman_rho": float(spearman.statistic),
                    "spearman_p": float(spearman.pvalue),
                    "warning": "Treat as exploratory unless settings and score were specified before viewing test gains.",
                },
                fh,
                indent=2,
            )
        generated["mismatch_summary"] = tables / "core_mismatch_summary.json"

    dm_frame = _adjust_families(dm_rows)
    dm_path = tables / "dm_tests_with_multiplicity.csv"
    _write_table(dm_frame, dm_path)
    generated["dm"] = dm_path
    return generated
