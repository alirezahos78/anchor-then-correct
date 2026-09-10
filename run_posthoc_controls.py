#!/usr/bin/env python3
"""ICASSP post-hoc analysis of original saved forecasts, version 3.0.

Put this file next to run_matched.py, then run:
    python run_posthoc_controls.py
Optional explicit path:
    python run_posthoc_controls.py --artifacts /path/to/artifacts_matched_control
Internal synthetic verification only:
    python run_posthoc_controls.py --self-test

Uses existing numpy, pandas, scipy. Never imports torch or project modules,
trains models, downloads data, installs packages, or modifies source artifacts.
Uses ALL markets/backbones/seeds declared in completion.json.
Forecasts are evaluated exactly as saved. Every configured market, backbone
and seed is retained for DM comparisons and descriptive checkpoint analysis.
New controls are post-hoc robustness analyses, not preregistered confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import tempfile
import unittest
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import scipy
    from scipy import stats
except ImportError as exc:
    raise SystemExit(
        f"Missing dependency: {exc.name}. Use the existing training environment "
        "that already has numpy, pandas and scipy. Nothing was installed."
    ) from exc

VERSION = "3.0"
CLIP = 30.0  # Same evaluation clipping as the completed matched runner.
CORE_ATOL = 2e-5  # Float32 neural output versus float64 OLS prediction.
METRIC_ATOL = 2e-6
COMPARATORS = ("direct", "core", "garch", "har")
METHODS = ("core", "har", "garch", "direct", "residual")
COMPARISON_PAIRS = tuple(("residual", m) for m in COMPARATORS)


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, data):
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(clean_json(data), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def vector(value, name):
    a = np.asarray(value, dtype=np.float64)
    require(a.ndim == 1 and len(a) > 0, f"{name}: expected a nonempty 1-D array")
    require(np.isfinite(a).all(), f"{name}: non-finite values; no rows were silently dropped")
    return a


def qlike(true, prediction):
    y, p = vector(true, "target"), vector(prediction, "prediction")
    require(y.shape == p.shape, "Prediction and target shapes differ")
    r = np.clip(2.0 * (y - p), -CLIP, CLIP)
    return np.exp(r) - r - 1.0


def clip_count(true, prediction):
    return int(np.count_nonzero(np.abs(2 * (np.asarray(true) - np.asarray(prediction))) > CLIP))





def dm_test(loss_a, loss_b, horizon):
    """Two-sided DM, Bartlett/Newey-West lag h-1 and HLN correction.

    Degenerate nonzero loss differences are explicitly non-testable. Their
    p-values count as 1 for multiplicity, preserving the declared family size.
    Exact numerical ties are reported as statistic 0, p=1.
    """
    a, b = vector(loss_a, "DM loss A"), vector(loss_b, "DM loss B")
    require(a.shape == b.shape, "DM arrays must align exactly")
    d = a - b
    n, lag = len(d), max(int(horizon) - 1, 0)
    require(n > max(10, horizon), "Insufficient observations for DM/HLN")
    mean = float(d.mean())
    base = dict(mean_loss_a_minus_b=mean, n=n, nw_lag=lag)
    if np.max(np.abs(d)) <= 1e-12:
        return dict(base, dm=0.0, p_raw=1.0, status="numerical_tie")
    centered = d - mean
    lrv = float(centered @ centered) / n
    for k in range(1, min(lag, n - 1) + 1):
        lrv += 2 * (1 - k / (lag + 1)) * float(centered[k:] @ centered[:-k]) / n
    variance = lrv / n
    hln = (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
    tolerance = np.finfo(float).eps * max(float(np.mean(d * d)), 1.0)
    if variance <= tolerance or hln <= 0:
        return dict(base, dm=float("nan"), p_raw=float("nan"), status="degenerate_variance")
    statistic = mean / np.sqrt(variance) * np.sqrt(hln)
    return dict(base, dm=float(statistic), p_raw=float(2 * stats.t.sf(abs(statistic), n - 1)), status="ok")


def adjusted(values, method):
    p = np.asarray(values, float)
    p = np.where(np.isfinite(p), p, 1.0)
    require(((p >= 0) & (p <= 1)).all(), "Invalid p-value")
    order = np.argsort(p, kind="stable")
    ranked, n = p[order], len(p)
    if method == "holm":
        result = np.maximum.accumulate(ranked * np.arange(n, 0, -1))
    elif method == "bh":
        result = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    else:
        raise ValueError(method)
    output = np.empty(n)
    output[order] = np.minimum(result, 1.0)
    return output


@dataclass
class Case:
    market: str
    backbone: str
    mode: str
    directory: Path
    meta: dict
    arrays: dict

    def true(self, split):
        return self.arrays["validation_true" if split == "val" else "test_true"]

    def dates(self, split):
        return self.arrays["validation_dates" if split == "val" else "test_dates"]

    def pred(self, split, method):
        aliases = {"residual": "hybrid", "direct": "hybrid", "core": "core_ols"}
        key = self.meta["method_keys"][aliases.get(method, method)]
        return self.arrays[f"{split}_{key}"]


def load_case(root, market, backbone, mode, horizon, seeds, provenance):
    case_id = f"matched_pair_{market}_{backbone}_{mode}"
    directory = root / "cases" / case_id
    j, p = directory / "result.json", directory / "predictions.npz"
    require(j.is_file() and p.is_file(),
        f"Missing saved case: {directory}\nThe tables directory alone cannot run these new controls.")
    meta = read_json(j)
    require(meta.get("case_id") == case_id and meta.get("index_name") == market,
            f"Case identity mismatch: {j}")
    require(int(meta["horizon"]) == horizon, f"Horizon mismatch: {j}")
    require(meta.get("guard_applied") is False, f"Unexpected guard in {j}")
    digest = sha256(p)
    require(meta.get("predictions_sha256") == digest, f"Prediction checksum mismatch: {p}")
    provenance[str(j.relative_to(root))] = sha256(j)
    provenance[str(p.relative_to(root))] = digest
    with np.load(p, allow_pickle=False) as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    require(set(("hybrid", "core_ols", "har", "garch")).issubset(meta["method_keys"]),
            f"Required model/baseline predictions absent: {j}")
    case = Case(market, backbone, mode, directory, meta, arrays)
    for split in ("val", "test"):
        dates = np.asarray(case.dates(split), dtype="datetime64[ns]")
        require(dates.ndim == 1 and not np.isnat(dates).any(), f"Invalid dates: {j}")
        require((np.diff(dates) > np.timedelta64(0, "ns")).all(), f"Unordered/duplicate dates: {j}")
        y = vector(case.true(split), f"{case_id}/{split}/target")
        require(len(dates) == len(y), f"Target/date length mismatch: {j}")
        for method in ("hybrid", "core", "har", "garch"):
            prediction = vector(case.pred(split, method), f"{case_id}/{split}/{method}")
            score = float(qlike(y, prediction).mean())
            require(len(prediction) == len(y), f"Prediction length mismatch: {j}")
            saved_metrics = meta["validation_metrics" if split == "val" else "test_metrics"]
            saved_key = "core_ols" if method == "core" else method
            require(np.isclose(score, saved_metrics[saved_key]["QLIKE"], atol=METRIC_ATOL, rtol=1e-6),
                    f"Saved metric does not match predictions: {case_id}/{split}/{method}")
    require(case.dates("val")[-1] < case.dates("test")[0], f"Validation/test chronology mismatch: {j}")
    seed_rows = meta.get("seed_metrics", [])
    require(len(seed_rows) == len(seeds) and {int(r["seed"]) for r in seed_rows} == set(seeds),
            f"Missing/duplicate seed metadata: {j}")
    for row in seed_rows:
        require(bool(row["selected_epoch_zero"]) == (int(row["best_epoch"]) == 0),
                f"Contradictory epoch-zero metadata: {j}")
        require(float(row["validation_QLIKE"]) <= float(row["initial_validation_QLIKE"]) + METRIC_ATOL,
                f"Checkpoint selection violates epoch-zero rule: {j}")
    return case


def align(a, b):
    for split in ("val", "test"):
        require(np.array_equal(a.dates(split), b.dates(split)), f"Dates differ: {a.directory} vs {b.directory}")
        require(np.array_equal(a.true(split), b.true(split)), f"Targets differ: {a.directory} vs {b.directory}")
        for method in ("core", "har", "garch"):
            require(np.allclose(a.pred(split, method), b.pred(split, method), rtol=0, atol=1e-10),
                    f"Baseline differs between cases: {method}/{a.market}")


def correction_summary(correction):
    f = vector(correction, "log-volatility correction")
    quantiles = np.quantile(f, [.05, .25, .5, .75, .95])
    output = dict(n_dates=len(f), mean_log_correction=float(f.mean()),
                  std_log_correction_population=float(f.std(ddof=0)),
                  rms_log_correction=float(np.sqrt(np.mean(f*f))),
                  max_abs_log_correction=float(np.max(np.abs(f))))
    output.update({f"q{int(q*100):02d}_log_correction": float(v)
                   for q, v in zip([.05, .25, .5, .75, .95], quantiles)})
    return output





def per_seed(case, seeds, provenance):
    """Verify and export every saved seed forecast without changing predictions."""
    rows, corrections, saved_predictions, daily = [], [], {}, []
    for original in sorted(case.meta["seed_metrics"], key=lambda x: int(x["seed"])):
        row = dict(market=case.market, backbone=case.backbone, mode=case.mode, **original)
        seed = int(row["seed"])
        path = case.directory / "checkpoints" / f"seed_{seed}_predictions.npz"
        row["daily_seed_prediction_available"] = path.is_file()
        require(path.is_file(),
                f"Missing individual-seed prediction cache: {path}\n"
                "All saved seeds are required. Use the complete saved run, not an exported tables folder.")
        digest = sha256(path)
        provenance[str(path.relative_to(case.directory.parent.parent))] = digest
        seed_meta = path.with_name(f"seed_{seed}_result.json")
        if seed_meta.is_file():
            require(read_json(seed_meta).get("predictions_sha256") == digest,
                    f"Seed prediction checksum mismatch: {path}")
            provenance[str(seed_meta.relative_to(case.directory.parent.parent))] = sha256(seed_meta)
        with np.load(path, allow_pickle=False) as saved:
            saved_predictions[seed] = {}
            for split, prefix in (("val", "validation"), ("test", "test")):
                require(np.array_equal(saved[prefix+"_dates"], case.dates(split)), f"Seed dates differ: {path}")
                require(np.array_equal(saved[prefix+"_true"], case.true(split)), f"Seed targets differ: {path}")
                prediction = vector(saved[prefix+"_prediction"], str(path))
                y = case.true(split)
                score = float(qlike(y, prediction).mean())
                require(np.isclose(score, row[prefix+"_QLIKE"], atol=METRIC_ATOL, rtol=1e-6),
                        f"Seed metric mismatch: {path}/{split}")
                row[prefix+"_QLIKE"] = score
                row[prefix+"_clipped"] = clip_count(y, prediction)
                saved_predictions[seed][split] = prediction
                daily.append(pd.DataFrame(dict(market=case.market, backbone=case.backbone,
                    mode=case.mode, seed=seed, split=split, date=pd.to_datetime(case.dates(split)),
                    true_logvol=y, prediction_logvol=prediction)))
                if case.mode == "residual":
                    correction = prediction - case.pred(split, "core")
                    if row["selected_epoch_zero"]:
                        require(np.max(np.abs(correction)) <= CORE_ATOL,
                                f"Epoch-zero prediction differs from core: {path}")
                    corrections.append(dict(market=case.market, backbone=case.backbone,
                        seed=seed, estimator="individual_seed", correction_kind="raw_neural_correction",
                        split=split, selected_epoch_zero=bool(row["selected_epoch_zero"]),
                        **correction_summary(correction)))
        rows.append(row)
    require(set(saved_predictions) == set(seeds), "Missing saved seeds")
    for split in ("val", "test"):
        reconstructed = np.mean([saved_predictions[s][split] for s in seeds], axis=0)
        require(np.allclose(reconstructed, case.pred(split, "hybrid"), rtol=0, atol=3e-6),
                f"Ensemble is not the mean of saved seed log predictions: {case.directory}")
    return rows, corrections, daily


def find_artifacts(explicit=None):
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if root.name == "cases":
            root = root.parent
        require((root / "completion.json").is_file(), f"No completion.json in {root}")
        return root
    candidates = set()
    locations = [Path.cwd(), Path(__file__).resolve().parent]
    for base in locations:
        for parent in (base, base.parent):
            for root in (parent, parent / "artifacts_matched_control"):
                if (root / "completion.json").is_file() and (root / "cases").is_dir():
                    candidates.add(root.resolve())
    require(len(candidates) == 1,
        "Cannot identify a unique saved run. Put this .py file next to run_matched.py,\n"
        "or use: python run_posthoc_controls.py --artifacts /path/to/artifacts_matched_control\n"
        f"Candidates: {sorted(map(str, candidates))}")
    return candidates.pop()


def csv(path, frame):
    frame.to_csv(path, index=False, float_format="%.17g")


def summarize_seeds(seed_frame, market_frame, backbones, score_column="test_QLIKE"):
    by_seed_rows, summary_rows, sensitivity = [], [], []
    for excluded in (None, "EEM"):
        if excluded and excluded not in seed_frame.market.unique():
            continue
        label = "all_markets" if excluded is None else "without_EEM"
        frame = seed_frame if excluded is None else seed_frame[seed_frame.market.ne(excluded)]
        grouped = frame.groupby(["backbone", "mode", "seed"]).agg(test_QLIKE=(score_column, "mean")).reset_index()
        grouped["analysis"] = label
        grouped["score_source"] = score_column
        grouped["n_markets"] = frame.market.nunique()
        by_seed_rows.append(grouped)
        summary = grouped.groupby(["backbone", "mode"]).test_QLIKE.agg(["mean", "std", "count"]).reset_index()
        summary["analysis"] = label
        summary["score_source"] = score_column
        summary_rows.append(summary)
        direct = summary[summary["mode"].eq("direct")]["mean"]
        residual = summary[summary["mode"].eq("residual")]["mean"]
        d_range, r_range = float(direct.max()-direct.min()), float(residual.max()-residual.min())
        sensitivity.append(dict(analysis=label, n_markets=frame.market.nunique(),
            score_source=score_column,
            direct_across_backbone_range=d_range, residual_across_backbone_range=r_range,
            range_ratio=d_range/r_range if r_range > 0 else None))
    lomo = []
    for backbone in backbones:
        group = market_frame[market_frame.backbone.eq(backbone)]
        for omitted in group.market:
            kept = group[group.market.ne(omitted)]
            if len(kept):
                row = dict(backbone=backbone, omitted_market=omitted, n_markets=len(kept))
                row.update({col: float(kept[col].mean()) for col in kept.columns if col.startswith("QLIKE_")})
                lomo.append(row)
    return pd.concat(by_seed_rows), pd.concat(summary_rows), pd.DataFrame(sensitivity), pd.DataFrame(lomo)


def run_analysis(root, output_parent=None, quiet=False):
    root = Path(root).resolve()
    completion = read_json(root / "completion.json")
    suite = read_json(root / "suite_definition.json")
    require(completion.get("complete") is True, "The saved matched run is not marked complete")
    markets, backbones = completion["all_markets"], completion["all_backbones"]
    seeds = [int(s) for s in completion["seeds"]]
    primary = completion["primary_table_backbone"]
    horizon = int(suite["horizon"])
    require(len(markets) > 1 and len(seeds) >= 2 and horizon > 0, "Invalid run dimensions")
    require(len(set(markets)) == len(markets) and len(set(backbones)) == len(backbones)
            and len(set(seeds)) == len(seeds), "Duplicate run identifiers")
    require(primary in backbones, "Primary backbone is absent")
    require(suite["markets"] == markets and suite["backbones"] == backbones and suite["seeds"] == seeds,
            "Completion and suite universes differ")
    expected_runs = len(markets) * len(backbones) * len(seeds) * 2
    require(completion["observed_training_runs"] == expected_runs, "Incomplete training count")
    require(completion.get("zero_initialized_both_modes") is True
            and completion.get("epoch_zero_candidate_both_modes") is True,
            "This script is for the completed matched-input/zero-readout protocol")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    parent = Path(output_parent).resolve() if output_parent else root.parent / "posthoc_controls"
    parent.mkdir(parents=True, exist_ok=True)
    out = parent / f"run_{timestamp}"
    out.mkdir(exist_ok=False)
    provenance = {n: sha256(root/n) for n in ("completion.json", "suite_definition.json")}
    policy = dict(version=VERSION, markets=markets, backbones=backbones, seeds=seeds,
        primary_backbone=primary, horizon=horizon,
        evaluation="QLIKE of the original mean log forecast across seeds; float64; clip 2*error to [-30,30]",
        checkpoint_selection="reuse original checkpoints and saved predictions without reselection",
        comparison_pairs=COMPARISON_PAIRS,
        dm="two-sided Student-t; Bartlett HAC lag h-1; Harvey-Leybourne-Newbold correction",
        multiplicity="Holm/BH across all markets within each backbone/method-pair family; "
            "also Holm jointly over all original-forecast comparisons for the primary backbone and over all exported tests",
        undefined_tests="NaN p_raw, counted as p=1 for multiplicity; never significant",
        study_status="post-hoc diagnostic controls on previously examined markets",
        robustness="sample SD of market-averaged seed scores; separate all-markets and without-EEM summaries",
        selection="all configured cases retained; no test-based market/model/baseline selection",
        core_tolerance_logvol=CORE_ATOL)
    write_json(out/"analysis_policy.json", policy)
    metric_rows, dm_rows, seed_rows = [], [], []
    corrections, abstention, daily, seed_daily = [], [], [], []
    try:
        for market in markets:
            if not quiet:
                print(f"[read] {market}: {len(backbones)*2} saved cases", flush=True)
            reference = None
            for backbone in backbones:
                residual = load_case(root, market, backbone, "residual", horizon, seeds, provenance)
                direct = load_case(root, market, backbone, "direct", horizon, seeds, provenance)
                align(residual, direct)
                rr = {int(s["seed"]): s for s in residual.meta["seed_metrics"]}
                dd = {int(s["seed"]): s for s in direct.meta["seed_metrics"]}
                require(all(rr[s]["initial_trainable_hash"] == dd[s]["initial_trainable_hash"] for s in seeds),
                        f"Paired initial weights differ: {market}/{backbone}")
                for key in ("features", "sizes", "date_ranges", "corrector_iv_mode", "input_sha256", "target_standardizer"):
                    require(key in residual.meta["metadata"]["data"] and key in direct.meta["metadata"]["data"],
                            f"Required paired-input metadata absent: {market}/{backbone}/{key}")
                    require(residual.meta["metadata"]["data"][key] == direct.meta["metadata"]["data"][key],
                            f"Paired input metadata differ: {market}/{backbone}/{key}")
                if reference is None:
                    reference = residual
                else:
                    align(reference, residual)
                full_core = all(rr[s]["selected_epoch_zero"] for s in seeds)
                n_zero = sum(bool(rr[s]["selected_epoch_zero"]) for s in seeds)
                abstention.append(dict(market=market, backbone=backbone, n_seeds=len(seeds),
                    n_epoch_zero=n_zero, all_seeds_epoch_zero=full_core, any_seed_epoch_zero=n_zero > 0))
                for case in (residual, direct):
                    sr, cr, sd = per_seed(case, seeds, provenance)
                    seed_rows.extend(sr)
                    corrections.extend(cr)
                    seed_daily.extend(sd)
                for split in ("val", "test"):
                    y = vector(residual.true(split), "target")
                    predictions = {m: vector(residual.pred(split, m), m) for m in ("core", "har", "garch")}
                    predictions.update(direct=vector(direct.pred(split, "direct"), "direct"),
                        residual=vector(residual.pred(split, "residual"), "residual"))
                    correction = predictions["residual"] - predictions["core"]
                    if full_core:
                        require(np.max(np.abs(correction)) <= CORE_ATOL,
                                f"All seeds select epoch zero but output differs from core: {market}/{backbone}")
                    corrections.append(dict(market=market, backbone=backbone, seed="ensemble",
                        estimator="mean_log_prediction", correction_kind="raw_neural_correction", split=split,
                        selected_epoch_zero=full_core, **correction_summary(correction)))
                    daily.append(pd.DataFrame(dict(market=market, backbone=backbone, split=split,
                        date=pd.to_datetime(residual.dates(split)), true_logvol=y,
                        **{name+"_logvol": p for name, p in predictions.items()},
                        correction_logvol=correction)))
                    if split == "test":
                        losses = {name: qlike(y, p) for name, p in predictions.items()}
                        metric_rows.append(dict(market=market, backbone=backbone, n_test=len(y),
                            **{"QLIKE_"+name: float(value.mean()) for name, value in losses.items()},
                            **{"clipped_"+name: clip_count(y, p) for name, p in predictions.items()}))
                        for method_a, method_b in COMPARISON_PAIRS:
                            dm_rows.append(dict(market=market, backbone=backbone,
                                family=f"{backbone}:{method_a}_vs_{method_b}", comparison=f"{method_a}_vs_{method_b}",
                                method_a=method_a, method_b=method_b,
                                **dm_test(losses[method_a], losses[method_b], horizon)))
        require(len(seed_rows) == expected_runs, "Missing seed rows in post-hoc analysis")
        metrics, dm = pd.DataFrame(metric_rows), pd.DataFrame(dm_rows)
        sf, af, cf = pd.DataFrame(seed_rows), pd.DataFrame(abstention), pd.DataFrame(corrections)
        for _, indexes in dm.groupby("family", sort=False).groups.items():
            require(len(indexes) == len(markets), "A DM family has missing markets")
            dm.loc[indexes, "p_holm"] = adjusted(dm.loc[indexes, "p_raw"], "holm")
            dm.loc[indexes, "p_bh"] = adjusted(dm.loc[indexes, "p_raw"], "bh")
        dm["p_holm_all_exported_tests"] = adjusted(dm.p_raw, "holm")
        primary_tests = dm.backbone.eq(primary)
        dm["p_holm_primary_joint"] = np.nan
        dm.loc[primary_tests, "p_holm_primary_joint"] = adjusted(dm.loc[primary_tests, "p_raw"], "holm")
        dm["significant_win_holm"] = dm.p_raw.notna() & dm.p_holm.lt(.05) & dm.mean_loss_a_minus_b.lt(0)
        dm["significant_loss_holm"] = dm.p_raw.notna() & dm.p_holm.lt(.05) & dm.mean_loss_a_minus_b.gt(0)
        primary_metrics = metrics[metrics.backbone.eq(primary)]
        primary_columns = ["market"] + ["QLIKE_"+m for m in METHODS]
        table = primary_metrics.set_index("market").loc[markets].reset_index()[primary_columns]
        table.loc[len(table)] = ["Average"] + [float(table[c].mean()) for c in primary_columns[1:]]
        by_seed, seed_stats, spread, lomo = summarize_seeds(sf, metrics, backbones)
        backbone_summary = []
        for backbone in backbones:
            frame = metrics[metrics.backbone.eq(backbone)]
            row = dict(backbone=backbone, n_markets=len(frame),
                       **{c: float(frame[c].mean()) for c in frame if c.startswith("QLIKE_")})
            for comparator in COMPARATORS:
                base = row["QLIKE_"+comparator]
                row["residual_gain_percent_vs_"+comparator] = 100*(base-row["QLIKE_residual"])/base if base > 0 else None
                delta = frame.QLIKE_residual - frame["QLIKE_"+comparator]
                row["numerical_wins_vs_"+comparator] = int(delta.lt(-1e-12).sum())
                row["numerical_ties_vs_"+comparator] = int(delta.abs().le(1e-12).sum())
                pair = dm[dm.backbone.eq(backbone) & dm.method_a.eq("residual") & dm.method_b.eq(comparator)]
                row["holm_wins_vs_"+comparator] = int(pair.significant_win_holm.sum())
                row["holm_losses_vs_"+comparator] = int(pair.significant_loss_holm.sum())
            backbone_summary.append(row)
        outputs = {"market_qlike.csv": metrics, "primary_qlike.csv": table, "dm_tests.csv": dm,
            "backbone_summary.csv": pd.DataFrame(backbone_summary),
            "primary_dm.csv": dm[dm.backbone.eq(primary)], "seed_metrics.csv": sf,
            "equal_weight_by_seed.csv": by_seed, "equal_weight_seed_mean_std.csv": seed_stats,
            "robustness_sensitivity.csv": spread, "leave_one_market_out.csv": lomo,
            "core_retention.csv": af, "correction_magnitude.csv": cf,
            "daily_predictions.csv": pd.concat(daily, ignore_index=True),
            "daily_seed_predictions.csv": pd.concat(seed_daily, ignore_index=True)}
        old_dm_path = root/"tables"/"deep_backbone_dm_tests.csv"
        if old_dm_path.is_file():
            old = pd.read_csv(old_dm_path)
            old = old.rename(columns={c: "original_"+c for c in ("dm", "p_raw", "p_holm")})
            if "method_a" in old:
                old = old[old.method_a.eq("residual")]
            compared = dm[dm.method_a.eq("residual")].merge(old[["market", "backbone", "method_b", "original_dm", "original_p_raw", "original_p_holm"]],
                on=["market", "backbone", "method_b"], how="inner")
            compared["holm_decision_changed"] = compared.p_holm.lt(.05) != compared.original_p_holm.lt(.05)
            outputs["original_dm_recheck.csv"] = compared
        for name, frame in outputs.items():
            csv(out/name, frame)
        zero_cells = int(af.all_seeds_epoch_zero.sum())
        residual_sf = sf[sf["mode"].eq("residual")]
        summary = dict(complete=True, analysis_version=VERSION, source_training_runs=expected_runs,
            market_backbone_cells=len(af), complete_core_retention_cells=zero_cells,
            complete_core_retention_percent=100*zero_cells/len(af),
            cells_with_any_epoch_zero=int(af.any_seed_epoch_zero.sum()),
            residual_seed_runs=len(residual_sf), residual_epoch_zero_runs=int(residual_sf.selected_epoch_zero.sum()),
            individual_daily_prediction_files_found=int(sf.daily_seed_prediction_available.sum()),
            individual_daily_prediction_files_expected=len(sf),
            ensemble_daily_predictions_exported=True,
            individual_daily_predictions_exported=True,
            primary_backbone=primary, primary_equal_weight_qlike=table.iloc[-1].to_dict(),
            primary_holm_significant_wins=dm[dm.backbone.eq(primary)].groupby("comparison").significant_win_holm.sum().to_dict(),
            primary_holm_significant_losses=dm[dm.backbone.eq(primary)].groupby("comparison").significant_loss_holm.sum().to_dict(),
            original_dm_holm_decisions_changed=int(outputs["original_dm_recheck.csv"].holm_decision_changed.sum())
                if "original_dm_recheck.csv" in outputs else None,
            notes=["Post-hoc diagnostics use the original saved forecasts and checkpoints.",
                   "Mean/SD of individual-seed scores and QLIKE of ensemble forecasts are different estimands.",
                   "Epoch-zero retention is measured on original checkpoints.",
                   "Correction variation is descriptive, not proof of useful learned dynamics or necessary depth.",
                   "Sample SD is across market-averaged seed scores, not a confidence interval.",
                   "Without-EEM and leave-one-market-out summaries reuse fitted models; no retraining.",
                   "All individual-seed prediction caches are required; no incomplete seed averaging is allowed.",
                   "Tiny DM differences versus the old table can arise from float64 versus native float32 loss arithmetic."])
        write_json(out/"summary.json", summary)
        write_json(out/"input_checksums.json", provenance)
        write_json(out/"environment.json", dict(python=platform.python_version(), numpy=np.__version__,
            pandas=pd.__version__, scipy=scipy.__version__, script_sha256=sha256(__file__)))
        with (out/"README.txt").open("w", encoding="utf-8") as stream:
            stream.write(
                "POST-HOC ICASSP ANALYSIS OF ORIGINAL FORECASTS -- VERSION 3.0\n\n"
                "Read primary_qlike.csv and primary_dm.csv for the saved primary backbone.\n"
                "Lower QLIKE is better. Negative mean_loss_a_minus_b favors method_a.\n"
                "All forecasts and checkpoints are reused exactly as saved.\n"
                "DM comparisons are residual versus direct, core, GARCH and HAR.\n"
                "p_holm/p_bh correct across markets within each backbone/method-pair family.\n"
                "p_holm_primary_joint corrects all original-forecast comparisons for the primary backbone.\n"
                "p_holm_all_exported_tests is a separate global correction across the exported comparisons.\n"
                "DM uses two-sided Student-t, Bartlett HAC lag h-1 and HLN correction.\n"
                "The evaluation clips 2*error to [-30,30], matching the completed run.\n"
                "core_retention.csv counts original epoch-zero selections.\n"
                "correction_magnitude.csv describes residual minus core for ensembles and individual seeds.\n"
                "Temporal correction SD uses ddof=0; market-averaged seed score SD uses ddof=1.\n"
                "Seed means are distinct from ensemble QLIKE; seed SD is not a confidence interval.\n"
                "daily_predictions.csv and daily_seed_predictions.csv retain original predictions and targets.\n"
                "These fitted-model DM tests do not integrate over training seeds or post-hoc study selection.\n"
                "All configured markets, backbones and seeds are retained.\n")
        archive = parent / f"posthoc_original_forecasts_v3_{timestamp}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for path in sorted(out.iterdir()):
                if path.is_file():
                    z.write(path, path.name)
            z.write(Path(__file__).resolve(), "run_posthoc_controls.py")
        with zipfile.ZipFile(archive) as z:
            require(z.testzip() is None, "Output ZIP integrity check failed")
        if not quiet:
            print(f"\n[complete v{VERSION}] No training, download or environment change.", flush=True)
            print(f"Original saved forecasts -- primary backbone: {primary}", flush=True)
            print(table.to_string(index=False), flush=True)
            print("Statistical results: primary_dm.csv; scores: primary_qlike.csv", flush=True)
            print(f"\nFull core retention: {zero_cells}/{len(af)} cells", flush=True)
            print(f"Individual seed forecasts found: {summary['individual_daily_prediction_files_found']}/{len(sf)}", flush=True)
            print(f"\nSEND THIS SINGLE ZIP:\n{archive}", flush=True)
        return out, archive
    except Exception as exc:
        write_json(out/"failure.json", dict(complete=False, error=str(exc)))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", type=Path, help="Existing matched-run output directory")
    parser.add_argument("--output-dir", type=Path, help="Parent for new timestamped result folders")
    parser.add_argument("--self-test", action="store_true", help="Run internal synthetic checks, not research experiments")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    try:
        root = find_artifacts(args.artifacts)
        print(f"[input] {root}", flush=True)
        run_analysis(root, args.output_dir)
    except (ValueError, KeyError, OSError, zipfile.BadZipFile) as exc:
        print(f"\nSTOP: {exc}\nNo training was launched and existing artifacts were not changed.", file=sys.stderr)
        return 1
    return 0


def self_test():
    """Defined below; tests use synthetic targets only, never supplied research results."""
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ControlTests)
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


class ControlTests(unittest.TestCase):




    def test_holm_bh_and_missing_test_family_size(self):
        np.testing.assert_allclose(adjusted([.01,.04,.03], "holm"), [.03,.06,.06])
        np.testing.assert_allclose(adjusted([.01,.04,.03], "bh"), [.03,.04,.04])
        np.testing.assert_allclose(adjusted([.01,float("nan"),.04], "holm"), [.03,1.,.08])

    def test_dm_hac_hln_reference_sign_and_degeneracy(self):
        k = np.arange(80)
        d = -.08 + .1*np.sin(k*.37) + .02*np.cos(k*.19)
        a, b = 1+d, np.ones(len(d))
        result = dm_test(a, b, 5)
        c = d-d.mean()
        gamma = [sum(c[t]*c[t-j] for t in range(j,len(c)))/len(c) for j in range(5)]
        variance = (gamma[0]+2*sum((1-j/5)*gamma[j] for j in range(1,5)))/len(c)
        correction = (len(c)+1-10+20/len(c))/len(c)
        expected = d.mean()/math.sqrt(variance)*math.sqrt(correction)
        self.assertAlmostEqual(result["dm"], expected, places=11)
        reverse = dm_test(b, a, 5)
        self.assertAlmostEqual(result["dm"], -reverse["dm"], places=12)
        self.assertAlmostEqual(result["p_raw"], reverse["p_raw"], places=12)
        self.assertEqual(dm_test(b,b,5)["p_raw"],1)
        self.assertEqual(dm_test(b+1,b,5)["status"],"degenerate_variance")

    @staticmethod
    def fixture(base, constant_neural=False, test_target_shift=0.0):
        """A fully synthetic on-disk run in the exact existing NPZ/JSON schema."""
        root = Path(base)/"synthetic_artifacts"
        root.mkdir()
        markets, backbones, seeds = ["SPX","EEM"], ["bimamba","lstm"], [1,2,3]
        completion = dict(complete=True, all_markets=markets, all_backbones=backbones,
            seeds=seeds, primary_table_backbone="bimamba", observed_training_runs=24,
            zero_initialized_both_modes=True, epoch_zero_candidate_both_modes=True)
        write_json(root/"completion.json",completion)
        write_json(root/"suite_definition.json",dict(markets=markets,backbones=backbones,seeds=seeds,horizon=5))
        for mi, market in enumerate(markets):
            dates = {"val":np.datetime64("2024-01-01")+np.arange(48),
                     "test":np.datetime64("2024-03-01")+np.arange(64)}
            targets = {sp:-4+.1*mi+.14*np.sin(np.arange(len(dt))*.31)+.03*np.cos(np.arange(len(dt))*.17)
                       for sp,dt in dates.items()}
            observed_targets = {sp: y + (test_target_shift if sp == "test" else 0.0) for sp,y in targets.items()}
            baseline = {sp:dict(core_ols=y-.2,har=y-.3,garch=y-.04+.04*np.sin(np.arange(len(y))*.23))
                        for sp,y in targets.items()}
            for bi, backbone in enumerate(backbones):
                for mode in ("direct","residual"):
                    case_id=f"matched_pair_{market}_{backbone}_{mode}"
                    directory=root/"cases"/case_id
                    (directory/"checkpoints").mkdir(parents=True)
                    rows, individual=[],{}
                    for seed in seeds:
                        zero = mode=="residual" and market=="EEM"
                        predictions={}
                        row=dict(seed=seed,best_epoch=0 if zero else seed,selected_epoch_zero=zero,
                                 initial_trainable_hash=f"identical-{backbone}-{seed}")
                        seed_arrays={}
                        for sp,prefix in (("val","validation"),("test","test")):
                            y=targets[sp]
                            if mode=="residual":
                                variation = 1.0 if constant_neural else np.sin(np.arange(len(y))*.4+bi)
                                pred=baseline[sp]["core_ols"]+ (0 if zero else .17+.003*seed*variation)
                            else:
                                variation = 1.0 if constant_neural else np.cos(np.arange(len(y))*.2+bi)
                                pred=y-.12-.006*seed*variation
                            predictions[sp]=pred
                            row[prefix+"_QLIKE"]=float(qlike(observed_targets[sp],pred).mean())
                            seed_arrays.update({prefix+"_dates":dates[sp].astype("datetime64[ns]"),
                                prefix+"_true":observed_targets[sp],prefix+"_prediction":pred})
                        row["initial_validation_QLIKE"]=row["validation_QLIKE"] if zero else row["validation_QLIKE"]+.2
                        path=directory/"checkpoints"/f"seed_{seed}_predictions.npz"
                        np.savez_compressed(path,**seed_arrays)
                        write_json(path.with_name(f"seed_{seed}_result.json"),dict(predictions_sha256=sha256(path)))
                        rows.append(row)
                        individual[seed]=predictions
                    method_keys={name:f"m{i:03}" for i,name in enumerate(["hybrid","core_ols","har","garch"])}
                    arrays, all_metrics={},{}
                    for sp,prefix in (("val","validation"),("test","test")):
                        arrays.update({prefix+"_dates":dates[sp].astype("datetime64[ns]"),prefix+"_true":observed_targets[sp]})
                        methods=dict(baseline[sp],hybrid=np.mean([individual[s][sp] for s in seeds],axis=0))
                        for method,key in method_keys.items():
                            arrays[sp+"_"+key]=methods[method]
                        all_metrics[prefix+"_metrics"]={m:dict(QLIKE=float(qlike(observed_targets[sp],v).mean())) for m,v in methods.items()}
                    path=directory/"predictions.npz"
                    np.savez_compressed(path,**arrays)
                    write_json(directory/"result.json",dict(case_id=case_id,index_name=market,horizon=5,
                        guard_applied=False,predictions_sha256=sha256(path),method_keys=method_keys,
                        seed_metrics=rows,metadata=dict(data=dict(features=["x"],sizes=[48,64],date_ranges={},
                            corrector_iv_mode="selected",input_sha256="same",target_standardizer=dict(mean=-4,std=.2))),
                        **all_metrics))
        return root

    def test_full_synthetic_run_exports_required_evidence(self):
        with tempfile.TemporaryDirectory(prefix="icassp_posthoc_test_") as temporary:
            root=self.fixture(temporary)
            before={str(p.relative_to(root)):sha256(p) for p in root.rglob("*") if p.is_file()}
            out,archive=run_analysis(root,Path(temporary)/"results",quiet=True)
            summary=read_json(out/"summary.json")
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["complete_core_retention_cells"],2)
            self.assertEqual(summary["residual_epoch_zero_runs"],6)
            self.assertEqual(summary["individual_daily_prediction_files_found"],24)
            dm=pd.read_csv(out/"dm_tests.csv")
            self.assertEqual(len(dm),16)
            self.assertEqual(set(zip(dm.method_a,dm.method_b)),set(COMPARISON_PAIRS))
            self.assertTrue(dm.groupby("family").size().eq(2).all())
            for _,group in dm.groupby("family"):
                np.testing.assert_allclose(group.p_holm,adjusted(group.p_raw,"holm"))
            joint=dm[dm.backbone.eq("bimamba")]
            np.testing.assert_allclose(joint.p_holm_primary_joint,adjusted(joint.p_raw,"holm"))
            daily=pd.read_csv(out/"daily_predictions.csv",float_precision="round_trip")
            for (market,backbone),group in daily.groupby(["market","backbone"]):
                losses={}
                for mode in ("residual","direct"):
                    case=load_case(root,market,backbone,mode,5,[1,2,3],{})
                    for split in ("val","test"):
                        rows=group[group.split.eq(split)]
                        np.testing.assert_array_equal(rows[mode+"_logvol"],case.pred(split,mode))
                        if mode=="residual":
                            for baseline in ("core","har","garch"):
                                np.testing.assert_array_equal(rows[baseline+"_logvol"],case.pred(split,baseline))
                test=group[group.split.eq("test")]
                for method in METHODS:
                    losses[method]=qlike(test.true_logvol,test[method+"_logvol"])
                for row in dm[dm.market.eq(market)&dm.backbone.eq(backbone)].itertuples():
                    expected=dm_test(losses[row.method_a],losses[row.method_b],5)
                    np.testing.assert_allclose([row.dm,row.p_raw],
                        [expected["dm"],expected["p_raw"]],atol=1e-12,equal_nan=True)
            seed_daily=pd.read_csv(out/"daily_seed_predictions.csv",float_precision="round_trip")
            seed_metrics=pd.read_csv(out/"seed_metrics.csv",float_precision="round_trip")
            self.assertEqual(len(seed_metrics),24)
            for (market,backbone,mode,seed),group in seed_daily.groupby(["market","backbone","mode","seed"]):
                p=root/"cases"/f"matched_pair_{market}_{backbone}_{mode}"/"checkpoints"/f"seed_{seed}_predictions.npz"
                with np.load(p) as saved:
                    for split,prefix in (("val","validation"),("test","test")):
                        rows=group[group.split.eq(split)]
                        np.testing.assert_array_equal(rows.prediction_logvol,saved[prefix+"_prediction"])
                test=group[group.split.eq("test")]
                row=seed_metrics[seed_metrics.market.eq(market)&seed_metrics.backbone.eq(backbone)
                    &seed_metrics["mode"].eq(mode)&seed_metrics.seed.eq(seed)].iloc[0]
                self.assertAlmostEqual(row.test_QLIKE,qlike(test.true_logvol,test.prediction_logvol).mean(),places=12)
            by_seed=pd.read_csv(out/"equal_weight_by_seed.csv")
            all_market=by_seed[by_seed.analysis.eq("all_markets")]
            expected=seed_metrics.groupby(["backbone","mode","seed"]).test_QLIKE.mean().sort_index()
            np.testing.assert_allclose(all_market.set_index(["backbone","mode","seed"]).test_QLIKE.sort_index(),expected)
            stats=pd.read_csv(out/"equal_weight_seed_mean_std.csv")
            expected_stats=all_market.groupby(["backbone","mode"]).test_QLIKE.agg(["mean","std"])
            actual=stats[stats.analysis.eq("all_markets")].set_index(["backbone","mode"])
            np.testing.assert_allclose(actual[["mean","std"]].sort_index(),expected_stats.sort_index(),atol=1e-12)
            after={str(p.relative_to(root)):sha256(p) for p in root.rglob("*") if p.is_file()}
            self.assertEqual(before,after)
            with zipfile.ZipFile(archive) as z:
                self.assertIsNone(z.testzip())
                for name in ("primary_dm.csv","daily_predictions.csv","daily_seed_predictions.csv","run_posthoc_controls.py"):
                    self.assertIn(name,z.namelist())

    def test_corrupt_predictions_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="icassp_posthoc_test_") as temporary:
            root=self.fixture(temporary)
            p=root/"cases/matched_pair_SPX_bimamba_residual/predictions.npz"
            with p.open("ab") as stream:
                stream.write(b"CORRUPTED")
            with self.assertRaisesRegex(ValueError,"checksum mismatch"):
                run_analysis(root,Path(temporary)/"results",quiet=True)

    def test_missing_seed_caches_are_rejected_without_retraining(self):
        with tempfile.TemporaryDirectory(prefix="icassp_posthoc_test_") as temporary:
            root=self.fixture(temporary)
            p=root/"cases/matched_pair_SPX_bimamba_residual/checkpoints/seed_1_predictions.npz"
            p.rename(p.with_suffix(".unavailable"))
            with self.assertRaisesRegex(ValueError,"Missing individual-seed prediction cache"):
                run_analysis(root,Path(temporary)/"results",quiet=True)
            self.assertTrue(p.with_suffix(".unavailable").exists())



    def test_missing_case_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory(prefix="icassp_posthoc_test_") as temporary:
            root=self.fixture(temporary)
            p=root/"cases/matched_pair_EEM_lstm_direct/predictions.npz"
            p.rename(p.with_suffix(".unavailable"))
            with self.assertRaisesRegex(ValueError,"Missing saved case"):
                run_analysis(root,Path(temporary)/"results",quiet=True)


if __name__ == "__main__":
    raise SystemExit(main())
