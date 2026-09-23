#!/usr/bin/env python3
"""Stacking-ensemble hybrid baseline (Peter et al., Array 2026).

Mechanism kept as published: GARCH and a standalone deep model act as
*parallel* base learners rather than a cascade, and a meta-learner fuses their
forecasts while exploiting feature interactions.

Protocol matched to this paper: the meta-learner is fitted on the validation
split only, scored by QLIKE, and applied unchanged to the test split.  Nothing
is fitted on test, and no neural network is retrained -- both base learners are
read from predictions already on disk, so the whole script runs in seconds.

Meta-features (``--features linear``, the default and the setting reported
in the paper) are ``[g, l, 1]``, with ``g`` the GARCH forecast and ``l`` the
direct deep forecast, both in log-volatility units.  ``--features
interactions`` adds ``g*l, g^2, l^2``; unregularized, these terms extrapolate
badly when a base learner diverges on test, so they are not the default.

The meta-learner is fitted by minimising validation QLIKE, matching the
objective used everywhere else here; ``--fit ols`` gives the least-squares
meta-learner of conventional stacking instead.

Usage
-----
    python run_stacking_baseline.py --config configs/matched_control.json \
        --backbone lstm

Writes ``<output_dir>/hybrid_baselines/<market>/stack/predictions.npz`` in the
same layout as the other hybrid baselines, plus a CSV summary and the
Diebold-Mariano comparison against the anchored model.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from volatility_pipeline.config import load_config
from volatility_pipeline.statistics import adjust_pvalues, daily_qlike, dm_test

DEFAULT_MARKETS = ["SPX", "NDX", "RUT", "DIA", "GLD", "USO", "EEM"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/matched_control.json")
    parser.add_argument("--backbone", default="lstm",
                        help="deep base learner; 'lstm' matches the original")
    parser.add_argument("--features", choices=["linear", "interactions"],
                        default="linear")
    parser.add_argument("--fit", choices=["qlike", "ols"], default="qlike")
    parser.add_argument("--baseline-dir", default="hybrid_baselines")
    parser.add_argument("--markets", nargs="*", default=None)
    return parser.parse_args()


def load_case(root: Path, case_id: str) -> dict:
    case_dir = root / "cases" / case_id
    payload = json.loads((case_dir / "result.json").read_text(encoding="utf-8"))
    with np.load(case_dir / "predictions.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    keys = payload["method_keys"]
    return {
        "val": {m: arrays[f"val_{k}"] for m, k in keys.items()},
        "test": {m: arrays[f"test_{k}"] for m, k in keys.items()},
        "val_true": arrays["validation_true"],
        "test_true": arrays["test_true"],
        "val_dates": arrays["validation_dates"],
        "test_dates": arrays["test_dates"],
    }


def design(g: np.ndarray, l: np.ndarray, kind: str) -> np.ndarray:
    columns = [g, l]
    if kind == "interactions":
        columns += [g * l, g**2, l**2]
    columns.append(np.ones_like(g))
    return np.column_stack(columns)


def fit_meta(x: np.ndarray, y: np.ndarray, how: str) -> np.ndarray:
    ols, *_ = np.linalg.lstsq(x, y, rcond=None)
    if how == "ols":
        return ols
    result = minimize(
        lambda w: float(daily_qlike(x @ w, y).mean()),
        ols, method="Nelder-Mead",
        options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-12},
    )
    # Fall back to least squares if the direct search fails to improve.
    if not result.success or not np.isfinite(result.fun):
        return ols
    return result.x if result.fun < daily_qlike(x @ ols, y).mean() else ols


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    root = Path(cfg.output_dir)
    markets = args.markets or list(
        getattr(cfg, "confirmatory_markets", DEFAULT_MARKETS)
    )
    out_dir = root / args.baseline_dir
    horizon = cfg.primary_horizon

    rows, families = [], []
    for market in markets:
        direct = load_case(root, f"matched_pair_{market}_{args.backbone}_direct")
        residual = load_case(root, f"matched_pair_{market}_{args.backbone}_residual")
        if not np.array_equal(direct["test_dates"], residual["test_dates"]):
            raise AssertionError(f"{market}: paired cases are not date-aligned")

        # Base learners: GARCH and the standalone deep model, in parallel.
        x_val = design(direct["val"]["garch"], direct["val"]["hybrid"], args.features)
        x_test = design(direct["test"]["garch"], direct["test"]["hybrid"], args.features)
        weights = fit_meta(x_val, direct["val_true"], args.fit)
        val_prediction = x_val @ weights
        test_prediction = x_test @ weights

        truth = direct["test_true"]
        loss_stack = daily_qlike(test_prediction, truth)
        loss_ours = daily_qlike(residual["test"]["hybrid"], truth)
        result = dm_test(loss_ours, loss_stack, horizon)
        families.append({
            "comparison": "ours_vs_stack", "method_a": "ours", "method_b": "stack",
            "backbone": args.backbone, "market": market,
            "mean_qlike_a": float(loss_ours.mean()),
            "mean_qlike_b": float(loss_stack.mean()),
            "mean_loss_a_minus_b": result.mean_difference,
            "dm": result.statistic, "p_raw": result.pvalue,
            "n": result.n, "nw_lag": result.lag,
        })

        market_dir = out_dir / market / "stack"
        market_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            market_dir / "predictions.npz",
            validation_dates=direct["val_dates"], test_dates=direct["test_dates"],
            validation_prediction=val_prediction, test_prediction=test_prediction,
            validation_true=direct["val_true"], test_true=truth,
            meta_weights=weights,
        )
        rows.append({
            "market": market, "kind": "stack", "backbone": args.backbone,
            "features": args.features, "fit": args.fit,
            "ensemble_test_QLIKE": float(loss_stack.mean()),
            "ensemble_val_QLIKE": float(daily_qlike(val_prediction,
                                                    direct["val_true"]).mean()),
            "garch_test_QLIKE": float(daily_qlike(direct["test"]["garch"],
                                                  truth).mean()),
            "direct_test_QLIKE": float(daily_qlike(direct["test"]["hybrid"],
                                                   truth).mean()),
            "ours_test_QLIKE": float(loss_ours.mean()),
            "meta_weights": " ".join(f"{w:+.4f}" for w in weights),
        })
        print(f"[done] {market:<4} stack test QLIKE = "
              f"{rows[-1]['ensemble_test_QLIKE']:.4f}  "
              f"(GARCH {rows[-1]['garch_test_QLIKE']:.4f}, "
              f"direct {rows[-1]['direct_test_QLIKE']:.4f}, "
              f"ours {rows[-1]['ours_test_QLIKE']:.4f})")

    for row, p in zip(families, adjust_pvalues([f["p_raw"] for f in families], "holm")):
        row["p_holm"] = float(p)

    summary_path = out_dir / "stacking_baseline.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    dm_path = out_dir / "stacking_dm_tests.csv"
    with dm_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(families[0].keys()))
        writer.writeheader()
        writer.writerows(families)

    stack_mean = np.mean([r["ensemble_test_QLIKE"] for r in rows])
    ours_mean = np.mean([r["ours_test_QLIKE"] for r in rows])
    wins = sum(f["mean_loss_a_minus_b"] < 0 for f in families)
    significant = sum(f["mean_loss_a_minus_b"] < 0 and f["p_holm"] < 0.05
                      for f in families)
    against = sum(f["mean_loss_a_minus_b"] > 0 and f["p_holm"] < 0.05
                  for f in families)
    print(f"\nequal-weight mean QLIKE: stack {stack_mean:.4f}, ours {ours_mean:.4f}")
    print(f"ours lower in {wins}/{len(families)} markets; "
          f"{significant} Holm-significant for ours, {against} for stack; "
          f"min adjusted p = {min(f['p_holm'] for f in families):.4f}")
    print(f"\nwrote {summary_path}\n      {dm_path}")


if __name__ == "__main__":
    main()
