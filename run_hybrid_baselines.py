#!/usr/bin/env python3
"""Run the loss-level (GINN) and input-level (Kim & Won) hybrid baselines.

Bundles are built exactly as ``deep_baselines.py`` builds them for the paired
runs: ``iv_column="ANCHOR"``, ``core_set="full"``, the configured
``paired_corrector_iv_mode``, and the same frozen split dates.  The per-market
implied-volatility index is already resolved into the generic ANCHOR column by
the dataset builder, so no per-market IV mapping is applied here.

The GINN blend weight defaults to lambda = 0.1, the value selected on
validation QLIKE over {0.01, 0.1, 0.3, 0.5, 0.7} (see README).  Test scores
are never used for selection, and the frozen dataset is never rebuilt.

Usage
-----
    # 0. cheap check that the GARCH reference series is recovered correctly
    python run_hybrid_baselines.py --config configs/matched_control.json \
        --stage verify

    # 1. evaluate both baselines on the seven evaluation markets
    python run_hybrid_baselines.py --config configs/matched_control.json

Outputs land in ``<output_dir>/hybrid_baselines/``:
    hybrid_baselines.csv       one row per market/baseline, for Table 1
    dm_inputs.npz              daily test predictions, for the DM/Holm script
    <market>/<kind>/...        per-run summaries, weights, predictions

If the development markets are rebuilt later, ``sweep_lambda`` in
``volatility_pipeline.hybrid_baselines`` can tune lambda on them; it is left out
of this runner deliberately.


"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from volatility_pipeline.config import load_config
from volatility_pipeline.data import derive_fixed_split_dates, prepare_data
from volatility_pipeline.hybrid_baselines import (
    parameter_report,
    run_baseline_ensemble,
    verify_reference,
)

# The dataset builder resolves each market's own IV index into this column.
IV_COLUMN = "ANCHOR"
# Selected on validation QLIKE; the original paper used 0.01.
SELECTED_GINN_LAMBDA = 0.1
DEFAULT_EVAL_MARKETS = ["SPX", "NDX", "RUT", "DIA", "GLD", "USO", "EEM"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/matched_control.json")
    parser.add_argument("--stage", choices=["verify", "evaluate"], default="evaluate")
    parser.add_argument("--backbone", default="lstm",
                        help="encoder; LSTM matches the original implementations")
    parser.add_argument(
        "--lam", type=float, default=SELECTED_GINN_LAMBDA,
        help=f"GINN blend weight (default {SELECTED_GINN_LAMBDA}, selected on "
             "validation QLIKE). lam=1 is plain direct training; lam=0 is GINN-0.",
    )
    parser.add_argument(
        "--ginn-reference", choices=["garch", "core"], default="garch",
        help="GINN as published regularises toward GARCH (default). 'core' "
             "substitutes the fitted OLS fusion and is a deviation from the "
             "method, for sensitivity analysis only.",
    )
    parser.add_argument(
        "--input-append", choices=["garch", "core"], default="garch",
        help="'garch' is the Kim & Won mechanism (default). 'core' appends all "
             "five core channels; that is an ablation of the anchored model, "
             "not a baseline.",
    )
    parser.add_argument("--kinds", nargs="*", default=["ginn", "input"])
    parser.add_argument("--markets", nargs="*", default=None,
                        help="defaults to the confirmatory universe")
    parser.add_argument("--tag", default=None,
                        help="suffix for the output folder, to keep variant "
                             "runs from overwriting each other")
    return parser.parse_args()


def build_bundle(cfg, market: str):
    """Mirror the paired-run bundle construction in deep_baselines.py."""
    split_dates = derive_fixed_split_dates(
        cfg, market, max(cfg.horizons), IV_COLUMN
    )
    return prepare_data(
        cfg,
        market,
        cfg.primary_horizon,
        split_dates,
        core_set="full",
        iv_column=IV_COLUMN,
        corrector_iv_mode=cfg.paired_corrector_iv_mode,
    )


def resolve_markets(cfg, override) -> list[str]:
    if override:
        return list(override)
    return list(getattr(cfg, "confirmatory_markets", DEFAULT_EVAL_MARKETS))


def check_features(bundle, expected: int | None) -> None:
    if expected is not None and bundle.n_features != expected:
        raise AssertionError(
            f"{bundle.index_name}: expected {expected} paired features, "
            f"got {bundle.n_features}"
        )


def run_verify(cfg, markets) -> None:
    for market in markets:
        bundle = build_bundle(cfg, market)
        check_features(bundle, cfg.expected_paired_features)
        garch = verify_reference(bundle, "garch")
        core = verify_reference(bundle, "core")
        print(f"{market:<4} features={bundle.n_features} "
              f"core_names={bundle.core_names}")
        print(f"     GARCH ref  mean={garch['mean_logvol']:+.3f} "
              f"sd={garch['std_logvol']:.3f} "
              f"range=[{garch['min_logvol']:+.3f}, {garch['max_logvol']:+.3f}]")
        print(f"     core  ref  reproduces core_logrv "
              f"(max dev {core['max_abs_dev_from_core_logrv']:.2e})")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    markets = resolve_markets(cfg, args.markets)

    folder = "hybrid_baselines" + (f"_{args.tag}" if args.tag else "")
    root = Path(cfg.output_dir) / folder
    root.mkdir(parents=True, exist_ok=True)

    if args.stage == "verify":
        run_verify(cfg, markets)
        return

    if "ginn" in args.kinds:
        print(f"[config] GINN lambda = {args.lam} "
              f"(fixed; reference = {args.ginn_reference})")

    rows, prediction_store = [], {}
    reported_capacity = False
    for market in markets:
        bundle = build_bundle(cfg, market)
        check_features(bundle, cfg.expected_paired_features)
        if not reported_capacity:
            print(f"[features] {bundle.n_features} shared encoder channels")
            print("[capacity]",
                  parameter_report(cfg, bundle, args.backbone, args.input_append))
            reported_capacity = True
        for kind in args.kinds:
            payload = run_baseline_ensemble(
                cfg, bundle, kind,
                corrector=args.backbone,
                lam=args.lam,
                reference=args.ginn_reference,
                append=args.input_append,
                artifact_dir=root / market / kind,
            )
            predictions = payload.pop("_predictions")
            prediction_store[f"{market}:{kind}:test"] = predictions["test"]
            prediction_store[f"{market}:{kind}:test_true"] = predictions["test_true"]
            prediction_store[f"{market}:{kind}:test_dates"] = np.asarray(
                bundle.dates["test"], dtype="datetime64[ns]"
            )
            rows.append({
                "market": market,
                "kind": kind,
                "backbone": args.backbone,
                "variant": (args.ginn_reference if kind == "ginn"
                            else args.input_append),
                "lam": payload["lam"] if kind == "ginn" else "",
                "params": payload["trainable_parameters"],
                "ensemble_test_QLIKE": payload["ensemble_test"]["QLIKE"],
                "seed_mean_QLIKE": payload["seed_test_mean"]["QLIKE"],
                "seed_sd_QLIKE": payload["seed_test_std"]["QLIKE"],
                "ensemble_val_QLIKE": payload["ensemble_validation"]["QLIKE"],
            })
            print(f"[done] {market:<4} {kind:<6} "
                  f"test QLIKE = {rows[-1]['ensemble_test_QLIKE']:.4f}")

    if not rows:
        raise SystemExit("no runs completed; check --kinds and --markets")

    csv_path = root / "hybrid_baselines.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(root / "dm_inputs.npz", **prediction_store)

    print(f"\nwrote {csv_path}")
    for kind in args.kinds:
        scores = [r["ensemble_test_QLIKE"] for r in rows if r["kind"] == kind]
        if scores:
            print(f"  equal-weight mean QLIKE, {kind:<6} = {np.mean(scores):.4f}")


if __name__ == "__main__":
    main()
