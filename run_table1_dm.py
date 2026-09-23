#!/usr/bin/env python3
"""Diebold-Mariano tests for every row of Table 1, with Holm correction.

Reuses ``volatility_pipeline.statistics`` so the numbers are produced exactly as
every other test in the paper: daily QLIKE differences, Newey-West/Bartlett
long-run variance at h-1 lags, the Harvey-Leybourne-Newbold small-sample
correction, and a two-sided t reference.

Family-wise scheme
------------------
Each (backbone, comparison) pair is one Holm family across the seven markets,
matching the protocol already described in the paper.  Families are mutually
independent, so adding comparisons never changes a previously reported
adjusted p-value.

Covered comparisons
-------------------
    ours_vs_direct    anchored output vs its paired direct counterpart
    ours_vs_core      anchored output vs the frozen OLS core
    ours_vs_garch     anchored output vs GARCH
    ours_vs_har       anchored output vs HAR
    direct_vs_garch   direct deep learning vs GARCH
    ours_vs_ginn      anchored output vs the loss-level hybrid
    ours_vs_input     anchored output vs the input-level hybrid
    ginn_vs_direct    loss-level hybrid vs direct deep learning
    input_vs_direct   input-level hybrid vs direct deep learning
    ginn_vs_garch     loss-level hybrid vs GARCH
    input_vs_garch    input-level hybrid vs GARCH

Hybrid comparisons run only for backbones that have a hybrid-baseline folder;
the rest are skipped silently.

Usage
-----
    python run_table1_dm.py --config configs/matched_control.json

Writes ``<output_dir>/table1_dm_tests.csv`` and prints the dagger assignments
for Table 1.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from volatility_pipeline.config import load_config
from volatility_pipeline.statistics import adjust_pvalues, daily_qlike, dm_test

DEFAULT_MARKETS = ["SPX", "NDX", "RUT", "DIA", "GLD", "USO", "EEM"]

# Table 1 row labels, keyed by the internal corrector name.
BACKBONE_LABEL = {
    "bimamba": "BiMamba",
    "lstm": "BiLSTM",
    "gru": "BiGRU",
    "mlp": "Window MLP",
    "patchtf": "PatchTransformer",
    "tsmixer": "TSMixer",
    "itransformer": "iTransformer",
}

# (label, method_a, method_b, needs_hybrid).
# A negative mean difference favours method_a.
COMPARISONS = [
    ("ours_vs_direct", "ours", "direct", False),
    ("ours_vs_core", "ours", "core", False),
    ("ours_vs_garch", "ours", "garch", False),
    ("ours_vs_har", "ours", "har", False),
    ("direct_vs_garch", "direct", "garch", False),
    ("ours_vs_ginn", "ours", "ginn", True),
    ("ours_vs_input", "ours", "input", True),
    ("ginn_vs_direct", "ginn", "direct", True),
    ("input_vs_direct", "input", "direct", True),
    ("ginn_vs_garch", "ginn", "garch", True),
    ("input_vs_garch", "input", "garch", True),
]

# Markers already used in the Table 1 caption.
MARKERS = {"ours_vs_direct": "\\dagger", "ours_vs_core": "\\ddagger"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/matched_control.json")
    parser.add_argument("--backbones", nargs="*", default=None,
                        help="defaults to cfg.correctors, i.e. every Table 1 row")
    parser.add_argument("--baseline-dir", default="hybrid_baselines",
                        help="folder written by run_hybrid_baselines.py")
    parser.add_argument("--markets", nargs="*", default=None)
    parser.add_argument("--comparisons", nargs="*", default=None,
                        help="subset of: " + ", ".join(c[0] for c in COMPARISONS))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", default="table1_dm_tests.csv")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_case(root: Path, case_id: str) -> dict | None:
    """Read one paired case: {method: test prediction} plus dates and truth."""
    case_dir = root / "cases" / case_id
    result_path, prediction_path = case_dir / "result.json", case_dir / "predictions.npz"
    if not result_path.exists() or not prediction_path.exists():
        return None
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    with np.load(prediction_path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    return {
        "methods": {
            method: arrays[f"test_{key}"]
            for method, key in payload["method_keys"].items()
        },
        "dates": arrays["test_dates"],
        "true": arrays["test_true"],
    }


def load_baseline(root: Path, folder: str, market: str, kind: str) -> dict | None:
    path = root / folder / market / kind / "predictions.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as saved:
        return {
            "prediction": saved["test_prediction"],
            "dates": saved["test_dates"],
            "true": saved["test_true"],
        }


def collect(cfg, market: str, backbone: str, folder: str) -> dict | None:
    """Gather every method's test prediction for one market and backbone."""
    root = Path(cfg.output_dir)
    residual = load_case(root, f"matched_pair_{market}_{backbone}_residual")
    direct = load_case(root, f"matched_pair_{market}_{backbone}_direct")
    if residual is None or direct is None:
        return None

    dates, truth = residual["dates"], residual["true"]
    series = {"ours": residual["methods"]["hybrid"],
              "direct": direct["methods"]["hybrid"],
              "_true": truth}
    if not np.array_equal(dates, direct["dates"]):
        raise AssertionError(f"{market}/{backbone}: direct dates differ")

    # Econometric references travel with the case; core_ols is the frozen
    # fusion the anchored model retains, not a refitted regression.
    for stored, name in (("core_ols", "core"), ("garch", "garch"), ("har", "har")):
        if stored in residual["methods"]:
            series[name] = residual["methods"][stored]

    for kind in ("ginn", "input"):
        loaded = load_baseline(root, folder, market, kind)
        if loaded is None:
            continue
        if not np.array_equal(dates, loaded["dates"]):
            raise AssertionError(f"{market}/{kind}: test dates differ from the paired run")
        if not np.allclose(truth, loaded["true"], atol=1e-8):
            raise AssertionError(f"{market}/{kind}: targets differ from the paired run")
        series[kind] = loaded["prediction"]
    return series


# ---------------------------------------------------------------------------
# testing
# ---------------------------------------------------------------------------


def run_family(panel: dict, markets: list[str], backbone: str,
               label: str, a: str, b: str, horizon: int) -> list[dict] | None:
    if any(a not in panel[m] or b not in panel[m] for m in markets):
        return None
    family = []
    for market in markets:
        series = panel[market]
        loss_a = daily_qlike(series[a], series["_true"])
        loss_b = daily_qlike(series[b], series["_true"])
        result = dm_test(loss_a, loss_b, horizon)
        family.append({
            "backbone": backbone,
            "backbone_label": BACKBONE_LABEL.get(backbone, backbone),
            "comparison": label,
            "method_a": a,
            "method_b": b,
            "market": market,
            "mean_qlike_a": float(loss_a.mean()),
            "mean_qlike_b": float(loss_b.mean()),
            "mean_loss_a_minus_b": result.mean_difference,
            "dm": result.statistic,
            "p_raw": result.pvalue,
            "n": result.n,
            "nw_lag": result.lag,
        })
    for row, p in zip(family, adjust_pvalues([r["p_raw"] for r in family], "holm")):
        row["p_holm"] = float(p)
    return family


def summarise(family: list[dict], alpha: float) -> str:
    a, b = family[0]["method_a"], family[0]["method_b"]
    wins = sum(r["mean_loss_a_minus_b"] < 0 for r in family)
    for_a = sum(r["mean_loss_a_minus_b"] < 0 and r["p_holm"] < alpha for r in family)
    for_b = sum(r["mean_loss_a_minus_b"] > 0 and r["p_holm"] < alpha for r in family)
    finite = [r["p_holm"] for r in family if np.isfinite(r["p_holm"])]
    smallest = min(finite) if finite else float("nan")
    return (f"{a} lower in {wins}/{len(family)}; "
            f"{for_a} sig. for {a}, {for_b} for {b}; min adj p={smallest:.4f}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    markets = args.markets or list(
        getattr(cfg, "confirmatory_markets", DEFAULT_MARKETS)
    )
    backbones = args.backbones or list(cfg.correctors)
    wanted = args.comparisons or [c[0] for c in COMPARISONS]
    horizon = cfg.primary_horizon

    rows: list[dict] = []
    for backbone in backbones:
        panel = {}
        for market in markets:
            series = collect(cfg, market, backbone, args.baseline_dir)
            if series is None:
                break
            panel[market] = series
        if len(panel) != len(markets):
            print(f"[skip] {backbone}: paired cases not found for every market")
            continue

        has_hybrid = all("ginn" in panel[m] or "input" in panel[m] for m in markets)
        print(f"\n{BACKBONE_LABEL.get(backbone, backbone)}"
              + ("  (hybrid baselines present)" if has_hybrid else ""))
        for label, a, b, needs_hybrid in COMPARISONS:
            if label not in wanted:
                continue
            if needs_hybrid and not has_hybrid:
                continue
            family = run_family(panel, markets, backbone, label, a, b, horizon)
            if family is None:
                continue
            rows.extend(family)
            print(f"  {label:<18} {summarise(family, args.alpha)}")

    if not rows:
        raise SystemExit("no comparisons produced; check --backbones and --comparisons")

    out = Path(cfg.output_dir) / args.out
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}  ({len(rows)} tests, "
          f"{len({(r['backbone'], r['comparison']) for r in rows})} Holm families)")

    # ---- dagger assignments for the Ours block of Table 1 ----
    print("\nTable 1 markers for the Ours block "
          f"(significant at alpha={args.alpha}, improvement only)")
    print(f"{'backbone':<18}" + "".join(f"{m:>12}" for m in markets))
    index = {(r["backbone"], r["comparison"], r["market"]): r for r in rows}
    for backbone in backbones:
        if not any(r["backbone"] == backbone for r in rows):
            continue
        line = f"{BACKBONE_LABEL.get(backbone, backbone):<18}"
        for market in markets:
            marks = ""
            for comparison, symbol in MARKERS.items():
                row = index.get((backbone, comparison, market))
                if (row and np.isfinite(row["p_holm"])
                        and row["p_holm"] < args.alpha
                        and row["mean_loss_a_minus_b"] < 0):
                    marks += symbol.replace("\\", "")
            line += f"{marks or '-':>12}"
        print(line)
    print("  dagger = beats its direct counterpart, ddagger = beats the core")

    # ---- compact per-comparison adjusted p table ----
    print("\nadjusted p-values by family")
    for backbone in backbones:
        subset = [r for r in rows if r["backbone"] == backbone]
        if not subset:
            continue
        print(f"\n{BACKBONE_LABEL.get(backbone, backbone)}")
        print(f"  {'comparison':<18}" + "".join(f"{m:>9}" for m in markets))
        for label, _, _, _ in COMPARISONS:
            family = {r["market"]: r for r in subset if r["comparison"] == label}
            if not family:
                continue
            line = f"  {label:<18}"
            for market in markets:
                row = family[market]
                favourable = row["mean_loss_a_minus_b"] < 0
                mark = "*" if (row["p_holm"] < args.alpha and favourable) else (
                    "x" if row["p_holm"] < args.alpha else " ")
                line += f"{row['p_holm']:>8.4f}{mark}"
            print(line)
    print("\n  * significant and favours method_a;  x significant and favours method_b")


if __name__ == "__main__":
    main()
