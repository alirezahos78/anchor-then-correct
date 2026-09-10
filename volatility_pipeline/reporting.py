from __future__ import annotations

import numpy as np
import pandas as pd


BACKBONE_LABELS = {
    "bimamba": "BiMamba-style",
    "lstm": "BiLSTM",
    "gru": "BiGRU",
    "mlp": "Window MLP",
    "patchtf": "PatchTransformer-style",
    "tsmixer": "TSMixer",
    "itransformer": "iTransformer-style",
}



REPORTED_METRICS = ("QLIKE", "RMSE", "MAE", "R2", "Corr")



def aggregate_seed_metrics(
    seeds: pd.DataFrame, ensemble_summary: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Build all mean/SD tables from the explicitly retained seed-level rows."""
    required = {
        "market",
        "backbone",
        "backbone_label",
        "mode",
        "seed",
        *(f"test_{metric}" for metric in REPORTED_METRICS),
    }
    missing = sorted(required - set(seeds.columns))
    if missing:
        raise ValueError(f"seed-level output is incomplete; missing columns: {missing}")

    group_market = ["market", "backbone", "backbone_label", "mode"]
    test_columns = [f"test_{metric}" for metric in REPORTED_METRICS]
    market_mean_std = (
        seeds.groupby(group_market, sort=False)[test_columns]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    market_mean_std.columns = [
        "_".join(str(part) for part in column if part).replace("_std", "_std_sample")
        if isinstance(column, tuple)
        else column
        for column in market_mean_std.columns
    ]

    equal_weight_by_seed = (
        seeds.groupby(["backbone", "backbone_label", "mode", "seed"], sort=False)[test_columns]
        .mean()
        .reset_index()
        .rename(columns={column: column.replace("test_", "equal_weight_") for column in test_columns})
    )
    equal_weight_columns = [f"equal_weight_{metric}" for metric in REPORTED_METRICS]
    backbone_mode_mean_std = (
        equal_weight_by_seed.groupby(["backbone", "backbone_label", "mode"], sort=False)[
            equal_weight_columns
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    backbone_mode_mean_std.columns = [
        "_".join(str(part) for part in column if part).replace("_std", "_std_sample")
        if isinstance(column, tuple)
        else column
        for column in backbone_mode_mean_std.columns
    ]

    paired_rows: list[dict] = []
    ensemble_lookup = ensemble_summary.set_index("backbone")
    for backbone in equal_weight_by_seed["backbone"].drop_duplicates():
        backbone_rows = backbone_mode_mean_std.loc[
            backbone_mode_mean_std["backbone"] == backbone
        ].set_index("mode")
        if not {"direct", "residual"} <= set(backbone_rows.index):
            raise AssertionError(f"{backbone}: direct/residual seed aggregates are incomplete")
        direct = backbone_rows.loc["direct"]
        residual = backbone_rows.loc["residual"]
        row = {
            "backbone": backbone,
            "backbone_label": BACKBONE_LABELS[backbone],
        }
        for metric in REPORTED_METRICS:
            base = f"equal_weight_{metric}"
            row[f"{metric}_direct_mean"] = float(direct[f"{base}_mean"])
            row[f"{metric}_direct_std_sample"] = float(direct[f"{base}_std_sample"])
            row[f"{metric}_residual_mean"] = float(residual[f"{base}_mean"])
            row[f"{metric}_residual_std_sample"] = float(residual[f"{base}_std_sample"])
        row["direct_minus_residual_QLIKE"] = row["QLIKE_direct_mean"] - row["QLIKE_residual_mean"]
        row["relative_gain_vs_direct_percent"] = (
            100.0 * row["direct_minus_residual_QLIKE"] / row["QLIKE_direct_mean"]
            if row["QLIKE_direct_mean"] != 0
            else float("nan")
        )
        row["n_seeds"] = int(direct["equal_weight_QLIKE_count"])
        if backbone in ensemble_lookup.index:
            extra = ensemble_lookup.loc[backbone]
            row["parameters_residual_median"] = int(extra["parameters_residual_median"])
            row["parameters_direct_median"] = int(extra["parameters_direct_median"])
            row["residual_wins_vs_direct"] = int(extra["residual_wins_vs_direct"])
            row["n_markets"] = int(extra["n_markets"])
        paired_rows.append(row)

    return {
        "market_mean_std": market_mean_std,
        "equal_weight_by_seed": equal_weight_by_seed,
        "backbone_mode_mean_std": backbone_mode_mean_std,
        "paired_mean_std": pd.DataFrame(paired_rows),
    }



def primary_qlike_with_har(
    metrics: pd.DataFrame, primary_backbone: str, market_order: list[str]
) -> pd.DataFrame:
    """Create the paper-facing QLIKE rows, including the previously omitted HAR baseline."""
    selected = metrics.loc[metrics["backbone"] == primary_backbone].copy()
    if len(selected) != len(market_order) or set(selected["market"]) != set(market_order):
        raise AssertionError("primary QLIKE table does not contain exactly one row per market")
    selected = selected.set_index("market").loc[market_order]
    label = "SSM" if primary_backbone == "bimamba" else BACKBONE_LABELS[primary_backbone]
    table = selected[
        ["QLIKE_core", "QLIKE_har", "QLIKE_garch", "QLIKE_direct", "QLIKE_residual"]
    ].rename(
        columns={
            "QLIKE_core": "Core OLS",
            "QLIKE_har": "HAR",
            "QLIKE_garch": "GARCH",
            "QLIKE_direct": f"{label}-D",
            "QLIKE_residual": f"{label}-R",
        }
    )
    table.index.name = "Market"
    table.loc["Average"] = table.mean(axis=0)
    return table.reset_index()



def _latex_primary_qlike_with_har(table: pd.DataFrame) -> str:
    """LaTeX fragment without bolding, so shallow columns can be merged later."""
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        " & ".join(table.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(str(row["Market"]) + " & " + " & ".join(
            f"{row[column]:.4f}" for column in table.columns[1:]
        ) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)

