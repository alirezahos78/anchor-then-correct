from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from volatility_pipeline.artifacts import file_sha256, write_json_atomic, write_npz_atomic
from volatility_pipeline.reporting import (
    REPORTED_METRICS, aggregate_seed_metrics, primary_qlike_with_har,
    _latex_primary_qlike_with_har,
)


class ReportingTests(unittest.TestCase):
    def market_metrics(self, backbone="bimamba"):
        return pd.DataFrame([
            {"market": "A", "backbone": backbone, "QLIKE_core": .5, "QLIKE_har": .4,
             "QLIKE_garch": .45, "QLIKE_direct": .6, "QLIKE_residual": .35},
            {"market": "B", "backbone": backbone, "QLIKE_core": .7, "QLIKE_har": .6,
             "QLIKE_garch": .55, "QLIKE_direct": .8, "QLIKE_residual": .5},
        ])

    def seed_metrics(self, selected_seeds=(1, 2, 3)):
        # Each market varies across seeds, but their equal-weight mean is stable.
        # Averaging market SDs would incorrectly report nonzero uncertainty.
        rows = []
        for market in ("A", "B"):
            for mode in ("direct", "residual"):
                for seed in selected_seeds:
                    score = (seed if market == "A" else 6 - seed) + (mode == "direct")
                    rows.append({
                        "market": market, "backbone": "bimamba", "backbone_label": "BiMamba-style",
                        "mode": mode, "seed": seed,
                        **{f"test_{metric}": float(score) for metric in REPORTED_METRICS},
                    })
        return pd.DataFrame(rows)

    def test_har_table_retains_order_and_equal_weight_mean(self):
        table = primary_qlike_with_har(self.market_metrics(), "bimamba", ["B", "A"])
        self.assertEqual(list(table["Market"]), ["B", "A", "Average"])
        self.assertAlmostEqual(table.loc[2, "HAR"], .5)
        self.assertAlmostEqual(table.loc[2, "SSM-R"], .425)
        self.assertIn("HAR", _latex_primary_qlike_with_har(table))

    def test_primary_table_rejects_missing_or_duplicate_markets(self):
        frame = self.market_metrics()
        for invalid in (frame.iloc[:1], pd.concat([frame.iloc[:1], frame.iloc[:1]])):
            with self.assertRaises(AssertionError):
                primary_qlike_with_har(invalid, "bimamba", ["A", "B"])

    def test_non_ssm_backbone_is_not_mislabeled(self):
        table = primary_qlike_with_har(self.market_metrics("lstm"), "lstm", ["A", "B"])
        self.assertIn("BiLSTM-D", table.columns)
        self.assertNotIn("SSM-D", _latex_primary_qlike_with_har(table))

    def test_seed_sd_uses_market_average_within_each_seed(self):
        outputs = aggregate_seed_metrics(self.seed_metrics(), pd.DataFrame(columns=["backbone"]))
        row = outputs["paired_mean_std"].iloc[0]
        self.assertEqual(row["n_seeds"], 3)
        self.assertAlmostEqual(row["QLIKE_direct_mean"], 4)
        self.assertAlmostEqual(row["QLIKE_residual_mean"], 3)
        self.assertAlmostEqual(row["QLIKE_residual_std_sample"], 0)
        market = outputs["market_mean_std"].iloc[0]
        self.assertAlmostEqual(market["test_QLIKE_std_sample"], 1)

    def test_single_seed_sd_is_undefined(self):
        outputs = aggregate_seed_metrics(self.seed_metrics((1,)), pd.DataFrame(columns=["backbone"]))
        self.assertTrue(np.isnan(outputs["paired_mean_std"].iloc[0]["QLIKE_direct_std_sample"]))

    def test_atomic_artifacts_roundtrip_and_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_npz_atomic(path / "predictions.npz", predictions=np.array([1., 2.]))
            with np.load(path / "predictions.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["predictions"], [1., 2.])
            first_hash = file_sha256(path / "predictions.npz")
            write_npz_atomic(path / "predictions.npz", predictions=np.array([2., 3.]))
            self.assertNotEqual(first_hash, file_sha256(path / "predictions.npz"))
            write_json_atomic(path / "result.json", {"complete": True})
            self.assertTrue(json.loads((path / "result.json").read_text())["complete"])
            self.assertFalse(list(path.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
