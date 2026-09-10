from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from data import build_market_dataset as builder
from run_deep_baselines import ensure_data, verify_frozen_data


class AutomaticDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "configs").mkdir()
        source_root = Path(__file__).resolve().parents[1]
        self.universe = json.loads((source_root / "configs/market_universe.json").read_text())
        (self.root / "configs/market_universe.json").write_text(json.dumps(self.universe))
        self.config = {
            "data_dir": "data/vol_dataset", "market_universe": "configs/market_universe.json",
            "primary_horizon": 5,
        }
        self.markets = ["SPX", "NDX", "RUT", "DIA", "GLD", "USO", "EEM"]

    def write_valid_fixture(self, directory: Path, markets=None):
        directory.mkdir(parents=True, exist_ok=True)
        files = {}
        for market in markets or self.markets:
            path = directory / f"{market}_vol.csv"
            path.write_text("Date,Close\n2020-01-01,100\n")
            files[market] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        manifest = {
            "download_start_inclusive": self.universe["start_inclusive"],
            "download_end_exclusive": self.universe["end_exclusive"],
            "requested_cutoff_inclusive": self.universe["requested_cutoff_inclusive"],
            "target_horizon": 5, "omitted_markets": [], "files": files,
        }
        (directory / "data_manifest.json").write_text(json.dumps(manifest))

    @staticmethod
    def synthetic_prices(ticker, start, end):
        # Synthetic fixtures validate software behavior; they are never paper results.
        rng = np.random.default_rng(123)
        dates = pd.bdate_range("2020-01-01", periods=300)
        close = 100 * np.exp(np.cumsum(rng.normal(0, .01, len(dates))))
        return pd.DataFrame({
            "Open": close * .999, "High": close * 1.01, "Low": close * .99,
            "Close": close, "Volume": rng.integers(1000, 100000, len(dates)).astype(float),
        }, index=dates)

    def test_fresh_start_builds_all_requested_csvs_and_manifest(self):
        def build_locally(config, project_root, markets, destination):
            builder.build(project_root / config["market_universe"], destination, 5,
                          markets=markets, cache_dir=project_root / "data/raw_cache")
        with patch("run_deep_baselines._build_frozen_data", side_effect=build_locally), \
             patch.object(builder, "download_one", side_effect=self.synthetic_prices) as download, \
             redirect_stdout(io.StringIO()):
            result = ensure_data(self.config, self.root, self.markets)
        self.assertEqual(result, self.root / "data/vol_dataset")
        verify_frozen_data({**self.config, "data_dir": str(result)}, self.root, self.markets)
        self.assertEqual(sorted(path.stem for path in result.glob("*_vol.csv")),
                         sorted(f"{market}_vol" for market in self.markets))
        requested = {call.args[0] for call in download.call_args_list}
        self.assertFalse(requested.intersection({"SPY", "QQQ", "IWM", "^RVX", "^VXEEM"}))
        self.assertTrue(all(call.args[2] == "2026-08-17" for call in download.call_args_list))
        for market in self.markets:
            frame = pd.read_csv(result / f"{market}_vol.csv", index_col=0)
            selected = frame.drop(columns=["Close", "target_logrv", "target_rv", "VIX_REFERENCE", "VIX_REFERENCE_chg"])
            self.assertEqual(len(selected.columns), 38)

    def test_valid_data_subdirectory_is_reused_without_download(self):
        self.write_valid_fixture(self.root / "data/vol_dataset")
        with patch("run_deep_baselines._build_frozen_data") as build, redirect_stdout(io.StringIO()):
            selected = ensure_data(self.config, self.root, self.markets)
        build.assert_not_called()
        self.assertEqual(selected, self.root / "data/vol_dataset")

    def test_legacy_location_is_reused_without_moving_files(self):
        self.write_valid_fixture(self.root / "vol_dataset")
        with patch("run_deep_baselines._build_frozen_data") as build, redirect_stdout(io.StringIO()):
            selected = ensure_data(self.config, self.root, self.markets)
        build.assert_not_called()
        self.assertEqual(selected, self.root / "vol_dataset")

    def test_failed_download_leaves_previous_files_intact(self):
        old = self.root / "data/vol_dataset"
        old.mkdir(parents=True)
        (old / "SPX_vol.csv").write_text("old data\n")
        with patch("run_deep_baselines._build_frozen_data", side_effect=RuntimeError("offline")), \
             redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "offline"):
            ensure_data(self.config, self.root, self.markets)
        self.assertEqual((old / "SPX_vol.csv").read_text(), "old data\n")
        self.assertFalse((old / "data_manifest.json").exists())
        self.assertFalse(list(old.parent.glob(".vol_dataset_building_*")))

    def test_invalid_previous_dataset_is_preserved_when_new_build_succeeds(self):
        old = self.root / "data/vol_dataset"
        old.mkdir(parents=True)
        (old / "SPX_vol.csv").write_text("old data\n")
        with patch("run_deep_baselines._build_frozen_data",
                   side_effect=lambda cfg, root, markets, dest: self.write_valid_fixture(dest, markets)), \
             redirect_stdout(io.StringIO()):
            ensure_data(self.config, self.root, self.markets)
        backups = list(old.parent.glob("vol_dataset_backup_*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual((backups[0] / "SPX_vol.csv").read_text(), "old data\n")
        verify_frozen_data(self.config, self.root, self.markets)

    def test_incomplete_new_build_is_never_published(self):
        with patch("run_deep_baselines._build_frozen_data",
                   side_effect=lambda cfg, root, markets, dest: self.write_valid_fixture(dest, ["SPX"])), \
             redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            ensure_data(self.config, self.root, self.markets)
        self.assertFalse((self.root / "data/vol_dataset").exists())

    def test_raw_download_cache_is_reused_but_changed_cutoff_is_not(self):
        cache = self.root / "raw_cache"
        with patch.object(builder, "download_one", side_effect=self.synthetic_prices) as download, \
             redirect_stdout(io.StringIO()):
            first = builder.cached_download("^VIX", "2010-01-01", "2026-08-17", cache)
            second = builder.cached_download("^VIX", "2010-01-01", "2026-08-17", cache)
            self.assertEqual(download.call_count, 1)
            pd.testing.assert_frame_equal(first, second, check_freq=False, check_names=False)
            builder.cached_download("^VIX", "2010-01-01", "2026-08-18", cache)
            self.assertEqual(download.call_count, 2)

    def test_download_failure_retries_and_preserves_successful_raw_cache(self):
        frame = self.synthetic_prices("^VIX", "2010-01-01", "2026-08-17")
        with patch.object(builder, "download_one", side_effect=[RuntimeError("temporary"), frame]) as download, \
             patch.object(builder.time, "sleep"), redirect_stdout(io.StringIO()):
            result = builder.cached_download("^VIX", "2010-01-01", "2026-08-17", self.root / "raw_cache")
        self.assertEqual(download.call_count, 2)
        self.assertEqual(len(result), 300)


if __name__ == "__main__":
    unittest.main()
