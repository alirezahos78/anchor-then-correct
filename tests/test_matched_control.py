from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from volatility_pipeline.config import ExperimentConfig
from volatility_pipeline.deep_baselines import _check_alignment, primary_qlike_with_har
from volatility_pipeline.models import AnchoredCorrector
from volatility_pipeline.training import train_seed, run_ensemble, _load_seed


def tiny_config(**updates) -> ExperimentConfig:
    cfg = ExperimentConfig(
        seq_len=3,
        d_model=4,
        n_layers=1,
        d_state=2,
        expand=1,
        d_conv=2,
        scan_chunk=3,
        hidden=4,
        n_scans=1,
        epochs=0,
        patience=0,
        batch_size=2,
        seeds=[1],
        device="cpu",
    )
    return cfg.with_updates(**updates) if updates else cfg


def build_model(*, use_core: bool) -> AnchoredCorrector:
    return AnchoredCorrector(
        tiny_config(),
        n_features=41,
        n_core=5,
        core_coef=(np.arange(5, dtype="float32") / 10.0, 0.25),
        corrector="bimamba",
        freeze_core=True,
        anchor=use_core,
        use_core=use_core,
        zero_init_readout=True,
    )


class TinyBundle:
    def __init__(self) -> None:
        x = torch.zeros(2, 3, 2)
        core = torch.tensor([[1.0, 2.0], [0.5, -1.0]])
        y = torch.zeros(2, 1, 1)
        self.datasets = {name: (x.clone(), core.clone(), y.clone()) for name in ("train", "val", "test")}
        self.n_features = 2
        self.n_core = 2
        self.core_coef = (np.array([0.4, -0.2], dtype="float32"), 0.1)
        self.target_scaler = SimpleNamespace(
            mean=2.0,
            std=0.5,
            inverse=lambda values: np.asarray(values) * 0.5 + 2.0,
        )
        core_standardized = core.numpy() @ self.core_coef[0] + self.core_coef[1]
        physical_core = self.target_scaler.inverse(core_standardized)
        self.core_logrv = {"val": physical_core.copy(), "test": physical_core.copy()}
        self.index_name = "SYNTH"
        self.dates = {name: np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[ns]")
                      for name in ("train", "val", "test")}

    def loaders(self, batch_size: int):
        return {
            name: torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(*values), batch_size=batch_size, shuffle=False
            )
            for name, values in self.datasets.items()
        }


class MatchedControlTests(unittest.TestCase):
    def test_zero_initialized_modes_share_trainable_initialization(self) -> None:
        torch.manual_seed(17)
        residual = build_model(use_core=True)
        torch.manual_seed(17)
        direct = build_model(use_core=False)

        residual_trainable = {
            name: value.detach().clone()
            for name, value in residual.named_parameters()
            if value.requires_grad
        }
        direct_trainable = {
            name: value.detach().clone()
            for name, value in direct.named_parameters()
            if value.requires_grad
        }
        self.assertEqual(residual_trainable.keys(), direct_trainable.keys())
        self.assertTrue(
            all(torch.equal(residual_trainable[name], direct_trainable[name]) for name in residual_trainable)
        )

        x = torch.randn(2, 3, 41)
        core = torch.randn(2, 5)
        self.assertTrue(torch.equal(direct(x, core), torch.zeros(2, 1, 1)))
        expected_core = residual.core(core).unsqueeze(-1)
        self.assertTrue(torch.allclose(residual(x, core), expected_core))

    def test_shared_epoch_zero_rule_has_expected_starting_forecasts(self) -> None:
        cfg = tiny_config()
        bundle = TinyBundle()
        residual = train_seed(
            cfg,
            bundle,
            1,
            anchor=True,
            use_core=True,
            zero_init_readout=True,
        )
        direct = train_seed(
            cfg,
            bundle,
            1,
            anchor=False,
            use_core=False,
            zero_init_readout=True,
        )
        self.assertEqual(residual.best_epoch, 0)
        self.assertEqual(direct.best_epoch, 0)
        self.assertEqual(residual.initial_trainable_hash, direct.initial_trainable_hash)
        np.testing.assert_allclose(residual.validation_prediction, bundle.core_logrv["val"])
        np.testing.assert_allclose(direct.validation_prediction, bundle.target_scaler.mean)

    def test_completed_seed_cache_reuses_predictions_without_retraining(self) -> None:
        cfg, bundle = tiny_config(), TinyBundle()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            first = run_ensemble(cfg, bundle, artifact_dir=path, cache_signature="test-signature", use_guard=False)
            with patch("volatility_pipeline.training.train_seed", side_effect=AssertionError("unexpected retraining")):
                second = run_ensemble(cfg, bundle, artifact_dir=path, cache_signature="test-signature", use_guard=False)
            np.testing.assert_array_equal(first.test_prediction, second.test_prediction)
            self.assertTrue((path / "seed_1_predictions.csv").exists())
            self.assertIsNone(_load_seed(path, "changed-signature", bundle, 1))
            (path / "seed_1_predictions.npz").write_bytes(b"interrupted file")
            self.assertIsNone(_load_seed(path, "test-signature", bundle, 1))

    def test_each_backbone_can_train_one_epoch_in_both_modes(self) -> None:
        cfg = tiny_config(epochs=1)
        bundle = TinyBundle()
        for backbone in cfg.correctors:
            with self.subTest(backbone=backbone):
                residual = train_seed(cfg, bundle, 1, corrector=backbone, anchor=True,
                                      use_core=True, zero_init_readout=True)
                direct = train_seed(cfg, bundle, 1, corrector=backbone, anchor=False,
                                    use_core=False, zero_init_readout=True)
                self.assertEqual(residual.initial_trainable_hash, direct.initial_trainable_hash)
                self.assertTrue(np.isfinite(residual.test_prediction).all())
                self.assertTrue(np.isfinite(direct.test_prediction).all())
                self.assertLessEqual(residual.validation["QLIKE"], residual.initial_validation["QLIKE"] + 1e-8)
                self.assertLessEqual(direct.validation["QLIKE"], direct.initial_validation["QLIKE"] + 1e-8)

    def test_alignment_rejects_feature_asymmetry(self) -> None:
        common = dict(
            validation_dates=np.array(["2026-01-01"], dtype="datetime64[D]"),
            test_dates=np.array(["2026-01-02"], dtype="datetime64[D]"),
            validation_true=np.array([1.0]),
            test_true=np.array([1.1]),
        )
        data = {
            "sizes": {"train": 2, "val": 1, "test": 1},
            "date_ranges": {},
            "corrector_iv_mode": "selected",
        }
        residual = SimpleNamespace(**common, metadata={"data": {**data, "features": ["a", "ANCHOR"]}})
        direct = SimpleNamespace(**common, metadata={"data": {**data, "features": ["a"]}})
        with self.assertRaisesRegex(AssertionError, "features"):
            _check_alignment(residual, direct, "SYNTH", "bimamba")

    def test_primary_table_promotes_har_and_adds_equal_weight_average(self) -> None:
        metrics = pd.DataFrame(
            [
                {"market": "A", "backbone": "bimamba", "QLIKE_core": 0.5, "QLIKE_har": 0.4,
                 "QLIKE_garch": 0.45, "QLIKE_direct": 0.6, "QLIKE_residual": 0.35},
                {"market": "B", "backbone": "bimamba", "QLIKE_core": 0.7, "QLIKE_har": 0.6,
                 "QLIKE_garch": 0.55, "QLIKE_direct": 0.8, "QLIKE_residual": 0.50},
            ]
        )
        table = primary_qlike_with_har(metrics, "bimamba", ["A", "B"])
        self.assertEqual(
            list(table.columns), ["Market", "Core OLS", "HAR", "GARCH", "SSM-D", "SSM-R"]
        )
        self.assertEqual(list(table["Market"]), ["A", "B", "Average"])
        self.assertAlmostEqual(table.loc[2, "HAR"], 0.5)


if __name__ == "__main__":
    unittest.main()
