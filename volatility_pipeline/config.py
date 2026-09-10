from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    data_dir: str = "vol_dataset"
    output_dir: str = "artifacts"
    indices: list[str] = field(default_factory=lambda: ["DJI", "IXIC", "NYA"])
    horizons: list[int] = field(default_factory=lambda: [1, 5, 10, 21])
    primary_horizon: int = 5
    seeds: list[int] = field(default_factory=lambda: [1, 2, 3])
    correctors: list[str] = field(
        default_factory=lambda: ["bimamba", "lstm", "gru", "mlp", "patchtf", "tsmixer", "itransformer"]
    )
    core_sets: list[str] = field(default_factory=lambda: ["har", "har_garch", "har_iv", "full"])
    iv_columns: list[str] = field(default_factory=lambda: ["VIX", "VXN"])
    corrector_iv_mode: str = "none"  # main result: no IV is routed around the core
    paired_corrector_iv_mode: str = "selected"
    matched_zero_init_readout: bool = True
    expected_paired_features: int | None = None

    seq_len: int = 60
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    batch_size: int = 32
    epochs: int = 100
    patience: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    grad_clip: float = 5.0
    train_loss: str = "qlike"
    selection_metric: str = "QLIKE"
    ensemble_guard: bool = True

    d_model: int = 32
    n_layers: int = 3
    d_state: int = 128
    expand: int = 2
    d_conv: int = 3
    scan_chunk: int = 32
    hidden: int = 32
    n_scans: int = 2

    rolling_folds: int = 3
    rolling_initial_train_ratio: float = 0.55
    rolling_val_ratio: float = 0.15
    rolling_test_ratio: float = 0.10
    device: str = "auto"
    force: bool = False

    # ICASSP 2027 mismatch-aware confirmation protocol.  These defaults keep
    # old configs valid; the dedicated config enables the new section.
    market_universe: str = "configs/market_universe.json"
    discovery_markets: list[str] = field(default_factory=lambda: ["DJI", "IXIC", "NYA"])
    confirmatory_markets: list[str] = field(
        default_factory=lambda: ["SPX", "NDX", "RUT", "DIA", "GLD", "USO", "EEM"]
    )
    robustness_markets: list[str] = field(default_factory=lambda: ["SPY", "QQQ", "IWM"])
    confirmatory_corrector: str = "bimamba"
    mismatch_score: str = "validation_core_qlike"
    mismatch_threshold: float = 0.28408271371139526
    mismatch_direction: str = "greater_equal"
    primary_table_count: int = 3
    primary_table_selection: str = "validation_mismatch_score"
    ridge_alphas: list[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0, 100.0])
    xgb_n_estimators: int = 400
    xgb_max_depth: int = 3
    xgb_learning_rate: float = 0.03
    xgb_subsample: float = 1.0
    xgb_colsample_bytree: float = 1.0
    bootstrap_repetitions: int = 2000
    bootstrap_block_length: int = 10
    report_top_k: int = 3

    def validate(self) -> None:
        if self.corrector_iv_mode not in {"none", "all", "selected"}:
            raise ValueError("corrector_iv_mode must be none, all, or selected")
        if self.paired_corrector_iv_mode not in {"all", "selected"}:
            raise ValueError("paired_corrector_iv_mode must retain IV information")
        if not self.matched_zero_init_readout:
            raise ValueError("matched-control runs require zero-initialized readouts in both modes")
        if self.expected_paired_features is not None and self.expected_paired_features < 1:
            raise ValueError("expected_paired_features must be positive")
        if self.train_loss not in {"mse", "qlike"}:
            raise ValueError("train_loss must be mse or qlike")
        if self.selection_metric not in {"QLIKE", "RMSE"}:
            raise ValueError("selection_metric must be QLIKE or RMSE")
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-9:
            raise ValueError("train/val/test ratios must sum to one")
        if self.primary_horizon not in self.horizons:
            raise ValueError("primary_horizon must be included in horizons")
        if self.mismatch_direction not in {"greater_equal", "less_equal"}:
            raise ValueError("mismatch_direction must be greater_equal or less_equal")
        if self.primary_table_selection != "validation_mismatch_score":
            raise ValueError("test-selected paper tables are forbidden; use validation_mismatch_score")
        if self.primary_table_count < 1:
            raise ValueError("primary_table_count must be positive")
        if self.primary_table_count > len(self.confirmatory_markets):
            raise ValueError("primary_table_count cannot exceed the frozen confirmatory set")
        groups = self.discovery_markets + self.confirmatory_markets + self.robustness_markets
        if len(groups) != len(set(groups)):
            raise ValueError("market groups must not overlap")
        if not self.ridge_alphas or any(alpha <= 0 for alpha in self.ridge_alphas):
            raise ValueError("ridge_alphas must contain positive values")

    def with_updates(self, **updates: Any) -> "ExperimentConfig":
        out = replace(self, **updates)
        out.validate()
        return out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    cfg = ExperimentConfig(**raw)
    cfg.validate()
    return cfg


def save_config(cfg: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(cfg.to_dict(), fh, indent=2, sort_keys=True)
