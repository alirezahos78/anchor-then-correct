from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig


CORE_SETS = {
    "har": ["hd", "hw", "hm"],
    "har_garch": ["hd", "hw", "hm", "garch"],
    "har_iv": ["hd", "hw", "hm", "iv"],
    "full": ["hd", "hw", "hm", "iv", "garch"],
}
IV_FEATURES = {
    "ANCHOR",
    "ANCHOR_chg",
    "VIX_REFERENCE",
    "VIX_REFERENCE_chg",
    "VIX",
    "VIX_chg",
    "VXN",
    "VXN_chg",
    "RVX",
    "RVX_chg",
    "VXD",
    "VXD_chg",
    "GVZ",
    "GVZ_chg",
    "OVX",
    "OVX_chg",
    "VXEEM",
    "VXEEM_chg",
}


@dataclass(frozen=True)
class SplitDates:
    train_end: pd.Timestamp  # first validation origin
    val_end: pd.Timestamp  # first test origin
    test_end: pd.Timestamp | None = None  # exclusive, used by rolling folds

    def as_dict(self) -> dict[str, str | None]:
        return {
            "train_end_exclusive": str(self.train_end.date()),
            "validation_end_exclusive": str(self.val_end.date()),
            "test_end_exclusive": None if self.test_end is None else str(self.test_end.date()),
        }


class Standardizer:
    def fit(self, values: np.ndarray) -> "Standardizer":
        values = np.asarray(values, float)
        self.mean = values.mean(axis=0)
        self.std = values.std(axis=0)
        self.std = np.where(self.std == 0, 1.0, self.std)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values) - self.mean) / self.std


class Standardizer1D:
    def fit(self, values: np.ndarray) -> "Standardizer1D":
        values = np.asarray(values, float)
        self.mean = float(values.mean())
        self.std = float(values.std()) or 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values) - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values) * self.std + self.mean


@dataclass
class DataBundle:
    datasets: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    dates: dict[str, np.ndarray]
    true_logrv: dict[str, np.ndarray]
    core_logrv: dict[str, np.ndarray]
    last_features: dict[str, np.ndarray]
    feature_names: list[str]
    core_names: list[str]
    n_features: int
    n_core: int
    target_scaler: Standardizer1D
    core_coef: tuple[np.ndarray, float]
    split_dates: SplitDates
    frame: pd.DataFrame
    split_masks: dict[str, np.ndarray]
    index_name: str
    horizon: int
    iv_column: str
    corrector_iv_mode: str
    data_path: Path

    def loaders(self, batch_size: int) -> dict[str, torch.utils.data.DataLoader]:
        return {
            name: torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(x, c, y),
                batch_size=batch_size,
                shuffle=name == "train",
                drop_last=False,
            )
            for name, (x, c, y) in self.datasets.items()
        }


def rebuild_target(close: pd.Series, horizon: int) -> pd.Series:
    returns = np.log(close.astype(float)).diff()
    future_sum = returns.pow(2).rolling(horizon).sum().shift(-horizon)
    return np.log(np.sqrt(future_sum).replace(0, np.nan))


def target_end_dates(index: pd.Index, horizon: int) -> pd.Series:
    dates = pd.Series(pd.to_datetime(index), index=index)
    return dates.shift(-horizon)


def make_split_masks(
    origins: pd.DatetimeIndex,
    ends: pd.Series,
    split: SplitDates,
) -> dict[str, np.ndarray]:
    """Assign origins only when their complete target window stays in the split."""
    origin = pd.Series(origins, index=origins)
    train = (origin < split.train_end) & (ends < split.train_end)
    val = (origin >= split.train_end) & (origin < split.val_end) & (ends < split.val_end)
    test = origin >= split.val_end
    test &= ends.notna()
    if split.test_end is not None:
        test &= (origin < split.test_end) & (ends < split.test_end)
    return {"train": train.to_numpy(bool), "val": val.to_numpy(bool), "test": test.to_numpy(bool)}


def _garch_hday_logvol(returns_percent: np.ndarray, horizon: int, refit: int = 25, min_train: int = 500) -> np.ndarray:
    from arch import arch_model

    r = np.asarray(returns_percent, float)
    out = np.full(len(r), np.nan)
    mu = omega = alpha = beta = long_var = conditional_var = None
    last_refit = -10**9
    for t in range(len(r)):
        if t >= min_train and t - last_refit >= refit:
            result = arch_model(
                pd.Series(r[: t + 1]), mean="Constant", vol="Garch", p=1, q=1, dist="normal"
            ).fit(disp="off")
            params = result.params
            mu = float(params["mu"])
            omega = float(params["omega"])
            alpha = float(params["alpha[1]"])
            beta = float(params["beta[1]"])
            persistence = alpha + beta
            long_var = omega / (1 - persistence) if 0 < persistence < 1 else float(np.var(r[: t + 1]))
            conditional_var = float(np.asarray(result.conditional_volatility)[-1] ** 2)
            last_refit = t
        if conditional_var is None:
            continue
        next_var = omega + alpha * (r[t] - mu) ** 2 + beta * conditional_var
        persistence = alpha + beta
        expected = next_var
        total = 0.0
        for _ in range(horizon):
            total += expected
            expected = long_var + persistence * (expected - long_var)
        out[t] = 0.5 * np.log(total) - np.log(100.0)
        conditional_var = next_var
    return out


_GARCH_CACHE: dict[tuple[str, int], pd.Series] = {}


def garch_series(path: Path, horizon: int) -> pd.Series:
    key = (str(path.resolve()), int(horizon))
    if key not in _GARCH_CACHE:
        df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
        returns = (np.log(df["Close"].astype(float)).diff() * 100.0).to_numpy()
        valid = np.isfinite(returns)
        values = np.full(len(returns), np.nan)
        values[valid] = _garch_hday_logvol(returns[valid], horizon)
        _GARCH_CACHE[key] = pd.Series(values, index=df.index)
    return _GARCH_CACHE[key]


def _corrector_features(df: pd.DataFrame, iv_column: str, mode: str) -> pd.DataFrame:
    drop = {"Close", "target_logrv", "target_rv"}
    features = df.drop(columns=[c for c in drop if c in df.columns]).copy()
    if mode == "none":
        features = features.drop(columns=[c for c in IV_FEATURES if c in features], errors="ignore")
    elif mode == "selected":
        other = IV_FEATURES - {iv_column, f"{iv_column}_chg"}
        features = features.drop(columns=[c for c in other if c in features], errors="ignore")
    elif mode != "all":
        raise ValueError(f"unknown corrector IV mode: {mode}")
    return features.astype("float32")


def _core_frame(df: pd.DataFrame, path: Path, horizon: int, iv_column: str) -> pd.DataFrame:
    if iv_column not in df:
        raise ValueError(f"{iv_column} is absent from {path}; rebuild the dataset")
    close = df["Close"].astype(float)
    variance = np.log(close).diff().pow(2)
    eps = 1e-12
    return pd.DataFrame(
        {
            "hd": np.log(np.sqrt(variance.rolling(1).sum()) + eps),
            "hw": np.log(np.sqrt(variance.rolling(5).sum()) + eps),
            "hm": np.log(np.sqrt(variance.rolling(22).sum()) + eps),
            "iv": np.log(df[iv_column].astype(float) + eps),
            "garch": garch_series(path, horizon).reindex(df.index),
        },
        index=df.index,
    )


def _base_frame(
    cfg: ExperimentConfig,
    index_name: str,
    horizon: int,
    iv_column: str,
    corrector_iv_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = Path(cfg.data_dir) / f"{index_name}_vol.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run python data/build_vol_dataset.py first")
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    features = _corrector_features(df, iv_column, corrector_iv_mode)
    core = _core_frame(df, path, horizon, iv_column)
    y = rebuild_target(df["Close"], horizon)
    end_date = target_end_dates(df.index, horizon)
    complete = features.notna().all(axis=1) & core.notna().all(axis=1) & y.notna() & end_date.notna()
    # Matched and VIX-reference anchor swaps must use exactly the same sample,
    # including their training history, so require both columns when present.
    alignment_columns = [name for name in ("ANCHOR", "VIX_REFERENCE") if name in df]
    if alignment_columns:
        complete &= df[alignment_columns].notna().all(axis=1)
    frame = pd.concat(
        [
            y.rename("target"),
            end_date.rename("target_end"),
            core.add_prefix("core:"),
        ],
        axis=1,
    ).loc[complete]
    return frame, features.loc[complete], core.loc[complete]


def derive_fixed_split_dates(
    cfg: ExperimentConfig,
    index_name: str,
    reference_horizon: int | None = None,
    iv_column: str = "VIX",
) -> SplitDates:
    reference_horizon = reference_horizon or max(cfg.horizons)
    frame, _, _ = _base_frame(cfg, index_name, reference_horizon, iv_column, "none")
    dates = frame.index
    n = len(dates)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)
    if min(n_train, n_val, n - n_train - n_val) <= cfg.seq_len:
        raise ValueError(f"not enough eligible rows for {index_name}")
    return SplitDates(train_end=dates[n_train], val_end=dates[n_train + n_val])


def derive_rolling_splits(
    cfg: ExperimentConfig,
    index_name: str,
    horizon: int,
    iv_column: str = "VIX",
) -> list[SplitDates]:
    frame, _, _ = _base_frame(cfg, index_name, horizon, iv_column, "none")
    dates = frame.index
    n = len(dates)
    train_n = int(n * cfg.rolling_initial_train_ratio)
    val_n = int(n * cfg.rolling_val_ratio)
    test_n = int(n * cfg.rolling_test_ratio)
    folds = []
    for fold in range(cfg.rolling_folds):
        train_end_i = train_n + fold * test_n
        val_end_i = train_end_i + val_n
        test_end_i = min(val_end_i + test_n, n)
        if test_end_i <= val_end_i or test_end_i >= n + 1:
            break
        test_end = None if test_end_i == n else dates[test_end_i]
        folds.append(SplitDates(dates[train_end_i], dates[val_end_i], test_end))
    if not folds:
        raise ValueError("rolling split configuration produced no valid folds")
    return folds


def prepare_data(
    cfg: ExperimentConfig,
    index_name: str,
    horizon: int,
    split_dates: SplitDates,
    core_set: str = "full",
    iv_column: str = "VIX",
    corrector_iv_mode: str | None = None,
) -> DataBundle:
    if core_set not in CORE_SETS:
        raise ValueError(f"unknown core_set={core_set}")
    corrector_iv_mode = corrector_iv_mode or cfg.corrector_iv_mode
    frame, features, core_all = _base_frame(cfg, index_name, horizon, iv_column, corrector_iv_mode)
    core = core_all[CORE_SETS[core_set]]
    masks = make_split_masks(frame.index, frame["target_end"], split_dates)
    if min(int(m.sum()) for m in masks.values()) < 2:
        raise ValueError(f"empty split for {index_name}, h={horizon}, dates={split_dates}")

    # Every fitted artifact uses only labels whose complete target window stays in train.
    train_mask = masks["train"]
    feature_scaler = Standardizer().fit(features.to_numpy()[train_mask])
    core_scaler = Standardizer().fit(core.to_numpy()[train_mask])
    target_scaler = Standardizer1D().fit(frame["target"].to_numpy()[train_mask])
    x_scaled = feature_scaler.transform(features.to_numpy()).astype("float32")
    c_scaled = core_scaler.transform(core.to_numpy()).astype("float32")
    y_scaled = target_scaler.transform(frame["target"].to_numpy()).astype("float32")

    design = np.c_[np.ones(int(train_mask.sum())), c_scaled[train_mask]]
    coef, *_ = np.linalg.lstsq(design, y_scaled[train_mask], rcond=None)
    core_coef = (coef[1:].astype("float32"), float(coef[0]))
    core_standardized = c_scaled @ core_coef[0] + core_coef[1]
    core_logrv_all = target_scaler.inverse(core_standardized)

    seq_x: list[np.ndarray] = []
    seq_c: list[np.ndarray] = []
    seq_y: list[list[float]] = []
    label_rows: list[int] = []
    for start in range(len(frame) - cfg.seq_len + 1):
        label = start + cfg.seq_len - 1
        seq_x.append(x_scaled[start : start + cfg.seq_len])
        seq_c.append(c_scaled[label])
        seq_y.append([y_scaled[label]])
        label_rows.append(label)
    xw = np.asarray(seq_x, dtype="float32")
    cw = np.asarray(seq_c, dtype="float32")
    yw = np.asarray(seq_y, dtype="float32")[:, :, None]
    labels = np.asarray(label_rows)

    datasets: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    dates: dict[str, np.ndarray] = {}
    truths: dict[str, np.ndarray] = {}
    core_predictions: dict[str, np.ndarray] = {}
    last_features: dict[str, np.ndarray] = {}
    for name, mask in masks.items():
        keep = mask[labels]
        rows = labels[keep]
        datasets[name] = (
            torch.from_numpy(xw[keep]),
            torch.from_numpy(cw[keep]),
            torch.from_numpy(yw[keep]),
        )
        dates[name] = frame.index.to_numpy()[rows]
        truths[name] = frame["target"].to_numpy(float)[rows]
        core_predictions[name] = np.asarray(core_logrv_all)[rows]
        last_features[name] = x_scaled[rows]

    # Hard safety checks: fitted labels and split labels never cross the next boundary.
    assert np.all(frame.loc[train_mask, "target_end"] < split_dates.train_end)
    assert np.all(frame.loc[masks["val"], "target_end"] < split_dates.val_end)
    for name in ("train", "val", "test"):
        if len(datasets[name][0]) == 0:
            raise ValueError(f"sequence construction emptied {name} for {index_name}")

    return DataBundle(
        datasets=datasets,
        dates=dates,
        true_logrv=truths,
        core_logrv=core_predictions,
        last_features=last_features,
        feature_names=list(features.columns),
        core_names=list(core.columns),
        n_features=features.shape[1],
        n_core=core.shape[1],
        target_scaler=target_scaler,
        core_coef=core_coef,
        split_dates=split_dates,
        frame=frame,
        split_masks=masks,
        index_name=index_name,
        horizon=horizon,
        iv_column=iv_column,
        corrector_iv_mode=corrector_iv_mode,
        data_path=Path(cfg.data_dir) / f"{index_name}_vol.csv",
    )


def baseline_predictions(bundle: DataBundle, split_name: str = "test") -> dict[str, np.ndarray]:
    """Causal baselines fitted on the exact purged train mask and aligned to model dates."""
    frame = bundle.frame
    train = bundle.split_masks["train"]
    split_dates = pd.to_datetime(bundle.dates[split_name])
    target = frame["target"].to_numpy(float)
    core_columns = [f"core:{name}" for name in bundle.core_names]

    def ols(columns: list[str]) -> np.ndarray:
        x_train = frame.loc[train, columns].to_numpy(float)
        design = np.c_[np.ones(len(x_train)), x_train]
        coef, *_ = np.linalg.lstsq(design, target[train], rcond=None)
        x_test = frame.loc[split_dates, columns].to_numpy(float)
        return np.c_[np.ones(len(x_test)), x_test] @ coef

    out = {
        f"core_ols_{'_'.join(bundle.core_names)}": ols(core_columns),
        "har": ols(["core:hd", "core:hw", "core:hm"]),
        "garch": frame.loc[split_dates, "core:garch"].to_numpy(float),
        f"{bundle.iv_column.lower()}_only": ols(["core:iv"]),
    }
    # Persistence is trailing h-day realized volatility, which equals the HAR h-window only for h=5.
    # Reconstruct it from Close so it is matched to each requested horizon.
    data_path = bundle.data_path
    if data_path.exists():
        source = pd.read_csv(data_path, index_col=0, parse_dates=True).sort_index()
        variance = np.log(source["Close"].astype(float)).diff().pow(2)
        persistence = np.log(np.sqrt(variance.rolling(bundle.horizon).sum()) + 1e-12)
        out["persistence"] = persistence.reindex(split_dates).to_numpy(float)
        ewma_var = variance.ewm(alpha=0.06, adjust=False).mean()
        out["ewma"] = (0.5 * np.log(ewma_var * bundle.horizon + 1e-12)).reindex(split_dates).to_numpy(float)
    return out


def split_summary(bundle: DataBundle) -> dict[str, Any]:
    # Verify paired numeric inputs as well as column names and dates.
    import hashlib

    input_hashes = {}
    for split, tensors in bundle.datasets.items():
        digest = hashlib.sha256()
        for tensor in tensors:
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(str((array.shape, array.dtype.str)).encode("utf-8"))
            digest.update(memoryview(array).cast("B"))
        input_hashes[split] = digest.hexdigest()
    return {
        "index": bundle.index_name,
        "horizon": bundle.horizon,
        "iv_column": bundle.iv_column,
        "corrector_iv_mode": bundle.corrector_iv_mode,
        "features": bundle.feature_names,
        "input_sha256": input_hashes,
        "target_standardizer": {"mean": bundle.target_scaler.mean, "std": bundle.target_scaler.std},
        "core": bundle.core_names,
        "split_dates": bundle.split_dates.as_dict(),
        "sizes": {name: int(len(values[0])) for name, values in bundle.datasets.items()},
        "date_ranges": {
            name: [str(pd.Timestamp(dates[0]).date()), str(pd.Timestamp(dates[-1]).date())]
            for name, dates in bundle.dates.items()
        },
    }
