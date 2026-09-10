from __future__ import annotations

import copy
import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import trange

from .config import ExperimentConfig
from .artifacts import file_sha256, write_json_atomic, write_npz_atomic
from .data import DataBundle
from .models import AnchoredCorrector
from .statistics import daily_qlike


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def volatility_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    error = pred - true
    denominator = float(np.square(true - true.mean()).sum())
    corr = float(np.corrcoef(pred, true)[0, 1]) if pred.std() > 0 and true.std() > 0 else float("nan")
    return {
        "RMSE": float(np.sqrt(np.square(error).mean())),
        "MAE": float(np.abs(error).mean()),
        "QLIKE": float(daily_qlike(pred, true).mean()),
        "R2": float(1.0 - np.square(error).sum() / denominator) if denominator > 0 else float("nan"),
        "Corr": corr,
    }


def qlike_loss_standardized(pred: torch.Tensor, true: torch.Tensor, target_std: float) -> torch.Tensor:
    ratio_log = torch.clamp(2.0 * target_std * (true - pred), -10.0, 10.0)
    return (torch.exp(ratio_log) - ratio_log - 1.0).mean()


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    bundle: DataBundle,
    device: str,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    predictions, targets = [], []
    for x, core, y in loader:
        predictions.append(model(x.to(device), core.to(device)).cpu().reshape(-1).numpy())
        targets.append(y.reshape(-1).numpy())
    pred = bundle.target_scaler.inverse(np.concatenate(predictions))
    true = bundle.target_scaler.inverse(np.concatenate(targets))
    if not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise FloatingPointError("non-finite predictions/targets; this run cannot be reported")
    return volatility_metrics(pred, true), pred, true


@dataclass
class SeedResult:
    seed: int
    initial_trainable_hash: str
    best_epoch: int
    initial_validation: dict[str, float]
    validation: dict[str, float]
    test: dict[str, float]
    validation_prediction: np.ndarray
    test_prediction: np.ndarray
    validation_true: np.ndarray
    test_true: np.ndarray
    history: list[dict[str, float]]
    state_dict: dict[str, torch.Tensor]


@dataclass
class EnsembleResult:
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    validation_prediction: np.ndarray
    test_prediction: np.ndarray
    validation_true: np.ndarray
    test_true: np.ndarray
    seed_metrics_mean: dict[str, float]
    seed_metrics_std: dict[str, float]
    guard_applied: bool
    seed_results: list[SeedResult]


def train_seed(
    cfg: ExperimentConfig,
    bundle: DataBundle,
    seed: int,
    corrector: str = "bimamba",
    freeze_core: bool = True,
    anchor: bool = True,
    use_core: bool = True,
    n_scans: int | None = None,
    zero_init_readout: bool | None = None,
) -> SeedResult:
    set_seed(seed)
    device = resolve_device(cfg.device)
    loaders = bundle.loaders(cfg.batch_size)
    model = AnchoredCorrector(
        cfg,
        bundle.n_features,
        bundle.n_core,
        bundle.core_coef,
        corrector=corrector,
        freeze_core=freeze_core,
        anchor=anchor,
        use_core=use_core,
        n_scans=n_scans,
        zero_init_readout=zero_init_readout,
    ).to(device)
    initial_hash = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            initial_hash.update(name.encode("utf-8"))
            initial_hash.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    initial_trainable_hash = initial_hash.hexdigest()

    # Epoch zero is a real candidate under exactly the same selection rule for
    # direct and residual modes.  Only its conditional path differs.
    initial_val, initial_val_pred, initial_val_true = evaluate_model(model, loaders["val"], bundle, device)
    zero_initialized = bool(anchor if zero_init_readout is None else zero_init_readout)
    if zero_initialized and use_core:
        if not np.allclose(initial_val_pred, bundle.core_logrv["val"], atol=2e-5, rtol=2e-5):
            raise AssertionError("epoch-zero anchored prediction is not equal to the frozen OLS core")
    if zero_initialized and not use_core:
        expected = np.full_like(initial_val_pred, bundle.target_scaler.mean, dtype=float)
        if not np.allclose(initial_val_pred, expected, atol=2e-5, rtol=2e-5):
            raise AssertionError("epoch-zero direct prediction is not equal to the train target mean")
    best_value = float(initial_val[cfg.selection_metric])
    if not np.isfinite(best_value):
        raise FloatingPointError("non-finite epoch-zero selection metric")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience_count = 0
    history = [{"epoch": 0, **initial_val}]

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    mse = nn.MSELoss()
    epoch_iterator = trange(
        1,
        cfg.epochs + 1,
        desc=f"{bundle.index_name}:{corrector}:seed{seed}",
        unit="epoch",
        leave=False,
    )
    for epoch in epoch_iterator:
        model.train()
        for x, core, y in loaders["train"]:
            x, core, y = x.to(device), core.to(device), y.to(device)
            optimizer.zero_grad()
            prediction = model(x, core)
            if cfg.train_loss == "qlike":
                loss = qlike_loss_standardized(prediction, y, bundle.target_scaler.std)
            else:
                loss = mse(prediction, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at seed {seed}, epoch {epoch}")
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(parameters, cfg.grad_clip)
            optimizer.step()
        validation, _, _ = evaluate_model(model, loaders["val"], bundle, device)
        history.append({"epoch": epoch, **validation})
        value = float(validation[cfg.selection_metric])
        if not np.isfinite(value):
            raise FloatingPointError(f"non-finite validation metric at seed {seed}, epoch {epoch}")
        epoch_iterator.set_postfix(best=f"{best_value:.4g}", val=f"{value:.4g}")
        if value < best_value:
            best_value = value
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_count = 0
        else:
            patience_count += 1
            if cfg.patience and patience_count >= cfg.patience:
                break

    model.load_state_dict(best_state)
    validation, val_pred, val_true = evaluate_model(model, loaders["val"], bundle, device)
    test, test_pred, test_true = evaluate_model(model, loaders["test"], bundle, device)
    # Both modes use the same checkpoint candidate set, including epoch zero.
    if validation[cfg.selection_metric] > initial_val[cfg.selection_metric] + 1e-8:
        raise AssertionError("checkpoint selection violated the shared epoch-zero rule")
    return SeedResult(
        seed=seed,
        initial_trainable_hash=initial_trainable_hash,
        best_epoch=best_epoch,
        initial_validation=initial_val,
        validation=validation,
        test=test,
        validation_prediction=val_pred,
        test_prediction=test_pred,
        validation_true=val_true,
        test_true=test_true,
        history=history,
        state_dict={name: value.detach().cpu() for name, value in best_state.items()},
    )


def _save_seed(directory: Path, signature: str | None, bundle: DataBundle, result: SeedResult) -> None:
    """Commit a completed seed immediately; JSON is the final commit marker."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"seed_{result.seed}"
    weights = directory / f"{stem}.pt"
    temporary = weights.with_suffix(".pt.tmp")
    torch.save(result.state_dict, temporary)
    temporary.replace(weights)
    predictions = directory / f"{stem}_predictions.npz"
    write_npz_atomic(
        predictions,
        validation_dates=np.asarray(bundle.dates["val"], dtype="datetime64[ns]"),
        test_dates=np.asarray(bundle.dates["test"], dtype="datetime64[ns]"),
        validation_prediction=result.validation_prediction,
        test_prediction=result.test_prediction,
        validation_true=result.validation_true,
        test_true=result.test_true,
    )
    csv_path = directory / f"{stem}_predictions.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed", "split", "date", "true_logrv", "prediction_logrv"])
        for split, prefix in (("val", "validation"), ("test", "test")):
            for date, true, pred in zip(
                bundle.dates[split], getattr(result, f"{prefix}_true"),
                getattr(result, f"{prefix}_prediction"),
            ):
                writer.writerow([result.seed, split, str(date), float(true), float(pred)])
    temporary.replace(csv_path)
    write_json_atomic(directory / f"{stem}_history.json", result.history)
    write_json_atomic(directory / f"{stem}_result.json", {
        "signature": signature,
        "seed": result.seed,
        "initial_trainable_hash": result.initial_trainable_hash,
        "best_epoch": result.best_epoch,
        "initial_validation": result.initial_validation,
        "validation": result.validation,
        "test": result.test,
        "history": result.history,
        "weights_sha256": file_sha256(weights),
        "predictions_sha256": file_sha256(predictions),
    })


def _load_seed(directory: Path, signature: str, bundle: DataBundle, seed: int) -> SeedResult | None:
    stem = f"seed_{seed}"
    metadata = directory / f"{stem}_result.json"
    if not metadata.exists():
        return None
    try:
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        if payload.get("signature") != signature or payload.get("seed") != seed:
            return None
        weights = directory / f"{stem}.pt"
        predictions = directory / f"{stem}_predictions.npz"
        if (file_sha256(weights) != payload["weights_sha256"]
                or file_sha256(predictions) != payload["predictions_sha256"]):
            raise ValueError("completed-seed artifact checksum mismatch")
        with np.load(predictions, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        for split, prefix in (("val", "validation"), ("test", "test")):
            if not np.array_equal(arrays[f"{prefix}_dates"], bundle.dates[split]):
                raise ValueError("completed-seed dates differ from current data")
            if not np.isfinite(arrays[f"{prefix}_prediction"]).all():
                raise ValueError("non-finite cached predictions")
        return SeedResult(
            seed=seed,
            initial_trainable_hash=payload["initial_trainable_hash"],
            best_epoch=payload["best_epoch"],
            initial_validation=payload["initial_validation"],
            validation=payload["validation"], test=payload["test"],
            history=payload["history"],
            # Aggregation needs predictions, not deserialization of the weights.
            # The intact checkpoint remains on disk and is verified above.
            state_dict={},
            **{key: arrays[key] for key in (
                "validation_prediction", "test_prediction", "validation_true", "test_true"
            )},
        )
    except (OSError, ValueError, KeyError, EOFError) as exc:
        print(f"[cache invalid] {stem}: {exc}; retraining this seed")
        return None


def run_ensemble(
    cfg: ExperimentConfig,
    bundle: DataBundle,
    corrector: str = "bimamba",
    freeze_core: bool = True,
    anchor: bool = True,
    use_core: bool = True,
    n_scans: int | None = None,
    zero_init_readout: bool | None = None,
    artifact_dir: Path | None = None,
    cache_signature: str | None = None,
    use_guard: bool | None = None,
) -> EnsembleResult:
    if not cfg.seeds or len(set(cfg.seeds)) != len(cfg.seeds):
        raise ValueError("seeds must be a nonempty list of distinct integers")
    seed_results = []
    for seed in cfg.seeds:
        result = None
        if artifact_dir is not None and cache_signature is not None and not cfg.force:
            result = _load_seed(artifact_dir, cache_signature, bundle, seed)
        if result is None:
            result = train_seed(
                cfg, bundle, seed, corrector=corrector, freeze_core=freeze_core,
                anchor=anchor, use_core=use_core, n_scans=n_scans,
                zero_init_readout=zero_init_readout,
            )
            if artifact_dir is not None:
                _save_seed(artifact_dir, cache_signature, bundle, result)
        else:
            print(f"[cached seed] {bundle.index_name}:{corrector}:seed{seed}")
        seed_results.append(result)
    val_true = seed_results[0].validation_true
    test_true = seed_results[0].test_true
    for result in seed_results[1:]:
        if not np.array_equal(result.validation_true, val_true) or not np.array_equal(result.test_true, test_true):
            raise AssertionError("seed runs produced different targets")
    val_prediction = np.mean([result.validation_prediction for result in seed_results], axis=0)
    test_prediction = np.mean([result.test_prediction for result in seed_results], axis=0)

    guard_enabled = cfg.ensemble_guard if use_guard is None else use_guard
    core_val = bundle.core_logrv["val"]
    core_test = bundle.core_logrv["test"]
    guard_applied = bool(
        guard_enabled
        and use_core
        and volatility_metrics(val_prediction, val_true)[cfg.selection_metric]
        > volatility_metrics(core_val, val_true)[cfg.selection_metric]
    )
    if guard_applied:
        val_prediction = core_val.copy()
        test_prediction = core_test.copy()

    keys = seed_results[0].test
    seed_mean = {key: float(np.mean([run.test[key] for run in seed_results])) for key in keys}
    seed_std = {
        key: float(np.std([run.test[key] for run in seed_results], ddof=1))
        if len(seed_results) > 1
        else float("nan")
        for key in keys
    }
    result = EnsembleResult(
        validation_metrics=volatility_metrics(val_prediction, val_true),
        test_metrics=volatility_metrics(test_prediction, test_true),
        validation_prediction=val_prediction,
        test_prediction=test_prediction,
        validation_true=val_true,
        test_true=test_true,
        seed_metrics_mean=seed_mean,
        seed_metrics_std=seed_std,
        guard_applied=guard_applied,
        seed_results=seed_results,
    )
    return result
