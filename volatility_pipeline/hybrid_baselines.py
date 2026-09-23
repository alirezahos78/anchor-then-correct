"""Loss-level and input-level hybrid baselines, each kept as its own method.

``ginn``   GINN (Xu et al., ICAIF 2024).  The GARCH forecast is a second
           pseudo-target in the training objective.  It never enters the
           forward pass.  Original loss:

               L = lam * MSE(sigma2_true, sigma2_hat)
                 + (1 - lam) * MSE(sigma2_GARCH, sigma2_hat)

           GINN applies one loss function to both terms.  Under the host
           protocol that loss is QLIKE, so both terms become QLIKE.  Nothing
           else about the mechanism is altered.

``input``  Kim & Won (ESWA 2018).  GARCH-type variance forecasts are supplied
           as additional encoder channels.  The network emits a direct
           prediction.

Neither baseline uses the fitted OLS fusion, the additive output path, the
zero-initialised readout, or the epoch-zero checkpoint candidate.  Those belong
to the anchored construction under test and are not imported here.

What is shared with the host protocol, and only this: horizon, QLIKE
objective, 60-day window, the 38-channel encoder input, parameter budget,
Adam settings, epoch budget, patience, gradient clipping, seeds, and
seed-averaged reporting.  Report the numbers as reimplementations under a
common protocol; they will not match the originals.

"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import trange

from .config import ExperimentConfig
from .artifacts import write_json_atomic, write_npz_atomic
from .data import DataBundle
from .models import AnchoredCorrector, build_corrector
from .training import (
    evaluate_model,
    qlike_loss_standardized,
    resolve_device,
    set_seed,
    volatility_metrics,
)

BASELINE_KINDS = ("ginn", "input")
GARCH_CHANNEL = "garch"


# ---------------------------------------------------------------------------
# GINN's regularisation target
# ---------------------------------------------------------------------------


class GarchReference(nn.Module):
    """Emits the GARCH forecast in standardised target units.

    The encoder receives the core channels already standardised by the fitted
    core scaler, so recovering G_{t,h} needs the train channel statistics:

        physical    = c_tilde * core_std + core_mean
        standardised target = (physical - target_mean) / target_std

    Both affine maps are folded into one frozen linear layer.  ``reference``
    selects which structured estimate plays GINN's role.  ``"garch"`` is the
    method as published and is the default.  ``"core"`` substitutes the fitted
    OLS fusion; it is a deviation from GINN, useful only as a sensitivity check
    on how much the strength of the regularisation target matters.
    """

    def __init__(
        self,
        bundle: DataBundle,
        reference: str = "garch",
    ):
        super().__init__()
        if reference not in ("garch", "core"):
            raise ValueError(f"unknown reference={reference}")
        self.reference = reference
        self.linear = nn.Linear(bundle.n_core, 1)
        target_mean = float(bundle.target_scaler.mean)
        target_std = float(bundle.target_scaler.std)
        with torch.no_grad():
            if reference == "core":
                weights, bias = bundle.core_coef
                self.linear.weight.copy_(
                    torch.as_tensor(weights, dtype=torch.float32).reshape(1, -1)
                )
                self.linear.bias.fill_(float(bias))
            else:
                index = garch_channel_index(bundle)
                scaler = bundle.core_scaler
                if scaler is None:
                    raise RuntimeError(
                        "bundle.core_scaler is unset; prepare the bundle with volatility_pipeline.data.prepare_data"
                    )
                channel_mean = float(np.asarray(scaler.mean).reshape(-1)[index])
                channel_std = float(np.asarray(scaler.std).reshape(-1)[index])
                nn.init.zeros_(self.linear.weight)
                self.linear.weight[0, index] = channel_std / target_std
                self.linear.bias.fill_((channel_mean - target_mean) / target_std)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, core: torch.Tensor) -> torch.Tensor:
        # Matches AnchoredCorrector's output shape [B, 1, 1].
        return self.linear(core).unsqueeze(-1)


def garch_channel_index(bundle: DataBundle) -> int:
    names = [str(name).lower() for name in bundle.core_names]
    if GARCH_CHANNEL not in names:
        raise RuntimeError(
            f"no '{GARCH_CHANNEL}' channel in core_names={bundle.core_names}; "
            "the loss-level baseline needs core_set='full'"
        )
    return names.index(GARCH_CHANNEL)


def verify_reference(bundle: DataBundle, reference: str = "garch") -> dict[str, float]:
    """Sanity-check the recovered series before spending a training budget.

    For ``reference='core'`` the emitted values must reproduce
    ``bundle.core_logrv`` exactly, which validates the whole affine path.
    For ``reference='garch'`` the reconstructed physical series is returned for
    inspection; log-volatility values well outside roughly [-6, 0] indicate a
    channel or scaler mismatch.
    """
    head = GarchReference(bundle, reference=reference)
    _, core_tensor, _ = bundle.datasets["test"]
    with torch.no_grad():
        standardised = head(core_tensor).reshape(-1).numpy()
    physical = bundle.target_scaler.inverse(standardised)
    report = {
        "mean_logvol": float(physical.mean()),
        "std_logvol": float(physical.std()),
        "min_logvol": float(physical.min()),
        "max_logvol": float(physical.max()),
    }
    if reference == "core":
        stored = bundle.core_logrv["test"]
        report["max_abs_dev_from_core_logrv"] = float(
            np.abs(physical - stored).max()
        )
        if not np.allclose(physical, stored, atol=1e-4, rtol=1e-4):
            raise AssertionError(
                "reference head does not reproduce the stored core forecast"
            )
    return report


# ---------------------------------------------------------------------------
# Input-level hybrid
# ---------------------------------------------------------------------------


class InputHybridCorrector(nn.Module):
    """Structured variance estimates supplied as additional encoder channels.

    ``append='garch'`` is the Kim & Won mechanism with the estimate available
    here.  ``append='core'`` instead appends all five core channels; that is
    not a baseline but an ablation of the anchored model, equalising the
    information set between the two paths.  Keep the two purposes separate
    when reporting.

    The structured estimate is constant within a forecast window and is
    broadcast across the observed positions as a static covariate.  The readout
    stays randomly initialised.
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        bundle: DataBundle,
        corrector: str = "bimamba",
        n_scans: int | None = None,
        append: str = "garch",
    ):
        super().__init__()
        if append not in ("garch", "core"):
            raise ValueError(f"unknown append={append}")
        self.append = append
        self.garch_index = garch_channel_index(bundle) if append == "garch" else None
        n_append = 1 if append == "garch" else bundle.n_core
        total = bundle.n_features + n_append
        self.total_features = total
        self.backbone = build_corrector(cfg, corrector, total, n_scans=n_scans)
        self.readout = nn.Linear(total, 1)

    def forward(self, x: torch.Tensor, core: torch.Tensor) -> torch.Tensor:
        if self.append == "garch":
            core = core[:, self.garch_index : self.garch_index + 1]
        broadcast = core.unsqueeze(1).expand(-1, x.shape[1], -1)
        sequence = self.backbone(torch.cat([x, broadcast], dim=-1))
        return self.readout(sequence.mean(1)).unsqueeze(-1)


def build_baseline_model(
    kind: str,
    cfg: ExperimentConfig,
    bundle: DataBundle,
    corrector: str = "bimamba",
    n_scans: int | None = None,
    append: str = "garch",
) -> nn.Module:
    if kind == "ginn":
        # Direct predictor: the core never enters the forward pass, and the
        # readout keeps its own random initialisation.
        return AnchoredCorrector(
            cfg, bundle.n_features, bundle.n_core, bundle.core_coef,
            corrector=corrector, freeze_core=True, anchor=False,
            use_core=False, n_scans=n_scans, zero_init_readout=False,
        )
    if kind == "input":
        return InputHybridCorrector(
            cfg, bundle, corrector=corrector, n_scans=n_scans, append=append
        )
    raise ValueError(f"unknown baseline kind={kind}")


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def ginn_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    reference: torch.Tensor,
    target_std: float,
    lam: float,
) -> torch.Tensor:
    """lam * QLIKE(y, yhat) + (1 - lam) * QLIKE(reference, yhat).

    lam = 1 recovers plain direct training; lam = 0 is GINN-0, which learns
    only to reproduce the structured forecast.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must lie in [0, 1], got {lam}")
    supervised = qlike_loss_standardized(prediction, target, target_std)
    if lam >= 1.0:
        return supervised
    regularised = qlike_loss_standardized(prediction, reference, target_std)
    if lam <= 0.0:
        return regularised
    return lam * supervised + (1.0 - lam) * regularised


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class BaselineSeedResult:
    seed: int
    kind: str
    lam: float
    initial_trainable_hash: str
    trainable_parameters: int
    best_epoch: int
    validation: dict[str, float]
    test: dict[str, float]
    validation_prediction: np.ndarray
    test_prediction: np.ndarray
    validation_true: np.ndarray
    test_true: np.ndarray
    history: list[dict[str, float]]
    state_dict: dict[str, torch.Tensor]


def train_baseline_seed(
    cfg: ExperimentConfig,
    bundle: DataBundle,
    seed: int,
    kind: str,
    corrector: str = "bimamba",
    lam: float = 0.5,
    n_scans: int | None = None,
    reference: str = "garch",
    append: str = "garch",
    progress: bool = True,
) -> BaselineSeedResult:
    if kind not in BASELINE_KINDS:
        raise ValueError(f"unknown baseline kind={kind}")
    set_seed(seed)
    device = resolve_device(cfg.device)
    loaders = bundle.loaders(cfg.batch_size)

    model = build_baseline_model(
        kind, cfg, bundle, corrector=corrector, n_scans=n_scans, append=append
    ).to(device)
    reference_head = (
        GarchReference(bundle, reference=reference).to(device)
        if kind == "ginn" and lam < 1.0
        else None
    )

    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode("utf-8"))
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    initial_trainable_hash = digest.hexdigest()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        parameters, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    # No epoch-zero candidate: an untrained readout is not a meaningful
    # forecast for either baseline, so selection starts after epoch one.
    best_value = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    patience_count = 0
    history: list[dict[str, float]] = []

    iterator = trange(
        1, cfg.epochs + 1,
        desc=f"{bundle.index_name}:{kind}:{corrector}:seed{seed}",
        unit="epoch", leave=False, disable=not progress,
    )
    for epoch in iterator:
        model.train()
        for x, core, y in loaders["train"]:
            x, core, y = x.to(device), core.to(device), y.to(device)
            optimizer.zero_grad()
            prediction = model(x, core)
            if reference_head is not None:
                loss = ginn_loss(
                    prediction, y, reference_head(core),
                    bundle.target_scaler.std, lam,
                )
            else:
                loss = qlike_loss_standardized(
                    prediction, y, bundle.target_scaler.std
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite training loss at seed {seed}, epoch {epoch}"
                )
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(parameters, cfg.grad_clip)
            optimizer.step()

        # Selection uses plain validation QLIKE, never the blended objective,
        # so the number stays comparable with every other column in Table 1.
        validation, _, _ = evaluate_model(model, loaders["val"], bundle, device)
        history.append({"epoch": epoch, **validation})
        value = float(validation[cfg.selection_metric])
        if not np.isfinite(value):
            raise FloatingPointError(
                f"non-finite validation metric at seed {seed}, epoch {epoch}"
            )
        iterator.set_postfix(best=f"{best_value:.4g}", val=f"{value:.4g}")
        if value < best_value:
            best_value, best_epoch = value, epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if cfg.patience and patience_count >= cfg.patience:
                break

    if best_state is None:
        raise RuntimeError("no epoch completed; cannot select a checkpoint")
    model.load_state_dict(best_state)
    validation, val_pred, val_true = evaluate_model(model, loaders["val"], bundle, device)
    test, test_pred, test_true = evaluate_model(model, loaders["test"], bundle, device)
    return BaselineSeedResult(
        seed=seed, kind=kind, lam=float(lam),
        initial_trainable_hash=initial_trainable_hash,
        trainable_parameters=int(trainable), best_epoch=best_epoch,
        validation=validation, test=test,
        validation_prediction=val_pred, test_prediction=test_pred,
        validation_true=val_true, test_true=test_true,
        history=history,
        state_dict={k: v.detach().cpu() for k, v in best_state.items()},
    )


def run_baseline_ensemble(
    cfg: ExperimentConfig,
    bundle: DataBundle,
    kind: str,
    corrector: str = "bimamba",
    lam: float = 0.5,
    n_scans: int | None = None,
    reference: str = "garch",
    append: str = "garch",
    artifact_dir: Path | None = None,
    progress: bool = True,
) -> dict:
    """Seed-average the log-volatility predictions, as Table 1 does.

    The ensemble core-fallback guard belongs to the anchored design and is not
    applied here.
    """
    seed_results = [
        train_baseline_seed(
            cfg, bundle, seed, kind, corrector=corrector, lam=lam,
            n_scans=n_scans, reference=reference, append=append,
            progress=progress,
        )
        for seed in cfg.seeds
    ]
    val_true = seed_results[0].validation_true
    test_true = seed_results[0].test_true
    for result in seed_results[1:]:
        if not np.array_equal(result.test_true, test_true):
            raise AssertionError("seed runs produced different targets")

    val_prediction = np.mean([r.validation_prediction for r in seed_results], axis=0)
    test_prediction = np.mean([r.test_prediction for r in seed_results], axis=0)
    keys = seed_results[0].test
    payload = {
        "market": bundle.index_name,
        "kind": kind,
        "backbone": corrector,
        "lam": float(lam) if kind == "ginn" else None,
        "reference": reference if kind == "ginn" else None,
        "append": append if kind == "input" else None,
        "trainable_parameters": seed_results[0].trainable_parameters,
        "ensemble_validation": volatility_metrics(val_prediction, val_true),
        "ensemble_test": volatility_metrics(test_prediction, test_true),
        "seed_test_mean": {
            k: float(np.mean([r.test[k] for r in seed_results])) for k in keys
        },
        "seed_test_std": {
            k: float(np.std([r.test[k] for r in seed_results], ddof=1))
            if len(seed_results) > 1 else float("nan")
            for k in keys
        },
        "seeds": [
            {
                "seed": r.seed,
                "best_epoch": r.best_epoch,
                "initial_trainable_hash": r.initial_trainable_hash,
                "validation": r.validation,
                "test": r.test,
            }
            for r in seed_results
        ],
    }
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(artifact_dir / "summary.json", payload)
        write_npz_atomic(
            artifact_dir / "predictions.npz",
            validation_dates=np.asarray(bundle.dates["val"], dtype="datetime64[ns]"),
            test_dates=np.asarray(bundle.dates["test"], dtype="datetime64[ns]"),
            validation_prediction=val_prediction,
            test_prediction=test_prediction,
            validation_true=val_true,
            test_true=test_true,
        )
        for result in seed_results:
            torch.save(result.state_dict, artifact_dir / f"seed_{result.seed}.pt")
    payload["_predictions"] = {
        "validation": val_prediction, "test": test_prediction,
        "validation_true": val_true, "test_true": test_true,
    }
    return payload


# ---------------------------------------------------------------------------
# lambda selection on the development universe
# ---------------------------------------------------------------------------

DEFAULT_LAMBDA_GRID = (0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0)


def sweep_lambda(
    cfg: ExperimentConfig,
    dev_bundles: list[DataBundle],
    corrector: str = "bimamba",
    grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
    reference: str = "garch",
    artifact_dir: Path | None = None,
) -> tuple[float, list[dict]]:
    """Choose lambda on the development markets using validation QLIKE only.

    GINN tunes its weight on a separate index held aside for that purpose; the
    development universe plays the same role here.  Test scores are never read.
    """
    records = []
    for lam in grid:
        scores = []
        for bundle in dev_bundles:
            result = run_baseline_ensemble(
                cfg, bundle, "ginn", corrector=corrector, lam=lam,
                reference=reference, progress=False,
            )
            scores.append(result["ensemble_validation"][cfg.selection_metric])
        record = {
            "lam": float(lam),
            "dev_validation_mean": float(np.mean(scores)),
            "per_market": {b.index_name: float(s) for b, s in zip(dev_bundles, scores)},
        }
        print(f"[lambda sweep] lam={lam:<5} dev val "
              f"{cfg.selection_metric}={record['dev_validation_mean']:.4f}")
        records.append(record)
    best = min(records, key=lambda r: r["dev_validation_mean"])
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            artifact_dir / "lambda_sweep.json",
            {"grid": list(grid), "reference": reference,
             "records": records, "selected": best["lam"]},
        )
    return float(best["lam"]), records


def parameter_report(
    cfg: ExperimentConfig,
    bundle: DataBundle,
    corrector: str = "bimamba",
    append: str = "garch",
) -> dict[str, int]:
    """Trainable counts for the paper's capacity paragraph."""
    reference = AnchoredCorrector(
        cfg, bundle.n_features, bundle.n_core, bundle.core_coef,
        corrector=corrector, freeze_core=True, anchor=True, use_core=True,
    )
    report = {
        "anchored": sum(p.numel() for p in reference.parameters() if p.requires_grad)
    }
    for kind in BASELINE_KINDS:
        model = build_baseline_model(
            kind, cfg, bundle, corrector=corrector, append=append
        )
        report[kind] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return report
