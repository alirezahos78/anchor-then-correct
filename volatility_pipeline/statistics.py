from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def daily_qlike(pred_logrv: np.ndarray, true_logrv: np.ndarray) -> np.ndarray:
    """QLIKE loss for log volatility predictions."""
    r = np.clip(2.0 * (true_logrv - pred_logrv), -30.0, 30.0)
    return np.exp(r) - r - 1.0


def daily_squared_error(pred_logrv: np.ndarray, true_logrv: np.ndarray) -> np.ndarray:
    return np.square(np.asarray(pred_logrv) - np.asarray(true_logrv))


@dataclass(frozen=True)
class DMResult:
    statistic: float
    pvalue: float
    mean_difference: float
    n: int
    lag: int


def _bartlett_lrv(centered: np.ndarray, lag: int) -> float:
    """Newey-West long-run variance with Bartlett weights."""
    n = len(centered)
    gamma0 = float(np.dot(centered, centered) / n)
    lrv = gamma0
    for k in range(1, min(lag, n - 1) + 1):
        gamma = float(np.dot(centered[k:], centered[:-k]) / n)
        weight = 1.0 - k / (lag + 1.0)
        lrv += 2.0 * weight * gamma
    return lrv


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, horizon: int, lag: int | None = None) -> DMResult:
    """Two-sided Diebold-Mariano test with NW/Bartlett HAC and HLN correction."""
    a, b = np.asarray(loss_a, float), np.asarray(loss_b, float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("loss arrays must be one-dimensional and have identical shapes")
    good = np.isfinite(a) & np.isfinite(b)
    d = a[good] - b[good]
    n = len(d)
    if n < 10:
        return DMResult(float("nan"), float("nan"), float("nan"), n, 0)
    lag = max(horizon - 1, 0) if lag is None else max(int(lag), 0)
    dbar = float(d.mean())
    lrv = _bartlett_lrv(d - dbar, lag)
    variance = lrv / n
    hln = (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
    tolerance = np.finfo(float).eps * max(float(np.square(d).mean()), 1.0)
    if variance <= tolerance or hln <= 0:
        return DMResult(float("nan"), float("nan"), dbar, n, lag)
    statistic = dbar / np.sqrt(variance) * np.sqrt(hln)
    pvalue = float(2.0 * stats.t.sf(abs(statistic), df=n - 1))
    return DMResult(float(statistic), pvalue, dbar, n, lag)


def fold_aware_dm(
    loss_a_folds: list[np.ndarray],
    loss_b_folds: list[np.ndarray],
    horizon: int,
    lag: int | None = None,
) -> DMResult:
    """Pooled DM without autocovariance products across fold boundaries."""
    if len(loss_a_folds) != len(loss_b_folds) or not loss_a_folds:
        raise ValueError("fold collections must be non-empty and aligned")
    lag = max(horizon - 1, 0) if lag is None else max(int(lag), 0)
    diffs: list[np.ndarray] = []
    for a, b in zip(loss_a_folds, loss_b_folds):
        a, b = np.asarray(a, float), np.asarray(b, float)
        good = np.isfinite(a) & np.isfinite(b)
        diffs.append(a[good] - b[good])
    joined = np.concatenate(diffs)
    n = len(joined)
    dbar = float(joined.mean())
    centered = [d - dbar for d in diffs]
    gamma0 = sum(float(np.dot(d, d)) for d in centered) / n
    lrv = gamma0
    for k in range(1, lag + 1):
        numerator = sum(float(np.dot(d[k:], d[:-k])) for d in centered if len(d) > k)
        gamma = numerator / n
        lrv += 2.0 * (1.0 - k / (lag + 1.0)) * gamma
    variance = lrv / n
    hln = (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
    tolerance = np.finfo(float).eps * max(float(np.square(joined).mean()), 1.0)
    if variance <= tolerance or hln <= 0:
        return DMResult(float("nan"), float("nan"), dbar, n, lag)
    statistic = dbar / np.sqrt(variance) * np.sqrt(hln)
    return DMResult(float(statistic), float(2 * stats.t.sf(abs(statistic), n - 1)), dbar, n, lag)


def adjust_pvalues(pvalues: list[float], method: str = "holm") -> np.ndarray:
    """Holm FWER or Benjamini-Hochberg FDR adjustment without statsmodels."""
    p = np.asarray(pvalues, float)
    out = np.full_like(p, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return out
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    if method == "holm":
        adjusted = np.maximum.accumulate((m - np.arange(m)) * ranked)
    elif method in {"bh", "fdr_bh"}:
        adjusted = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    else:
        raise ValueError("method must be holm or bh")
    restored = np.empty(m)
    restored[order] = np.minimum(adjusted, 1.0)
    out[valid] = restored
    return out


def moving_block_bootstrap_mean_ci(
    values: np.ndarray,
    block_length: int = 10,
    repetitions: int = 2000,
    seed: int = 2027,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Circular moving-block bootstrap CI for a serially dependent mean."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": n}
    length = max(1, min(int(block_length), n))
    repetitions = max(1, int(repetitions))
    blocks_needed = int(np.ceil(n / length))
    offsets = np.arange(length)
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        starts = rng.integers(0, n, size=blocks_needed)
        indices = ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]
        means[repetition] = float(values[indices].mean())
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(means, alpha)),
        "ci_high": float(np.quantile(means, 1.0 - alpha)),
        "n": n,
        "block_length": length,
        "repetitions": repetitions,
    }


def spearman_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alternative: str = "greater",
    repetitions: int = 20000,
    seed: int = 2027,
) -> dict[str, float | int | str]:
    """Asset-level permutation test; assets, not daily rows, are exchangeable."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    n = len(x)
    if n < 3:
        return {"rho": float("nan"), "pvalue": float("nan"), "n": n, "alternative": alternative}
    observed = float(stats.spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(int(repetitions)):
        statistic = float(stats.spearmanr(x, rng.permutation(y)).statistic)
        if alternative == "greater":
            exceed += statistic >= observed
        elif alternative == "less":
            exceed += statistic <= observed
        elif alternative == "two-sided":
            exceed += abs(statistic) >= abs(observed)
        else:
            raise ValueError("alternative must be greater, less, or two-sided")
    return {
        "rho": observed,
        "pvalue": float((exceed + 1) / (int(repetitions) + 1)),
        "n": n,
        "alternative": alternative,
        "permutations": int(repetitions),
    }
