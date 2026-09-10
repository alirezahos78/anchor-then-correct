#!/usr/bin/env python3
"""Build the frozen multi-market ICASSP 2027 dataset.

The builder is configuration driven.  It never selects or drops a market based
on forecasting results.  Each output file contains a uniform ``ANCHOR`` column,
while the manifest records whether that column is VIX, VXN, RVX, VXD, GVZ, OVX,
or VXEEM for the market.  Yahoo's end date is exclusive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


LOG2 = np.log(2.0)


def _flatten(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.copy()
        frame.columns = frame.columns.get_level_values(0)
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    return frame.loc[~frame.index.duplicated()].sort_index()


def download_one(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    frame = yf.download(
        ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False, timeout=30,
    )
    if frame is None or frame.empty:
        raise RuntimeError(f"no data returned for {ticker}")
    frame = _flatten(frame)
    columns = [name for name in ("Open", "High", "Low", "Close", "Volume") if name in frame]
    if "Close" not in columns:
        raise RuntimeError(f"download for {ticker} has no Close column")
    return frame[columns].astype("float64")


def cached_download(
    ticker: str, start: str, end: str, cache_dir: Path | None = None, retries: int = 3,
) -> pd.DataFrame:
    """Retain each successful raw series so interrupted builds need fewer requests."""
    request = {"ticker": ticker, "start": start, "end": end, "auto_adjust": True, "schema": 1}
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    csv_path = metadata_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        csv_path, metadata_path = cache_dir / f"{key}.csv", cache_dir / f"{key}.json"
        if csv_path.is_file() and metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if (metadata["request"] == request
                        and metadata["sha256"] == hashlib.sha256(csv_path.read_bytes()).hexdigest()):
                    frame = pd.read_csv(csv_path, index_col=0, parse_dates=True, float_precision="round_trip")
                    if not frame.empty and "Close" in frame:
                        print(f"[raw cache] {ticker}", flush=True)
                        return _flatten(frame).astype("float64")
            except (OSError, ValueError, KeyError):
                pass

    for attempt in range(1, retries + 1):
        try:
            print(f"[download {attempt}/{retries}] {ticker}", flush=True)
            frame = download_one(ticker, start, end)
            if cache_dir is not None:
                temporary = csv_path.with_suffix(".csv.tmp")
                frame.to_csv(temporary)
                temporary.replace(csv_path)
                metadata = {"request": request, "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest()}
                temporary = metadata_path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                temporary.replace(metadata_path)
            return frame
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(f"download failed for {ticker} after {retries} attempts: {exc}") from exc
            print(f"[retry] {ticker}: {exc}", flush=True)
            time.sleep(min(2 ** attempt, 8))
    raise AssertionError("unreachable retry state")


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1.0 / length, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1.0 / length, adjust=False).mean()
    return 100.0 - 100.0 / (1.0 + up / down.replace(0, np.nan))


def technical_features(ohlcv: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    close = ohlcv["Close"].astype(float)
    returns = np.log(close).diff()
    features = pd.DataFrame(index=ohlcv.index)
    features["ret"] = returns
    for lag in (1, 2, 3):
        features[f"mom{lag}"] = returns.shift(lag)
    for period in (5, 10, 15, 20):
        features[f"ROC_{period}"] = close / close.shift(period) - 1.0
    for span in (10, 20, 50):
        features[f"EMArat_{span}"] = close / close.ewm(span=span, adjust=False).mean() - 1.0
    variance = returns.pow(2)
    for window in (5, 10, 22):
        features[f"rv_{window}"] = np.sqrt(variance.rolling(window).sum())
        features[f"std_{window}"] = returns.rolling(window).std()

    if all(name in ohlcv for name in ("Open", "High", "Low")):
        opening, high, low = ohlcv["Open"], ohlcv["High"], ohlcv["Low"]
        gk = 0.5 * np.log(high / low).pow(2) - (2.0 * LOG2 - 1.0) * np.log(close / opening).pow(2)
        pk = np.log(high / low).pow(2) / (4.0 * LOG2)
        features["gk"], features["pk"] = gk, pk
        for window in (5, 10, 22):
            features[f"gk_{window}"] = gk.rolling(window).mean()
        features["hl_range"] = np.log(high / low)

    if "Volume" in ohlcv and int((ohlcv["Volume"] > 0).sum()) >= 100:
        volume = ohlcv["Volume"].replace(0, np.nan)
        features["vol_chg"] = np.log(volume).diff()
        for window in (5, 10, 22):
            features[f"volz_{window}"] = (
                (volume - volume.rolling(window).mean()) / volume.rolling(window).std()
            )
    features["rsi14"] = rsi(close)
    return features, returns


def future_target(returns: pd.Series, horizon: int) -> tuple[pd.Series, pd.Series]:
    future_variance = returns.pow(2).rolling(horizon).sum().shift(-horizon)
    realized = np.sqrt(future_variance)
    return np.log(realized.replace(0, np.nan)), realized


def causal_reindex(series: pd.Series, dates: pd.Index) -> pd.Series:
    """Align external closes without backward filling or future information."""
    return series.reindex(dates).ffill(limit=1)


def leak_screen(features: pd.DataFrame, returns: pd.Series, threshold: float = 0.15) -> list[dict]:
    next_return = returns.shift(-1)
    warnings: list[dict] = []
    for column in features:
        good = features[column].notna() & next_return.notna()
        if int(good.sum()) < 100:
            continue
        correlation = float(np.corrcoef(features.loc[good, column], next_return.loc[good])[0, 1])
        if np.isfinite(correlation) and abs(correlation) > threshold:
            warnings.append({"feature": column, "next_return_correlation": correlation})
    return sorted(warnings, key=lambda row: -abs(row["next_return_correlation"]))


def load_universe(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        universe = json.load(handle)
    names = [market["name"] for market in universe["markets"]]
    if len(names) != len(set(names)):
        raise ValueError("market names in the universe must be unique")
    return universe


def build(
    universe_path: Path, out_dir: Path, horizon: int, allow_missing: bool = False,
    markets: list[str] | None = None, cache_dir: Path | None = None,
) -> dict:
    universe = load_universe(universe_path)
    if markets is not None:
        declared = {market["name"]: market for market in universe["markets"]}
        if not markets or len(markets) != len(set(markets)) or any(name not in declared for name in markets):
            raise ValueError("requested markets must be distinct names from the configured universe")
        universe = {**universe, "markets": [declared[name] for name in markets]}
    start, end = universe["start_inclusive"], universe["end_exclusive"]
    requests: dict[str, str] = {}
    for market in universe["markets"]:
        requests[f"target:{market['name']}"] = market["ticker"]
        requests[f"anchor:{market['anchor_name']}"] = market["anchor_ticker"]
    for name, ticker in universe.get("shared_exogenous", {}).items():
        requests[f"exo:{name}"] = ticker
    reference_anchor = universe.get("reference_anchor")
    if reference_anchor:
        requests[f"anchor:{reference_anchor['name']}"] = reference_anchor["ticker"]
    for name, ticker in universe.get("spillover_benchmarks", {}).items():
        requests[f"spill:{name}"] = ticker

    by_ticker: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    for ticker in sorted(set(requests.values())):
        try:
            frame = cached_download(ticker, start, end, cache_dir=cache_dir)
            by_ticker[ticker] = frame
            print(f"[data] {ticker:8s} {len(frame):5d} rows "
                  f"{frame.index.min().date()} -> {frame.index.max().date()}")
        except Exception as exc:  # preserve the exact unavailable ticker in the manifest
            failures[ticker] = f"{type(exc).__name__}: {exc}"
    if failures and not allow_missing:
        details = "\n".join(f"  {ticker}: {message}" for ticker, message in failures.items())
        raise RuntimeError(f"frozen universe download failed; no market was silently dropped:\n{details}")

    out_dir.mkdir(parents=True, exist_ok=True)
    file_records: dict[str, dict] = {}
    omitted: list[dict] = []
    for market in universe["markets"]:
        if market["ticker"] not in by_ticker or market["anchor_ticker"] not in by_ticker:
            omitted.append({"market": market["name"], "reason": "target or anchor unavailable"})
            continue
        target_frame = by_ticker[market["ticker"]]
        features, returns = technical_features(target_frame)

        # Use the same three liquid US benchmarks as causal spillover features.
        for benchmark, ticker in universe.get("spillover_benchmarks", {}).items():
            if ticker not in by_ticker or ticker == market["ticker"]:
                continue
            benchmark_return = np.log(by_ticker[ticker]["Close"]).diff()
            aligned = causal_reindex(benchmark_return, target_frame.index)
            features[f"{benchmark}_ret"] = aligned
            features[f"{benchmark}_rv10"] = np.sqrt(aligned.pow(2).rolling(10).sum())

        anchor_close = causal_reindex(by_ticker[market["anchor_ticker"]]["Close"], target_frame.index)
        features["ANCHOR"] = anchor_close
        features["ANCHOR_chg"] = anchor_close.diff()
        if reference_anchor and reference_anchor["ticker"] in by_ticker:
            reference_close = causal_reindex(
                by_ticker[reference_anchor["ticker"]]["Close"], target_frame.index
            )
            features[reference_anchor["name"]] = reference_close
            features[f"{reference_anchor['name']}_chg"] = reference_close.diff()
        for name, ticker in universe.get("shared_exogenous", {}).items():
            if ticker not in by_ticker:
                continue
            external = causal_reindex(by_ticker[ticker]["Close"], target_frame.index)
            features[name] = external
            features[f"{name}_chg"] = external.diff()

        target_log, target_rv = future_target(returns, horizon)
        output = features.copy()
        output["Close"] = target_frame["Close"]
        output["target_logrv"] = target_log
        output["target_rv"] = target_rv
        path = out_dir / f"{market['name']}_vol.csv"
        output.to_csv(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        warnings = leak_screen(features, returns)
        file_records[market["name"]] = {
            **market,
            "path": path.name,
            "sha256": digest,
            "rows_total": int(len(output)),
            "rows_complete": int(len(output.dropna())),
            "first_date": str(output.index.min().date()),
            "last_date": str(output.index.max().date()),
            "leak_screen_warnings": warnings[:10],
        }
        status = "OK" if not warnings else f"WARN({len(warnings)})"
        print(f"[built] {market['name']:5s} anchor={market['anchor_name']:5s} "
              f"complete={len(output.dropna()):5d} leak-screen={status}")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Yahoo Finance via yfinance",
        "universe_file": str(universe_path),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "requested_markets": [market["name"] for market in universe["markets"]],
        "download_start_inclusive": start,
        "download_end_exclusive": end,
        "requested_cutoff_inclusive": universe["requested_cutoff_inclusive"],
        "target_horizon": horizon,
        "strict_universe": not allow_missing,
        "download_failures": failures,
        "omitted_markets": omitted,
        "files": file_records,
    }
    with (out_dir / "data_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    with (out_dir / "market_universe_resolved.json").open("w", encoding="utf-8") as handle:
        json.dump(universe, handle, indent=2, sort_keys=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build frozen multi-market volatility data")
    parser.add_argument("--universe", default="configs/market_universe.json")
    parser.add_argument("--out-dir", default="vol_dataset")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--markets", nargs="+", help="build these configured markets and their inputs")
    parser.add_argument("--cache-dir", help="reuse successful raw downloads on subsequent attempts")
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(
        Path(args.universe), Path(args.out_dir), args.horizon, args.allow_missing,
        markets=args.markets, cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )
