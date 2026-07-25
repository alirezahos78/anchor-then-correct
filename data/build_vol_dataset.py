#!/usr/bin/env python3
"""
Build a CLEAN, CAUSAL, leak-free dataset for VOLATILITY forecasting of the three
main US indices (NASDAQ Composite, NYSE Composite, Dow Jones) from official daily
data (Yahoo Finance via yfinance).

What it does
------------
1. Pulls daily OHLCV for ^IXIC, ^NYA, ^DJI  + exogenous ^VIX, ^TNX
2. Builds CNNpred-style technical features -- ALL strictly backward-looking (causal)
3. Builds an h-day-ahead realized-volatility target from FUTURE returns only
4. Runs leak self-checks (no feature may correlate with the next-day signed return;
   target window never overlaps the feature window)
5. Saves one CSV per index: <NAME>_vol.csv  (Date, features..., Close, target_logrv, target_rv)

Run on a machine with internet (e.g. Colab):  pip install yfinance && python build_vol_dataset.py
Everything is configurable in CONFIG below (or via --start/--end/--horizon).

NOTE on the target: with daily data the realized variance is the close-to-close
sum of squared log returns over the next h days. This is the standard daily proxy
(HAR/GARCH live here too). If you later get 5-min data, swap in a proper realized
variance -- the rest of the pipeline is unchanged.
"""
import argparse, sys
import numpy as np
import pandas as pd

CONFIG = dict(
    start="2010-01-01",
    end=None,                 # None -> today
    horizon=5,                # predict realized vol over the NEXT h trading days
    indices={"IXIC": "^IXIC", "NYA": "^NYA", "DJI": "^DJI"},   # NASDAQ / NYSE / Dow
    exo={"VIX": "^VIX", "VXN": "^VXN", "TNX": "^TNX"},          # implied vols (S&P + NASDAQ), 10Y yield (shared, exogenous)
    vol_windows=(5, 10, 22),
    roc_periods=(5, 10, 15, 20),
    ema_spans=(10, 20, 50),
    out_dir="vol_dataset",
)

LOG2 = np.log(2.0)


# ----------------------------- causal feature helpers -----------------------------
def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def garman_klass(o, h, l, c):
    # daily variance estimator from OHLC (causal: only day-t bar)
    return 0.5 * np.log(h / l) ** 2 - (2 * LOG2 - 1) * np.log(c / o) ** 2


def parkinson(h, l):
    return (1.0 / (4 * LOG2)) * np.log(h / l) ** 2


def build_features(name, data, exo, cfg):
    """All features use information up to and including day t only."""
    df = data[name]
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]
    r = np.log(c).diff()                                   # daily log return
    f = pd.DataFrame(index=df.index)

    # momentum / rate-of-change (backward)
    f["ret"] = r
    f["mom"] = r                                           # today's return (known at close of t)
    f["mom1"], f["mom2"], f["mom3"] = r.shift(1), r.shift(2), r.shift(3)
    for k in cfg["roc_periods"]:
        f[f"ROC_{k}"] = c / c.shift(k) - 1.0
    for s in cfg["ema_spans"]:
        f[f"EMArat_{s}"] = c / c.ewm(span=s, adjust=False).mean() - 1.0   # ratio is stationary

    # past realized volatility (legit predictors of future vol -- this is the signal)
    var = r ** 2
    for w in cfg["vol_windows"]:
        f[f"rv_{w}"] = np.sqrt(var.rolling(w).sum())
        f[f"std_{w}"] = r.rolling(w).std()

    # range-based daily vol (OHLC)
    gk, pk = garman_klass(o, h, l, c), parkinson(h, l)
    f["gk"], f["pk"] = gk, pk
    for w in cfg["vol_windows"]:
        f[f"gk_{w}"] = gk.rolling(w).mean()
    f["hl_range"] = np.log(h / l)

    # volume
    f["vol_chg"] = np.log(v.replace(0, np.nan)).diff()
    for w in cfg["vol_windows"]:
        f[f"volz_{w}"] = (v - v.rolling(w).mean()) / v.rolling(w).std()

    f["rsi14"] = rsi(c, 14)

    # cross-market spillover: other indices' return and past vol (known at close of t)
    for other in data:
        if other == name:
            continue
        ro = np.log(data[other]["Close"]).diff()
        f[f"{other}_ret"] = ro
        f[f"{other}_rv10"] = np.sqrt((ro ** 2).rolling(10).sum())

    # exogenous: VIX (implied vol, a strong legit predictor) and 10Y yield, level + change
    for ename, edf in exo.items():
        ec = edf["Close"]
        f[ename] = ec
        f[f"{ename}_chg"] = ec.diff()

    return f, c, r


def build_target(r, h):
    """log realized vol over the NEXT h days: sqrt(sum_{i=1..h} r_{t+i}^2). Strictly future."""
    var = r ** 2
    fut_sumsq = var.rolling(h).sum().shift(-h)             # at day t -> sum over t+1..t+h
    rv = np.sqrt(fut_sumsq)
    return np.log(rv.replace(0, np.nan)), rv


# ----------------------------- leak self-check -----------------------------
def leak_check(f, r, thr=0.15):
    """No feature may meaningfully correlate with the NEXT-day signed return.
    (Past-vol features correlating with FUTURE vol is fine -- that's the real signal.)"""
    nxt = r.shift(-1)
    bad = []
    for col in f.columns:
        x = f[col]
        m = x.notna() & nxt.notna()
        if m.sum() < 100:
            continue
        cc = np.corrcoef(x[m], nxt[m])[0, 1]
        if abs(cc) > thr:
            bad.append((col, round(float(cc), 3)))
    return sorted(bad, key=lambda t: -abs(t[1]))


# ----------------------------- data download -----------------------------
def _flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def download(tickers, start, end):
    import yfinance as yf
    out = {}
    for name, tk in tickers.items():
        df = yf.download(tk, start=start, end=end, auto_adjust=True, progress=False)
        if df is None or len(df) == 0:
            raise RuntimeError(f"no data returned for {tk} ({name})")
        df = _flatten(df)
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        out[name] = df[keep].astype("float64")
        print(f"  {name:5s} ({tk}): {len(df)} rows  {df.index.min().date()} -> {df.index.max().date()}")
    return out


def align(frames):
    """Restrict every series to the common set of trading days (US markets share a calendar)."""
    common = None
    for df in frames.values():
        common = df.index if common is None else common.intersection(df.index)
    return {k: v.reindex(common).sort_index() for k, v in frames.items()}


# ----------------------------- main -----------------------------
def main(cfg):
    import os
    print("Downloading daily data from Yahoo Finance ...")
    idx = download(cfg["indices"], cfg["start"], cfg["end"])
    exo = download(cfg["exo"], cfg["start"], cfg["end"])

    idx = align(idx)
    exo = align({**idx, **exo})                            # align exo onto the same calendar
    exo = {k: exo[k] for k in cfg["exo"]}

    # notebooks/ and extras/ expect <NAME>_vol.csv sitting next to them (no out_dir prefix), so every
    # CSV is written both to cfg["out_dir"] and, repo-relatively, straight into notebooks/ and extras/ --
    # this makes `python data/build_vol_dataset.py` + opening any notebook work with no manual file move.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    notebook_dirs = [d for d in (os.path.join(repo_root, "notebooks"), os.path.join(repo_root, "extras"))
                     if os.path.isdir(d)]

    os.makedirs(cfg["out_dir"], exist_ok=True)
    for name in cfg["indices"]:
        f, c, r = build_features(name, idx, exo, cfg)
        y_log, y_rv = build_target(r, cfg["horizon"])

        leaks = leak_check(f, r)
        flag = "OK (no leak)" if not leaks else f"!! LEAK: {leaks[:5]}"
        out = f.copy()
        out["Close"] = c
        out["target_logrv"] = y_log
        out["target_rv"] = y_rv
        usable = out.dropna()
        path = os.path.join(cfg["out_dir"], f"{name}_vol.csv")
        out.to_csv(path)
        dest_paths = [path]
        for d in notebook_dirs:
            dpath = os.path.join(d, f"{name}_vol.csv")
            out.to_csv(dpath)
            dest_paths.append(dpath)
        print(f"{name}: {out.shape[1]-3} features | usable rows {len(usable)} | leak-check {flag} "
              f"-> {', '.join(dest_paths)}")
    print("done.")


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default=CONFIG["start"])
    p.add_argument("--end", default=CONFIG["end"])
    p.add_argument("--horizon", type=int, default=CONFIG["horizon"])
    a = p.parse_args()
    cfg = dict(CONFIG)
    cfg.update(start=a.start, end=a.end, horizon=a.horizon)
    return cfg


if __name__ == "__main__":
    main(_args())
