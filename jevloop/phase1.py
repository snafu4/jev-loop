"""`jevloop phase1`: do indicators derived from base data predict BTC?

Walk-forward test on hourly Coinbase data since 2021:
- every month, fit each model only on rows whose outcome was known before
  that month began (row t needs t + horizon <= month start), then predict
  the month;
- the last `--holdout-days` are never used unless `--include-holdout`:
  save them for one final check;
- models: the up-rate so far (the bar to beat), 24h momentum alone,
  logistic regression on all indicators, shallow gradient boosting on all
  indicators;
- scores: AUC with a 95% interval from resampling whole days (neighbouring
  hours are not independent), Brier skill against the up-rate model, AUC
  per year, and trading it: decide every `horizon` hours, buy when P(up)
  clears a threshold fixed in advance, hold for the horizon, pay 0.5%
  round-trip taker fees (Alpaca crypto, lowest tier).

No Jev calls: this is the baseline Jev's features would have to improve.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .history import LOG_DIR, load_hourly, load_stock_bars
from .indicators import bars_to_frame, build_features, forward_log_return, label_end_time

OUT_DIR = LOG_DIR / "phase1"

# Per market: data, the default horizons (in bars) and round-trip cost.
# BTC: Alpaca crypto fees at the lowest tier, 0.25% each way.
# SPY: commission-free; the 1-cent spread (~0.002%/side on ~$600) and the
# SEC/FINRA fees on sells come to well under 0.01%; 0.02% is conservative.
MARKETS = {
    "btc": {"horizons": [4, 24], "cost": 0.005, "bar": "1h", "short": False},
    "spy": {"horizons": [8, 13], "cost": 0.0002, "bar": "30m regular-hours", "short": True},
}
THRESHOLDS = (0.55, 0.60)  # fixed in advance, not tuned on results


def _models(features: list[str], momentum_col: str) -> dict:
    return {
        "logistic, all indicators": (
            features,
            lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)),
        ),
        "gradient boosting, all indicators": (
            features,
            lambda: HistGradientBoostingClassifier(
                max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=200,
                l2_regularization=1.0, random_state=0,
            ),
        ),
        f"momentum only ({momentum_col})": (
            [momentum_col],
            lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        ),
    }


def walk_forward(data: pd.DataFrame, features: list[str], horizon_h: int, test_end: pd.Timestamp,
                 first_train_days: int = 365) -> pd.DataFrame:
    """Out-of-sample P(up) for every row from first_train_days in until the
    last row whose outcome is known by test_end. `data["label_end"]` is when
    each row's outcome became known (t + horizon on 24/7 crypto; can be the
    next morning on stocks)."""
    momentum_col = f"ret_{horizon_h}h" if f"ret_{horizon_h}h" in features else "ret_24h"
    models = _models(features, momentum_col)
    months = pd.date_range(data.index[0] + pd.Timedelta(days=first_train_days), data.index[-1], freq="MS", tz="UTC")
    parts = []
    for ms in months:
        me = ms + pd.offsets.MonthBegin(1)
        train = data[data["label_end"] <= ms]  # only outcomes already known
        test = data[(data.index >= ms) & (data.index < me) & (data["label_end"] <= test_end)]
        if test.empty or train["up"].nunique() < 2:
            continue
        out = pd.DataFrame(index=test.index)
        out["y"] = test["up"].values
        out["ret"] = test["fwd"].values
        out["base rate so far"] = train["up"].mean()
        for name, (cols, make) in models.items():
            model = make().fit(train[cols].values, train["up"].values)
            out[name] = model.predict_proba(test[cols].values)[:, 1]
        parts.append(out)
    return pd.concat(parts)


def _auc_ci(y: np.ndarray, p: np.ndarray, days: np.ndarray, n_boot: int = 400, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    uniq = np.unique(days)
    groups = {d: np.flatnonzero(days == d) for d in uniq}
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[d] for d in rng.choice(uniq, size=len(uniq), replace=True)])
        if len(np.unique(y[idx])) == 2:
            vals.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _brier_skill(y, p, p_base) -> float:
    return 1 - np.mean((p - y) ** 2) / np.mean((p_base - y) ** 2)


def report(oos: pd.DataFrame, horizon_h: int, cost: float = 0.005, bar: str = "1h", short: bool = False) -> dict:
    y = oos["y"].values.astype(int)
    days = oos.index.floor("D").values
    model_cols = [c for c in oos.columns if c not in ("y", "ret")]
    print(f"\n=== horizon {horizon_h} bars ({bar} bars): {len(oos):,} out-of-sample bars, "
          f"{oos.index[0]:%Y-%m-%d} .. {oos.index[-1]:%Y-%m-%d}, up-rate {100 * y.mean():.1f}%")
    print(f"  {'model':38s} {'AUC':>6s}  {'95% interval':>15s}  {'Brier skill':>11s}  verdict")
    summary = {}
    for col in model_cols:
        p = oos[col].values
        auc = roc_auc_score(y, p)
        lo, hi = _auc_ci(y, p, days)
        skill = _brier_skill(y, p, oos["base rate so far"].values)
        verdict = "above chance" if lo > 0.5 else "below chance" if hi < 0.5 else "no better than chance"
        print(f"  {col:38s} {auc:6.3f}  {lo:6.3f} - {hi:6.3f}  {skill:+11.4f}  {verdict}")
        summary[col] = {"auc": auc, "ci": [lo, hi], "brier_skill": skill}

    print("  AUC by year:")
    years = sorted(set(oos.index.year))
    print("  " + " " * 38 + "".join(f"{yr:>8d}" for yr in years))
    for col in model_cols:
        if col == "base rate so far":
            continue
        cells = []
        for yr in years:
            m = oos.index.year == yr
            cells.append(f"{roc_auc_score(y[m], oos[col].values[m]):8.3f}" if len(set(y[m])) == 2 else "     n/a")
        print(f"  {col:38s}" + "".join(cells))

    grid = oos.iloc[::horizon_h]  # one decision per horizon: trades never overlap
    gross = np.expm1(grid["ret"].values)
    print(f"  trading it (a decision every {horizon_h} bars, hold {horizon_h} bars, "
          f"{100 * cost:.2f}% round-trip cost):")
    print(f"    {'always long':36s} n={len(grid):5d}  mean net {100 * np.mean(gross - cost):+.3f}% per trade"
          f"  (before costs {100 * np.mean(gross):+.3f}%)")
    sides = [("long ", 1)] + ([("short", -1)] if short else [])
    for col in model_cols:
        if col == "base rate so far":
            continue
        for thr in THRESHOLDS:
            for side, sign in sides:
                take = grid[col].values >= thr if sign > 0 else grid[col].values <= 1 - thr
                label = f"P>={thr:.2f}" if sign > 0 else f"P<={1 - thr:.2f}"
                if take.sum() == 0:
                    print(f"    {col[:22]:22s} {side} {label}  n=    0")
                    continue
                pnl = sign * gross[take]
                net = pnl - cost
                print(f"    {col[:22]:22s} {side} {label}  n={take.sum():5d}  before costs {100 * pnl.mean():+.3f}%"
                      f"  net {100 * net.mean():+.3f}% per trade  hit {100 * (net > 0).mean():4.1f}%"
                      f"  total {100 * net.sum():+7.1f}%")
    return summary


def _load(market: str, start: dt.datetime):
    if market == "btc":
        btc = bars_to_frame(load_hourly("BTC-USD", start))
        eth = bars_to_frame(load_hourly("ETH-USD", start))
        return btc, build_features(btc, eth)
    spy = bars_to_frame(load_stock_bars("SPY", start), bar_seconds=1800, fill_grid=False)
    qqq = bars_to_frame(load_stock_bars("QQQ", start), bar_seconds=1800, fill_grid=False)
    return spy, build_features(spy, qqq, main_label="spy", other_label="qqq", tz="America/New_York", weekends=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jev-loop phase1")
    parser.add_argument("--market", choices=sorted(MARKETS), default="btc",
                        help="btc: hourly Coinbase BTC-USD (ETH as cross-asset); spy: 30-minute regular-hours SPY (QQQ)")
    parser.add_argument("--start", default=None, help="first day of history (btc 2021-01-01, spy 2019-01-01)")
    parser.add_argument("--horizons", type=int, nargs="+", default=None, help="prediction horizons in bars")
    parser.add_argument("--cost", type=float, default=None, help="round-trip trading cost as a fraction")
    parser.add_argument("--holdout-days", type=int, default=180, help="most recent days kept out")
    parser.add_argument("--include-holdout", action="store_true", help="score the held-out period instead (do this once)")
    args = parser.parse_args(argv)
    cfg = MARKETS[args.market]
    horizons = args.horizons or cfg["horizons"]
    cost = cfg["cost"] if args.cost is None else args.cost

    start = dt.datetime.fromisoformat(args.start or ("2021-01-01" if args.market == "btc" else "2019-01-01")).replace(tzinfo=dt.UTC)
    print("loading history (cached)...")
    prices, feats = _load(args.market, start)
    features = list(feats.columns)
    holdout_start = prices.index[-1] - pd.Timedelta(days=args.holdout_days)
    print(f"{args.market.upper()}: {len(features)} indicators on {cfg['bar']} bars; history {prices.index[0]:%Y-%m-%d} .. "
          f"{prices.index[-1]:%Y-%m-%d}; holdout from {holdout_start:%Y-%m-%d} "
          f"({'SCORED' if args.include_holdout else 'untouched'}); round-trip cost {100 * cost:.2f}%")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    tag = "holdout" if args.include_holdout else "dev"
    for h in horizons:
        data = feats.copy()
        data["fwd"] = forward_log_return(prices, h)
        data["up"] = (data["fwd"] > 0).astype(int)
        data["label_end"] = label_end_time(prices, h)
        data = data.dropna()
        if args.include_holdout:
            oos = walk_forward(data, features, h, prices.index[-1])
            oos = oos[oos.index >= holdout_start]
        else:
            oos = walk_forward(data[data["label_end"] < holdout_start], features, h, holdout_start)
        oos.to_csv(OUT_DIR / f"oos_{args.market}_{tag}_{h}.csv")
        results[f"{h} bars"] = report(oos, h, cost=cost, bar=cfg["bar"], short=cfg["short"])
    (OUT_DIR / f"summary_{args.market}_{tag}.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
