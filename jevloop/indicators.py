"""Phase 1 features: indicators derived from hourly base data, in code.

Every feature at row t uses only bars that have closed by t (rows are
indexed by bar close time). Nothing here is sent to Jev: TypeSafe's own
docs say Jev "is not a calculator" and compares numbers poorly, so numeric
work stays in code and a statistical model weighs the indicators.

Groups:
- returns over 1h .. 1 week
- trend: distance from SMA 20/50/100/200, SMA-50 slope, 50/200 cross, MACD
- momentum / mean reversion: RSI 14/48, stochastic %K/%D, 1-week z-score
- volatility: ATR, Bollinger %B and width, realised vol 24h/1w, their
  ratio, where 24h vol sits in its 30-day range
- volume: 24h volume z-score vs 30 days, on-balance-volume flow, VWAP gap
- structure: distance to and hours since the 72h high / low
- time: hour of day (sin/cos), weekend
- cross-asset: ETH/BTC ratio momentum, ETH 24h return
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .history import Bars

HOUR = 3600


def bars_to_frame(bars: Bars, bar_seconds: int = HOUR, fill_grid: bool = True) -> pd.DataFrame:
    """Bars (keyed by bar start) -> DataFrame indexed by bar CLOSE time.

    fill_grid=True (24/7 crypto): a complete grid, where a missing bar
    carries the last close forward with zero volume (nothing traded).
    fill_grid=False (stocks): only the bars that exist, so nights and
    weekends are not filled with fake flat bars; windows then count
    trading bars, not clock hours."""
    ks = sorted(bars)
    idx = pd.to_datetime([k + bar_seconds for k in ks], unit="s", utc=True)
    df = pd.DataFrame([bars[k] for k in ks], index=idx, columns=["open", "high", "low", "close", "volume"])
    if not fill_grid:
        return df
    full = pd.date_range(idx[0], idx[-1], freq=pd.Timedelta(seconds=bar_seconds), tz="UTC")
    df = df.reindex(full)
    df["close"] = df["close"].ffill()
    for col in ("open", "high", "low"):
        df[col] = df[col].fillna(df["close"])
    df["volume"] = df["volume"].fillna(0.0)
    return df


def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _hours_since_extreme(s: pd.Series, n: int, highest: bool) -> pd.Series:
    fn = np.argmax if highest else np.argmin
    return s.rolling(n).apply(lambda w: n - 1 - fn(w), raw=True)


def build_features(btc: pd.DataFrame, eth: pd.DataFrame, main_label: str = "btc", other_label: str = "eth",
                   tz: str = "UTC", weekends: bool = True) -> pd.DataFrame:
    """Indicators for `btc` (the traded asset) with `eth` as the
    cross-asset. Windows count bars (hourly for crypto, 30-minute
    regular-hours bars for stocks). Time of day is taken in `tz`."""
    c, h, lo, v = btc["close"], btc["high"], btc["low"], btc["volume"]
    logc = np.log(c)
    f = pd.DataFrame(index=btc.index)

    for k in (1, 4, 12, 24, 72, 168):
        f[f"ret_{k}h"] = logc - logc.shift(k)

    for n in (20, 50, 100, 200):
        f[f"dist_sma{n}"] = c / c.rolling(n).mean() - 1
    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    f["sma50_slope_10h"] = sma50 / sma50.shift(10) - 1
    f["sma50_above_sma200"] = (sma50 > sma200).astype(float).where(sma200.notna())

    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = (ema12 - ema26) / c
    signal = macd.ewm(span=9, adjust=False).mean()
    f["macd"], f["macd_hist"] = macd, macd - signal

    f["rsi_14"], f["rsi_48"] = _rsi(c, 14), _rsi(c, 48)
    ll14, hh14 = lo.rolling(14).min(), h.rolling(14).max()
    stoch_k = 100 * (c - ll14) / (hh14 - ll14).replace(0, np.nan)
    f["stoch_k"], f["stoch_d"] = stoch_k, stoch_k.rolling(3).mean()
    f["zscore_168h"] = (c - c.rolling(168).mean()) / c.rolling(168).std()

    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_14_pct"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / c
    m20, s20 = c.rolling(20).mean(), c.rolling(20).std()
    f["boll_pctb"] = (c - (m20 - 2 * s20)) / (4 * s20).replace(0, np.nan)
    f["boll_width"] = 4 * s20 / m20
    r1 = logc.diff()
    rv24, rv168 = r1.rolling(24).std(), r1.rolling(168).std()
    f["rv_24h"], f["rv_168h"] = rv24, rv168
    f["rv_ratio_24_168"] = rv24 / rv168
    lo720, hi720 = rv24.rolling(720).min(), rv24.rolling(720).max()
    f["rv24_in_30d_range"] = (rv24 - lo720) / (hi720 - lo720).replace(0, np.nan)

    lv = np.log1p(v.rolling(24).sum())
    f["volume_z_24h_vs_30d"] = (lv - lv.rolling(720).mean()) / lv.rolling(720).std()
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    f["obv_flow_24h"] = (obv - obv.shift(24)) / v.rolling(24).sum().replace(0, np.nan)
    vwap24 = (c * v).rolling(24).sum() / v.rolling(24).sum().replace(0, np.nan)
    f["dist_vwap_24h"] = c / vwap24 - 1

    hh72, ll72 = h.rolling(72).max(), lo.rolling(72).min()
    f["dist_high_72h"], f["dist_low_72h"] = c / hh72 - 1, c / ll72 - 1
    f["hours_since_high_72h"] = _hours_since_extreme(h, 72, highest=True)
    f["hours_since_low_72h"] = _hours_since_extreme(lo, 72, highest=False)

    local = btc.index.tz_convert(tz)
    hour = local.hour + local.minute / 60
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    if weekends:
        f["weekend"] = (local.dayofweek >= 5).astype(float)

    pair = f"{other_label}{main_label}"
    ratio = np.log(eth["close"].reindex(btc.index).ffill() / c)
    f[f"{pair}_ret_24h"] = ratio - ratio.shift(24)
    f[f"{pair}_ret_168h"] = ratio - ratio.shift(168)
    loge = np.log(eth["close"].reindex(btc.index).ffill())
    f[f"{other_label}_ret_24h"] = loge - loge.shift(24)
    return f


def forward_log_return(btc: pd.DataFrame, horizon_h: int) -> pd.Series:
    """Label for row t: log return from close at t to the close `horizon_h`
    bars later."""
    logc = np.log(btc["close"])
    return logc.shift(-horizon_h) - logc


def label_end_time(btc: pd.DataFrame, horizon_h: int) -> pd.Series:
    """When row t's label is known: the close time `horizon_h` bars later.
    On stocks that can be the next morning, so walk-forward must use this,
    not t + horizon hours."""
    return pd.Series(btc.index, index=btc.index).shift(-horizon_h)
