"""State v2 and replay outcomes, built from 1-minute bars.

State v2 exists because the live snapshot (state.py) gave Jev almost nothing
to judge direction from: three near-zero returns, a top-3 book dominated by
~$80 orders, a tape with ~1 print a minute, and half the fields bookkeeping
zeros. v2 is all market context, computed in code (the split holds):

- returns, ranges and volatility over 15m / 1h / 4h / 24h,
- where the price sits in its 1h and 24h range, distance from 24h VWAP,
- volume in the last hour against the 24h hourly average,
- the gap between Alpaca's price (the venue) and Coinbase's (the market).

State v3 adds the price path itself (build_state_v3): the last 24h as 96
15-minute returns and the last 30 days as daily returns, so Jev can judge
the shape of the move rather than 14 summary numbers.

No absolute prices and no timestamps: Jev must not be able to recognise a
date or price level from its training data and "remember" what came next.

Strict timing: the state for decision minute t uses only bars keyed < t
(a bar keyed k closes at k+60). Outcomes use only bars keyed >= t.
"""

from __future__ import annotations

import math

from .history import Bars

MIN = 60
HOUR_BARS = 60
DAY_BARS = 1440

MOVE_PCT = 0.5  # first-touch and big-move threshold
FLAT_PCT = 0.25  # |1h return| below this counts as flat


def _closes(bars: Bars, end_t: int, n: int, max_missing: float = 0.02) -> list[float] | None:
    """The n closes for bars keyed end_t-n*60 .. end_t-60, forward-filled.
    None if more than max_missing of them are missing."""
    out, last, missing = [], None, 0
    for k in range(end_t - n * MIN, end_t, MIN):
        bar = bars.get(k)
        if bar is None:
            missing += 1
            if last is None:
                continue
            out.append(last)
        else:
            last = bar[3]
            out.append(last)
    if missing > max_missing * n or len(out) < n * (1 - max_missing):
        return None
    return out


def _pct(a: float, b: float) -> float:
    return (b / a - 1) * 100


def _vol_pct_per_hour(closes: list[float]) -> float:
    rets = [math.log(b / a) for a, b in zip(closes, closes[1:]) if a > 0 and b > 0]
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(HOUR_BARS) * 100


def _range_position(closes: list[float]) -> float:
    lo, hi = min(closes), max(closes)
    return 0.5 if hi == lo else (closes[-1] - lo) / (hi - lo)


def build_state_v2(coinbase: Bars, alpaca: Bars, t: int) -> dict | None:
    """Market-context state at decision minute t (epoch seconds, minute
    aligned). None when the 24h of history it needs is incomplete."""
    day = _closes(coinbase, t, DAY_BARS)
    if day is None:
        return None
    last_alpaca = _entry_price(alpaca, t)  # last Alpaca close in the 5 minutes before t
    if last_alpaca is None:
        return None
    now = day[-1]
    hour, four_h = day[-HOUR_BARS:], day[-4 * HOUR_BARS:]

    vols = [coinbase.get(k, (0, 0, 0, 0, 0))[4] for k in range(t - DAY_BARS * MIN, t, MIN)]
    pv = sum(
        coinbase[k][3] * coinbase[k][4]
        for k in range(t - DAY_BARS * MIN, t, MIN)
        if k in coinbase
    )
    vol_sum = sum(vols)
    vwap = pv / vol_sum if vol_sum > 0 else now
    avg_hour_volume = vol_sum / 24
    last_hour_volume = sum(vols[-HOUR_BARS:])

    r = lambda x: round(x, 3)  # noqa: E731
    return {
        "market": "BTC/USD. Prices and volume from Coinbase; the loop trades on Alpaca.",
        "return_15m_pct": r(_pct(day[-16], now)),
        "return_1h_pct": r(_pct(day[-HOUR_BARS - 1], now)),
        "return_4h_pct": r(_pct(day[-4 * HOUR_BARS - 1], now)),
        "return_24h_pct": r(_pct(day[0], now)),
        "volatility_last_1h_pct_per_hour": r(_vol_pct_per_hour(hour)),
        "volatility_last_4h_pct_per_hour": r(_vol_pct_per_hour(four_h)),
        "volatility_last_24h_pct_per_hour": r(_vol_pct_per_hour(day)),
        "range_1h_pct": r(_pct(min(hour), max(hour))),
        "range_24h_pct": r(_pct(min(day), max(day))),
        "position_in_1h_range": r(_range_position(hour)),  # 0 = at the low, 1 = at the high
        "position_in_24h_range": r(_range_position(day)),
        "distance_from_24h_vwap_pct": r(_pct(vwap, now)),
        "volume_last_1h_vs_24h_hourly_avg": r(last_hour_volume / avg_hour_volume)
        if avg_hour_volume > 0
        else None,
        "alpaca_vs_coinbase_price_pct": r(_pct(now, last_alpaca)),
    }


PATH_15M_STEPS = 96  # 24h of 15-minute returns
PATH_DAILY_STEPS = 30  # 30 days of daily returns


def _close_at(bars: Bars, t: int, tolerance_bars: int = 5) -> float | None:
    """Close of the last bar that closed at or before t (bar keyed t-60),
    looking back up to tolerance_bars for a gap."""
    for k in range(t - MIN, t - (tolerance_bars + 1) * MIN, -MIN):
        if k in bars:
            return bars[k][3]
    return None


def _path(bars: Bars, t: int, step_bars: int, steps: int) -> list[float] | None:
    """`steps` consecutive returns (%) ending at t, each over `step_bars`
    minutes, oldest first. None if any anchor price is missing."""
    anchors = []
    for i in range(steps, -1, -1):
        p = _close_at(bars, t - i * step_bars * MIN)
        if p is None:
            return None
        anchors.append(p)
    return [round(_pct(a, b), 3) for a, b in zip(anchors, anchors[1:])]


def build_state_v3(coinbase: Bars, alpaca: Bars, t: int) -> dict | None:
    """State v2 plus the price path itself, so Jev sees the shape of the move
    and not only 14 summary numbers: the last 24h as 96 fifteen-minute
    returns and the last 30 days as daily returns, both oldest first, in
    percent. Still no prices or timestamps. Needs 30 days of Coinbase
    history before t."""
    state = build_state_v2(coinbase, alpaca, t)
    if state is None:
        return None
    path_15m = _path(coinbase, t, 15, PATH_15M_STEPS)
    path_daily = _path(coinbase, t, DAY_BARS, PATH_DAILY_STEPS)
    if path_15m is None or path_daily is None:
        return None
    return {
        **state,
        "returns_15m_last_24h_pct_oldest_first": path_15m,
        "returns_daily_last_30d_pct_oldest_first": path_daily,
    }


STATE_BUILDERS = {"v2": build_state_v2, "v3": build_state_v3}
HISTORY_DAYS_NEEDED = {"v2": 1, "v3": PATH_DAILY_STEPS + 1}


MAX_MISSING_OUTCOME_BARS = 6  # of 60; Alpaca omits minutes with no activity


def _entry_price(alpaca: Bars, t: int, lookback_bars: int = 5) -> float | None:
    """Last Alpaca close in the few minutes before t."""
    for k in range(t - MIN, t - (lookback_bars + 1) * MIN, -MIN):
        if k in alpaca:
            return alpaca[k][3]
    return None


def label_outcome(alpaca: Bars, t: int, horizon_bars: int = HOUR_BARS) -> dict | None:
    """What actually happened on Alpaca over [t, t + horizon), from the last
    Alpaca close before t.

    Alpaca leaves out minutes with no activity (often on weekends); a missing
    minute means the price did not move, so it carries the last close
    forward. None when more than MAX_MISSING_OUTCOME_BARS are missing (too
    little data to say what happened) or there is no recent entry price.

    first_touch: up_first / down_first / neither, or ambiguous when one bar
    reaches both +MOVE_PCT and -MOVE_PCT (order unknown at 1-minute detail).
    """
    entry = _entry_price(alpaca, t)
    if entry is None:
        return None
    up_px, dn_px = entry * (1 + MOVE_PCT / 100), entry * (1 - MOVE_PCT / 100)
    first = None
    last_close = entry
    missing = 0
    for k in range(t, t + horizon_bars * MIN, MIN):
        bar = alpaca.get(k)
        if bar is None:
            missing += 1
            if missing > MAX_MISSING_OUTCOME_BARS:
                return None
            bar = (last_close, last_close, last_close, last_close, 0.0)
        _, high, low, close, _ = bar
        last_close = close
        if first is None:
            hit_up, hit_dn = high >= up_px, low <= dn_px
            if hit_up and hit_dn:
                first = "ambiguous"
            elif hit_up:
                first = "up_first"
            elif hit_dn:
                first = "down_first"
    ret = _pct(entry, last_close)
    return {
        "return_1h_pct": round(ret, 4),
        "direction_1h": "up" if ret > FLAT_PCT else "down" if ret < -FLAT_PCT else "flat",
        "first_touch_1h": first or "neither",
        "big_move_1h": first is not None,
    }
