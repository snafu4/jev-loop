"""Historical 1-minute BTC/USD bars for replay, cached one file per UTC day.

Two sources, for two jobs:

- Coinbase (public candles, no key): where most BTC/USD actually trades.
  Its volume and moves are meaningful, so the replay state is built from it.
- Alpaca (v1beta3 crypto bars, paper keys): the venue this loop trades on.
  Outcomes are scored on its prices. Its feed is thin (18 BTC traded on
  2026-09-21 vs 14,151 on Coinbase) and it sat ~0.2% away from Coinbase,
  so the gap between the two is itself a feature.

Bars are {minute_epoch: (open, high, low, close, volume)}, keyed by the
minute's start. A bar keyed t covers [t, t+60) and is only known at t+60.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import time
from pathlib import Path

import requests

LOG_DIR = Path(os.environ.get("JEV_LOOP_HOME", str(Path.home() / ".jev-loop")))
HISTORY_DIR = LOG_DIR / "history"

ALPACA_BARS_URL = "https://data.alpaca.markets/v1beta3/crypto/us/bars"
COINBASE_CANDLES_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"

Bars = dict[int, tuple[float, float, float, float, float]]


def _iso(t: dt.datetime) -> str:
    return t.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fetch_alpaca_day(day: dt.date, session: requests.Session) -> Bars:
    start = dt.datetime.combine(day, dt.time(), tzinfo=dt.UTC)
    end = start + dt.timedelta(days=1)
    bars: Bars = {}
    token = None
    while True:
        params = {
            "symbols": "BTC/USD",
            "timeframe": "1Min",
            "start": _iso(start),
            "end": _iso(end),
            "limit": 10000,
        }
        if token:
            params["page_token"] = token
        resp = session.get(ALPACA_BARS_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        for b in data.get("bars", {}).get("BTC/USD", []):
            ts = int(dt.datetime.fromisoformat(b["t"].replace("Z", "+00:00")).timestamp())
            if start.timestamp() <= ts < end.timestamp():
                bars[ts] = (b["o"], b["h"], b["l"], b["c"], b["v"])
        token = data.get("next_page_token")
        if not token:
            return bars


def _fetch_coinbase_day(day: dt.date, session: requests.Session) -> Bars:
    start = dt.datetime.combine(day, dt.time(), tzinfo=dt.UTC)
    # Coinbase rejects an end in the future (400), so today stops at now
    now = dt.datetime.now(dt.UTC).replace(second=0, microsecond=0)
    end = min(start + dt.timedelta(days=1), now)
    bars: Bars = {}
    s = start
    while s < end:  # at most 300 candles per request
        e = min(s + dt.timedelta(minutes=300), end)
        resp = session.get(
            COINBASE_CANDLES_URL,
            params={"granularity": 60, "start": _iso(s), "end": _iso(e)},
            timeout=30,
        )
        resp.raise_for_status()
        for ts, low, high, opn, close, vol in resp.json():  # [time, low, high, open, close, volume]
            if start.timestamp() <= ts < end.timestamp():
                bars[int(ts)] = (opn, high, low, close, vol)
        s = e
        time.sleep(0.15)  # stay well under Coinbase's public rate limit
    return bars


def _session(source: str) -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "jev-loop"
    if source == "alpaca":
        # the skill's own .env, so research commands work without `source .env`
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
        s.headers["APCA-API-KEY-ID"] = os.environ["ALPACA_API_KEY"]
        s.headers["APCA-API-SECRET-KEY"] = os.environ["ALPACA_SECRET_KEY"]
    return s


FETCHERS = {"alpaca": _fetch_alpaca_day, "coinbase": _fetch_coinbase_day}


def load_bars(source: str, first_day: dt.date, last_day: dt.date, verbose: bool = True) -> Bars:
    """Bars for every day in [first_day, last_day], from cache where possible.
    Completed days are cached; today is always re-fetched and never cached."""
    if source not in FETCHERS:
        raise ValueError(f"unknown source {source!r}")
    folder = HISTORY_DIR / source
    folder.mkdir(parents=True, exist_ok=True)
    today = dt.datetime.now(dt.UTC).date()
    session = None
    out: Bars = {}
    day = first_day
    while day <= last_day:
        path = folder / f"{day.isoformat()}.json"
        if day < today and path.exists():
            day_bars = {int(k): tuple(v) for k, v in json.loads(path.read_text()).items()}
        else:
            session = session or _session(source)
            day_bars = FETCHERS[source](day, session)
            if verbose:
                print(f"  fetched {source} {day}: {len(day_bars)} bars")
            if day < today:
                path.write_text(json.dumps(day_bars))
        out.update(day_bars)
        day += dt.timedelta(days=1)
    return out


# ------------------------------------------------ hourly, multi-year ----

COINBASE_PRODUCT_CANDLES_URL = "https://api.exchange.coinbase.com/products/{product}/candles"


def load_hourly(product: str, start: dt.datetime, verbose: bool = True) -> Bars:
    """Hourly Coinbase candles for `product` (e.g. "BTC-USD") from `start`
    to the last completed hour, keyed by the hour's start (a bar keyed t
    closes at t+3600). Cached in one file per product; only hours after
    the cache's last bar are downloaded (300 candles per request)."""
    folder = HISTORY_DIR / "coinbase_1h"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{product}.json"
    bars: Bars = {}
    if path.exists():
        bars = {int(k): tuple(v) for k, v in json.loads(path.read_text()).items()}
    start_ts = int(start.timestamp()) // 3600 * 3600
    end_ts = int(dt.datetime.now(dt.UTC).timestamp()) // 3600 * 3600  # current hour is incomplete
    have = [k for k in bars if k >= start_ts]
    fetch_from = start_ts if not have or min(have) > start_ts + 3600 else max(have) + 3600
    if fetch_from < end_ts:
        session = _session("coinbase")
        url = COINBASE_PRODUCT_CANDLES_URL.format(product=product)
        s = fetch_from
        n_before = len(bars)
        while s < end_ts:
            e = min(s + 300 * 3600, end_ts)
            resp = session.get(url, params={"granularity": 3600, "start": _iso(dt.datetime.fromtimestamp(s, dt.UTC)),
                                            "end": _iso(dt.datetime.fromtimestamp(e, dt.UTC))}, timeout=30)
            if resp.status_code == 429:
                time.sleep(2)
                continue
            resp.raise_for_status()
            for ts, low, high, opn, close, vol in resp.json():
                if s <= ts < e:
                    bars[int(ts)] = (opn, high, low, close, vol)
            s = e
            time.sleep(0.15)
        path.write_text(json.dumps(bars))
        if verbose:
            print(f"  {product}: +{len(bars) - n_before} hourly bars (cache now {len(bars)})")
    return {k: v for k, v in bars.items() if k >= start_ts}


# ------------------------------------------ US stocks, regular hours ----

ALPACA_STOCK_BARS_URL = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"


def _is_regular_hours(ts: int, bar_seconds: int) -> bool:
    """Bar starts at or after 9:30 and ends by 16:00, New York time."""
    from zoneinfo import ZoneInfo

    t = dt.datetime.fromtimestamp(ts, dt.UTC).astimezone(ZoneInfo("America/New_York"))
    if t.weekday() >= 5:
        return False
    start_min = t.hour * 60 + t.minute
    return 9 * 60 + 30 <= start_min and start_min + bar_seconds // 60 <= 16 * 60


def load_stock_bars(symbol: str, start: dt.datetime, timeframe: str = "30Min", bar_seconds: int = 1800,
                    verbose: bool = True) -> Bars:
    """Regular-hours bars for a US stock from Alpaca's consolidated (SIP)
    feed, split- and dividend-adjusted (adjustment=all, so a dividend is not
    mistaken for a price drop). SIP carries all venues' volume; the free IEX
    feed carries a few percent and returned no bars for 2019. Keyed by bar
    start; cached per symbol and timeframe, extended incrementally."""
    folder = HISTORY_DIR / "alpaca_stocks"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{symbol}_{timeframe}.json"
    bars: Bars = {}
    if path.exists():
        bars = {int(k): tuple(v) for k, v in json.loads(path.read_text()).items()}
    start_ts = int(start.timestamp())
    # SIP data from the last 15 minutes needs a paid plan; stop short of it
    end = dt.datetime.now(dt.UTC) - dt.timedelta(minutes=20)
    have = [k for k in bars if k >= start_ts]
    fetch_from = start if not have or min(have) > start_ts + 7 * 86400 else dt.datetime.fromtimestamp(max(have) + bar_seconds, dt.UTC)
    if fetch_from < end:
        session = _session("alpaca")
        url = ALPACA_STOCK_BARS_URL.format(symbol=symbol)
        token, n_before = None, len(bars)
        while True:
            params = {"timeframe": timeframe, "start": _iso(fetch_from), "end": _iso(end), "feed": "sip",
                      "adjustment": "all", "limit": 10000}
            if token:
                params["page_token"] = token
            resp = session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            for b in data.get("bars") or []:
                ts = int(dt.datetime.fromisoformat(b["t"].replace("Z", "+00:00")).timestamp())
                if _is_regular_hours(ts, bar_seconds):
                    bars[ts] = (b["o"], b["h"], b["l"], b["c"], b["v"])
            token = data.get("next_page_token")
            if not token:
                break
        path.write_text(json.dumps(bars))
        if verbose:
            print(f"  {symbol} {timeframe}: +{len(bars) - n_before} regular-hours bars (cache now {len(bars)})")
    return {k: v for k, v in bars.items() if k >= start_ts}


# ------------------------------------------ forex (Dukascopy, free) ----

DUKASCOPY_URL = "https://datafeed.dukascopy.com/datafeed/{symbol}/{y}/{m0:02d}/{rest}"
FX_POINT = {"EURUSD": 1e5, "GBPUSD": 1e5}  # prices are stored as integers in points


def _parse_bi5(content: bytes, period_start: int, point: float) -> Bars:
    """Dukascopy candle file: LZMA-compressed 24-byte big-endian records of
    (seconds from period start, open, close, low, high as integer points,
    volume as float32). Zero-volume records are the closed market (weekends,
    holidays) and are dropped. Returned in this module's (o, h, l, c, v)."""
    import lzma
    import struct

    if not content:
        return {}
    raw = lzma.decompress(content)
    out: Bars = {}
    for i in range(len(raw) // 24):
        t, o, c, lo, hi, v = struct.unpack(">5if", raw[i * 24:(i + 1) * 24])
        if v > 0:
            out[period_start + t] = (o / point, hi / point, lo / point, c / point, float(v))
    return out


def _dukascopy_get(session: requests.Session, url: str) -> bytes:
    for attempt in range(6):
        try:
            resp = session.get(url, timeout=30)
        except (requests.ConnectionError, requests.Timeout):
            # Dukascopy sometimes drops or stalls a connection; back off and retry
            time.sleep(3 * (attempt + 1))
            continue
        if resp.status_code == 404:
            return b""  # not published yet (current period)
        if resp.status_code in (429, 503):
            time.sleep(2)
            continue
        resp.raise_for_status()
        time.sleep(0.1)
        return resp.content
    raise requests.ConnectionError(f"Dukascopy kept failing for {url}; rerun to resume (completed files are cached)")


def _fx_session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0 (jev-loop research)"
    return s


def load_fx_hourly(symbol: str, start: dt.datetime, verbose: bool = True) -> Bars:
    """Hourly BID candles from Dukascopy, one cached file per completed month.
    Keyed by the hour's start; only hours the market was open."""
    point = FX_POINT[symbol]
    folder = HISTORY_DIR / "dukascopy" / f"{symbol}_1h"
    folder.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.UTC)
    this_month = (now.year, now.month)
    out: Bars = {}
    session, fetched = None, 0
    y, m = start.year, start.month
    while (y, m) <= this_month:
        path = folder / f"{y:04d}-{m:02d}.json"
        month_start = int(dt.datetime(y, m, 1, tzinfo=dt.UTC).timestamp())
        if (y, m) != this_month and path.exists():
            bars = {int(k): tuple(v) for k, v in json.loads(path.read_text()).items()}
        else:
            session = session or _fx_session()
            url = DUKASCOPY_URL.format(symbol=symbol, y=y, m0=m - 1, rest="BID_candles_hour_1.bi5")
            bars = _parse_bi5(_dukascopy_get(session, url), month_start, point)
            fetched += 1
            if (y, m) != this_month:
                path.write_text(json.dumps(bars))
        out.update(bars)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    if verbose and fetched:
        print(f"  {symbol} 1h: fetched {fetched} month files (total {len(out)} open-market hours)")
    return {k: v for k, v in out.items() if k >= int(start.timestamp())}


def load_fx_minutes(symbol: str, first_day: dt.date, last_day: dt.date, verbose: bool = True) -> Bars:
    """1-minute BID candles from Dukascopy, one cached file per completed day."""
    point = FX_POINT[symbol]
    folder = HISTORY_DIR / "dukascopy" / f"{symbol}_1m"
    folder.mkdir(parents=True, exist_ok=True)
    today = dt.datetime.now(dt.UTC).date()
    out: Bars = {}
    session, fetched = None, 0
    day = first_day
    while day <= min(last_day, today - dt.timedelta(days=1)):  # today is not published yet; asking stalls
        path = folder / f"{day.isoformat()}.json"
        day_start = int(dt.datetime.combine(day, dt.time(), tzinfo=dt.UTC).timestamp())
        if day < today and path.exists():
            bars = {int(k): tuple(v) for k, v in json.loads(path.read_text()).items()}
        else:
            session = session or _fx_session()
            url = DUKASCOPY_URL.format(symbol=symbol, y=day.year, m0=day.month - 1,
                                       rest=f"{day.day:02d}/BID_candles_min_1.bi5")
            bars = _parse_bi5(_dukascopy_get(session, url), day_start, point)
            fetched += 1
            if day < today:
                path.write_text(json.dumps(bars))
        out.update(bars)
        day += dt.timedelta(days=1)
    if verbose and fetched:
        print(f"  {symbol} 1m: fetched {fetched} day files (total {len(out)} open-market minutes)")
    return out
