"""Forex (EURUSD) support: Dukascopy candle parsing, per-market question
thresholds (BTC's wording must not change), and a state without a venue gap."""

import lzma
import struct

from jevloop.battery import build_questions_v2
from jevloop.history import _parse_bi5
from jevloop.state_v2 import build_state_v2, label_outcome

T0 = 1_788_000_000 // 86400 * 86400


def _bi5(records):
    raw = b"".join(struct.pack(">5if", *r) for r in records)
    return lzma.compress(raw, format=lzma.FORMAT_ALONE)


def test_parse_bi5_reorders_fields_scales_points_and_drops_closed_market():
    # file order is (t, open, close, low, high, volume)
    content = _bi5([
        (0, 108000, 108010, 107990, 108020, 12.5),
        (60, 108010, 108010, 108010, 108010, 0.0),  # closed market filler
    ])
    bars = _parse_bi5(content, T0, 1e5)
    assert bars == {T0: (1.08, 1.0802, 1.0799, 1.0801, 12.5)}  # (o, h, l, c, v)
    assert _parse_bi5(b"", T0, 1e5) == {}


def test_btc_question_wording_is_unchanged_by_the_scaling():
    q = build_questions_v2()
    assert q["direction_1h"]["criteria"]["up"] == "more than 0.25% above the current price"
    assert "rises 0.5% above" in q["first_touch_1h"]["criteria"]["up_first"]
    assert "at least 0.5% away" in q["big_move_1h"]["instructions"]


def test_eurusd_questions_use_scaled_thresholds():
    q = build_questions_v2(0.1, 0.05)
    assert q["direction_1h"]["criteria"]["flat"] == "within 0.05% of the current price"
    assert "rises 0.1% above" in q["first_touch_1h"]["criteria"]["up_first"]
    assert "at least 0.1% away" in q["big_move_1h"]["instructions"]


def _fx_minutes(n, start, price=1.1):
    return {start + 60 * i: (price, price * 1.00002, price * 0.99998, price, 5.0) for i in range(n)}


def test_fx_state_has_no_venue_gap_and_its_own_description():
    bars = _fx_minutes(1500, T0)
    s = build_state_v2(bars, bars, T0 + 1500 * 60, desc="EUR/USD spot.", venue_gap=False)
    assert s["market"] == "EUR/USD spot."
    assert "alpaca_vs_coinbase_price_pct" not in s


def test_fx_outcome_uses_scaled_threshold():
    t = T0 + 60
    bars = _fx_minutes(70, T0, price=1.0)
    bars[t + 600] = (1.0, 1.0012, 1.0, 1.0011, 5.0)  # +0.12%: a 0.1% move, not a 0.5% one
    assert label_outcome(bars, t, move_pct=0.1, flat_pct=0.05)["first_touch_1h"] == "up_first"
    assert label_outcome(bars, t)["first_touch_1h"] == "neither"  # BTC's 0.5% default


def test_v3_daily_path_bridges_weekends_for_fx():
    from jevloop.state_v2 import build_state_v3

    # 40 days of minutes (to Aug 9) with Saturday/Sunday removed
    import datetime as dt

    start = int(dt.datetime(2026, 7, 1, tzinfo=dt.UTC).timestamp())
    bars = {k: v for k, v in _fx_minutes(40 * 1440, start).items()
            if dt.datetime.fromtimestamp(k, dt.UTC).weekday() < 5}
    t = int(dt.datetime(2026, 8, 1, 12, tzinfo=dt.UTC).timestamp())
    while dt.datetime.fromtimestamp(t, dt.UTC).weekday() in (0, 5, 6):  # a mid-week t with a full 24h behind it
        t += 86400
    assert build_state_v3(bars, bars, t, venue_gap=False) is None  # crypto tolerance: weekend anchors missing
    s = build_state_v3(bars, bars, t, venue_gap=False, path_tolerance_bars=3 * 1440)
    assert len(s["returns_daily_last_30d_pct_oldest_first"]) == 30
