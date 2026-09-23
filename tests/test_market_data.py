"""The snapshot must be built from the real tape: real timestamps, real
prices, the latest trades rather than the first ones of the day."""

from datetime import datetime, timezone

import pytest

from jevloop import loop
from jevloop.assets import resolve_symbol
from jevloop.execution.alpaca import AlpacaPaperClient

BTC = resolve_symbol("BTC/USD")


def _epoch(y, mo, d, h, mi, s, us=0):
    return datetime(y, mo, d, h, mi, s, us, tzinfo=timezone.utc).timestamp()


# -- timestamps ------------------------------------------------------------


def test_parse_ts_nanoseconds_truncated_to_micro():
    assert loop._parse_ts("2026-09-22T21:44:44.638490131Z") == pytest.approx(
        _epoch(2026, 9, 22, 21, 44, 44, 638490)
    )


def test_parse_ts_without_fraction_and_with_offset():
    assert loop._parse_ts("2026-09-22T21:44:44Z") == _epoch(2026, 9, 22, 21, 44, 44)
    assert loop._parse_ts("2026-09-22T21:44:44.5+00:00") == pytest.approx(
        _epoch(2026, 9, 22, 21, 44, 44, 500000)
    )


def test_parse_ts_garbage_is_none():
    assert loop._parse_ts("") is None
    assert loop._parse_ts("not a time") is None


# -- the tape --------------------------------------------------------------


def test_parse_trades_keeps_real_prices_times_and_sides():
    now = _epoch(2026, 9, 22, 22, 0, 0)
    raw = [
        {"t": "2026-09-22T21:59:00Z", "p": 86200.0, "s": 0.01, "tks": "B"},
        {"t": "2026-09-22T21:58:00Z", "p": 86100.0, "s": 0.02, "tks": "S"},
    ]
    tape = loop._parse_trades(raw, now)
    assert [x[1] for x in tape] == [86100.0, 86200.0]  # oldest first
    assert [x[3] for x in tape] == ["sell", "buy"]
    assert tape[0][0] == _epoch(2026, 9, 22, 21, 58, 0)


def test_parse_trades_drops_future_prints_and_bad_rows():
    now = _epoch(2026, 9, 22, 22, 0, 0)
    raw = [
        {"t": "2026-09-22T22:00:05Z", "p": 1.0, "s": 1.0, "tks": "B"},  # future
        {"t": "", "p": 86000.0, "s": 1.0},  # no timestamp
        {"t": "2026-09-22T21:59:00Z", "p": 0, "s": 1.0},  # no price
        {"t": "2026-09-22T21:59:00Z", "p": 86000.0, "s": 1.0},  # equity: no tks
    ]
    tape = loop._parse_trades(raw, now)
    assert len(tape) == 1
    assert tape[0][3] is None  # unknown side, never counted as a buy or sell


def test_recent_trades_asks_for_a_recent_window_newest_first():
    client = AlpacaPaperClient(api_key="x", secret_key="y", spec=BTC)
    seen = {}

    def fake_request(method, url, **kwargs):
        seen.update(kwargs["params"])
        return {"trades": {"BTC/USD": [{"t": "b"}, {"t": "a"}]}}

    client._request = fake_request
    trades = client.get_recent_trades(since_s=2100)
    assert seen["sort"] == "desc"
    assert "start" in seen  # without it, Alpaca starts at midnight UTC
    start = datetime.strptime(seen["start"], "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    age = datetime.now(timezone.utc).timestamp() - start.timestamp()
    assert 2090 <= age <= 2110
    assert [t["t"] for t in trades] == ["a", "b"]  # returned oldest first


# -- microprice ------------------------------------------------------------


def test_microprice_leans_toward_the_thin_side():
    # Heavy bid, thin ask: the ask is likely to go next, so price leans up.
    mp = loop._microprice([(100.0, 9.0)], [(101.0, 1.0)], 100.5)
    assert mp == pytest.approx(100.9)


def test_microprice_falls_back_to_mid_without_a_book():
    assert loop._microprice([], [(101.0, 1.0)], 100.5) == 100.5
    assert loop._microprice([(100.0, 0.0)], [(101.0, 0.0)], 100.5) == 100.5
