"""Replay must not look ahead, must not leak dates or price levels to Jev,
must label outcomes correctly, keep the holdout out, and resume cleanly."""

import datetime as dt
import json
import math
import types

import pytest

from jevloop import replay, state_v2
from jevloop.battery import build_questions_v2
from jevloop.state_v2 import build_state_v2, build_state_v3, label_outcome

T0 = int(dt.datetime(2026, 9, 1, tzinfo=dt.UTC).timestamp())
DAY = 86400


def _bars(start, n, price=80_000.0, step=0.0, vol=1.0):
    """n one-minute bars from start, a gentle wave so ranges are non-zero."""
    out = {}
    for i in range(n):
        p = price * (1 + step * i) * (1 + 0.001 * math.sin(i / 7))
        out[start + 60 * i] = (p, p * 1.0002, p * 0.9998, p, vol)
    return out


# -- state v2 -----------------------------------------------------------------


def test_state_uses_only_bars_before_t():
    cb = _bars(T0, 3000)
    alp = _bars(T0, 3000, price=79_900.0)
    t = T0 + DAY + 3600
    before = build_state_v2(cb, alp, t)
    # rewriting everything from t onward must not change the state
    for k in range(t, T0 + 3000 * 60, 60):
        cb[k] = (1.0, 1.0, 1.0, 1.0, 999.0)
        alp[k] = (1.0, 1.0, 1.0, 1.0, 999.0)
    assert build_state_v2(cb, alp, t) == before
    # but the bar that closed at t does
    cb[t - 60] = (90_000.0, 90_000.0, 90_000.0, 90_000.0, 1.0)
    assert build_state_v2(cb, alp, t) != before


def test_state_carries_no_timestamps_or_price_levels():
    cb = _bars(T0, 3000)
    alp = _bars(T0, 3000, price=79_900.0)
    state = build_state_v2(cb, alp, T0 + DAY + 3600)
    numbers = [v for v in state.values() if isinstance(v, (int, float))]
    assert all(abs(v) < 1000 for v in numbers)  # percents and ratios only
    assert not any("time" in k or "date" in k or k in ("price", "mid") for k in state)


def test_state_is_none_without_24h_of_history():
    cb = _bars(T0, 600)
    assert build_state_v2(cb, cb, T0 + 600 * 60) is None


def test_alpaca_gap_feature():
    cb = _bars(T0, 3000)
    alp = {k: tuple(x * 0.998 for x in v[:4]) + (v[4],) for k, v in cb.items()}
    state = build_state_v2(cb, alp, T0 + DAY + 3600)
    assert state["alpaca_vs_coinbase_price_pct"] == pytest.approx(-0.2, abs=1e-3)


# -- outcomes -----------------------------------------------------------------


def _flat(start, n, p=100.0):
    return {start + 60 * i: (p, p, p, p, 1.0) for i in range(n)}


def test_first_touch_up_down_neither_ambiguous():
    t = T0 + 60
    base = _flat(T0, 70)
    up = dict(base); up[t + 600] = (100, 100.6, 100, 100.5, 1)
    assert label_outcome(up, t)["first_touch_1h"] == "up_first"
    down = dict(base); down[t + 600] = (100, 100, 99.4, 99.5, 1)
    assert label_outcome(down, t)["first_touch_1h"] == "down_first"
    assert label_outcome(base, t)["first_touch_1h"] == "neither"
    both = dict(base); both[t + 600] = (100, 100.6, 99.4, 100, 1)
    assert label_outcome(both, t)["first_touch_1h"] == "ambiguous"


def test_direction_and_big_move():
    t = T0 + 60
    bars = _flat(T0, 70)
    bars[t + 59 * 60] = (100, 100.3, 100, 100.3, 1)  # last bar closes +0.3%
    o = label_outcome(bars, t)
    assert o["direction_1h"] == "up" and not o["big_move_1h"]


def test_a_few_missing_minutes_carry_the_price_forward():
    # Alpaca omits minutes with no activity (185 hours were lost to this,
    # mostly weekends, before gaps were filled)
    bars = _flat(T0, 70)
    for k in (10, 11, 30):
        del bars[T0 + 60 * k]
    o = label_outcome(bars, T0 + 60)
    assert o is not None and o["first_touch_1h"] == "neither"


def test_too_many_missing_minutes_give_no_outcome():
    bars = _flat(T0, 70)
    for k in range(10, 20):  # 10 of 60 > MAX_MISSING_OUTCOME_BARS
        del bars[T0 + 60 * k]
    assert label_outcome(bars, T0 + 60) is None


def test_entry_price_looks_back_a_few_minutes():
    bars = _flat(T0, 70)
    del bars[T0]  # no bar in the minute just before t = T0 + 60
    assert label_outcome(bars, T0 + 60) is None  # nothing within the lookback
    bars = _flat(T0, 70)
    del bars[T0 + 120]
    assert label_outcome(bars, T0 + 180) is not None  # uses T0 + 60's close


# -- questions, sampling, scoring ----------------------------------------------


def test_v2_questions_pass_the_split_guard_and_define_every_option():
    q = build_questions_v2()  # raises SplitViolation if off the allow-list
    for spec in q.values():
        if spec["type"] == "choice":
            assert all(desc for desc in spec["criteria"].values())


def test_holdout_days_are_not_sampled():
    first, last = dt.date(2026, 8, 1), dt.date(2026, 8, 30)
    cut = int(dt.datetime(2026, 8, 24, tzinfo=dt.UTC).timestamp())
    kept = replay.sample_times(first, last, 60, cut, include_holdout=False)
    assert kept and max(kept) < cut
    assert max(replay.sample_times(first, last, 60, cut, include_holdout=True)) >= cut


def test_auc():
    assert replay.auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert replay.auc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == 0.5


def test_replay_writes_samples_and_resumes_without_repeating(tmp_path, monkeypatch):
    today = dt.datetime.now(dt.UTC).date()
    start = int(dt.datetime.combine(today - dt.timedelta(days=4), dt.time(), tzinfo=dt.UTC).timestamp())
    cb = _bars(start, 5 * 1440)
    alp = _bars(start, 5 * 1440, price=79_900.0)
    monkeypatch.setattr(replay, "REPLAY_DIR", tmp_path)
    monkeypatch.setattr(replay, "load_bars", lambda source, a, b, verbose=True: cb if source == "coinbase" else alp)
    args = types.SimpleNamespace(days=4, every=180, holdout_days=1, include_holdout=False,
                                 variant="t", max_calls=3, mock=True, state="v2", same_times_as=None)
    path = replay.run_replay(args)
    first = [json.loads(l)["t"] for l in path.read_text().splitlines()]
    assert len(first) == 3
    args.max_calls = None
    replay.run_replay(args)
    all_t = [json.loads(l)["t"] for l in path.read_text().splitlines()]
    assert len(all_t) == len(set(all_t))  # nothing repeated
    assert all_t[:3] == first
    assert replay.score(path, holdout_start=None)  # scoring runs end to end


# -- state v3: the price path ------------------------------------------------------


def test_v3_adds_the_price_path_oldest_first():
    cb = _bars(T0, 32 * 1440, step=0.00001)  # slow steady rise
    alp = _bars(T0, 32 * 1440, price=79_900.0)
    t = T0 + 31 * DAY
    s = build_state_v3(cb, alp, t)
    assert len(s["returns_15m_last_24h_pct_oldest_first"]) == 96
    assert len(s["returns_daily_last_30d_pct_oldest_first"]) == 30
    assert s["returns_daily_last_30d_pct_oldest_first"][-1] > 0  # rising
    assert all(abs(v) < 100 for v in s["returns_15m_last_24h_pct_oldest_first"])  # percents, not prices
    # every v2 field is still there
    assert set(build_state_v2(cb, alp, t)) <= set(s)


def test_v3_does_not_look_ahead():
    cb = _bars(T0, 33 * 1440)
    alp = _bars(T0, 33 * 1440, price=79_900.0)
    t = T0 + 31 * DAY
    before = build_state_v3(cb, alp, t)
    for k in range(t, T0 + 33 * 1440 * 60, 60):
        cb[k] = (1.0, 1.0, 1.0, 1.0, 999.0)
    assert build_state_v3(cb, alp, t) == before


def test_v3_needs_30_days_of_history():
    cb = _bars(T0, 10 * 1440)
    assert build_state_v3(cb, cb, T0 + 5 * DAY) is None


def test_same_times_as_reuses_another_variants_decision_points(tmp_path, monkeypatch):
    today = dt.datetime.now(dt.UTC).date()
    start = int(dt.datetime.combine(today - dt.timedelta(days=4), dt.time(), tzinfo=dt.UTC).timestamp())
    cb = _bars(start, 5 * 1440)
    alp = _bars(start, 5 * 1440, price=79_900.0)
    monkeypatch.setattr(replay, "REPLAY_DIR", tmp_path)
    monkeypatch.setattr(replay, "load_bars", lambda source, a, b, verbose=True: cb if source == "coinbase" else alp)
    ref_times = [start + DAY + 3600 * k for k in (1, 5, 9)]
    (tmp_path / "ref.jsonl").write_text("".join(json.dumps({"t": x}) + "\n" for x in ref_times))
    args = types.SimpleNamespace(days=4, every=60, holdout_days=1, include_holdout=False,
                                 variant="same", max_calls=None, mock=True, state="v2", same_times_as="ref")
    path = replay.run_replay(args)
    assert sorted(json.loads(l)["t"] for l in path.read_text().splitlines()) == ref_times
