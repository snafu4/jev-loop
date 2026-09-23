"""calibrate.py must score Jev's own P(up), over a time horizon, against a
base-rate baseline. The original scored quote_environment's confidence as
if it were P(up), counted the horizon in ticks, and had no baseline."""

import pytest

from jevloop import calibrate


def _tick(ts, mid, p_up=None, call="neutral"):
    t = {"ts": ts, "mid": mid, "direction": call}
    if p_up is not None:
        t["direction_probs"] = {"up": p_up, "down": 1 - p_up, "neutral": 0.0}
    return t


def test_scores_direction_probabilities_not_quote_confidence():
    ticks = [
        {**_tick(0, 100.0, p_up=0.9, call="up"), "quote_environment_conf": 0.1},
        _tick(300, 101.0),
    ]
    pairs = calibrate.pair_predictions(ticks, horizon_s=300)
    assert pairs == [(0.9, 1, "up", pytest.approx(100.0))]


def test_ticks_without_probabilities_are_skipped_not_guessed():
    ticks = [_tick(0, 100.0), _tick(300, 101.0)]
    assert calibrate.pair_predictions(ticks, horizon_s=300) == []


def test_horizon_is_time_based_and_does_not_cross_run_gaps():
    # the next tick is an hour later (a different run): no outcome
    ticks = [_tick(0, 100.0, p_up=0.5), _tick(3600, 120.0)]
    assert calibrate.pair_predictions(ticks, horizon_s=300) == []


def test_skill_score_zero_for_base_rate_positive_for_signal():
    no_signal = [(0.5, 1, "up", 1.0), (0.5, 0, "down", -1.0)]
    assert calibrate.skill_score(no_signal) == pytest.approx(0.0)
    signal = [(0.9, 1, "up", 1.0), (0.1, 0, "down", -1.0)]
    assert calibrate.skill_score(signal) > 0.5


def test_hit_rates_per_call():
    pairs = [(0.8, 1, "up", 2.0), (0.8, 0, "up", -2.0), (0.2, 0, "down", -3.0)]
    rates = calibrate.hit_rates(pairs)
    assert rates["up"] == (2, 0.5)
    assert rates["down"] == (1, 1.0)
