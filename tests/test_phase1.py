"""Phase 1 must not look ahead: indicators use only closed bars, labels
line up with the horizon, and walk-forward training never includes a row
whose outcome was unknown when the test month began."""

import numpy as np
import pandas as pd
import pytest

from jevloop import phase1
from jevloop.indicators import bars_to_frame, build_features, forward_log_return

H = 3600
T0 = 1_600_000_000 // H * H


def _bars(n, seed=0, price=100.0):
    rng = np.random.default_rng(seed)
    closes = price * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    return {T0 + i * H: (c, c * 1.002, c * 0.998, c, float(rng.uniform(1, 10))) for i, c in enumerate(closes)}


def test_frame_is_indexed_by_close_time_and_fills_gaps():
    bars = _bars(10)
    del bars[T0 + 5 * H]
    df = bars_to_frame(bars)
    assert df.index[0] == pd.Timestamp(T0 + H, unit="s", tz="UTC")  # bar keyed T0 closes at T0+1h
    assert len(df) == 10
    gap = df.loc[pd.Timestamp(T0 + 6 * H, unit="s", tz="UTC")]
    assert gap["volume"] == 0 and gap["close"] == df["close"].iloc[4]


def test_indicators_do_not_look_ahead():
    btc, eth = _bars(1500), _bars(1500, seed=1)
    full = build_features(bars_to_frame(btc), bars_to_frame(eth))
    cut = 1200
    for k in list(btc)[cut:]:
        btc[k] = (1.0, 1.0, 1.0, 1.0, 1.0)
        eth[k] = (1.0, 1.0, 1.0, 1.0, 1.0)
    changed = build_features(bars_to_frame(btc), bars_to_frame(eth))
    upto = full.index[cut - 1]  # last row built only from untouched bars
    pd.testing.assert_frame_equal(full.loc[:upto], changed.loc[:upto])


def test_label_is_the_return_over_the_next_horizon():
    df = bars_to_frame(_bars(50))
    fwd = forward_log_return(df, 4)
    assert fwd.iloc[10] == pytest.approx(np.log(df["close"].iloc[14] / df["close"].iloc[10]))
    assert fwd.iloc[-4:].isna().all()


def test_walk_forward_trains_only_on_resolved_outcomes(monkeypatch):
    n = 24 * 500
    idx = pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(3)
    data = pd.DataFrame({"x": rng.normal(size=n), "ret_24h": rng.normal(size=n)}, index=idx)
    data["fwd"] = rng.normal(size=n)
    data["up"] = (data["fwd"] > 0).astype(int)

    seen = []

    class Spy:
        def __init__(self, inner): self.inner = inner
        def fit(self, X, y):
            seen.append(len(X)); self.inner.fit(X, y); return self
        def predict_proba(self, X): return self.inner.predict_proba(X)

    real = phase1._models
    monkeypatch.setattr(phase1, "_models", lambda f, m: {k: (c, (lambda mk=mk: Spy(mk()))) for k, (c, mk) in real(f, m).items()})
    horizon = 24
    oos = phase1.walk_forward(data, ["x", "ret_24h"], horizon, idx[-1], first_train_days=200)
    # every test month's model saw only rows at least `horizon` hours before it
    for ms in pd.date_range(idx[0] + pd.Timedelta(days=200), idx[-1], freq="MS", tz="UTC"):
        expected = int((data.index <= ms - pd.Timedelta(hours=horizon)).sum())
        assert expected in seen
    # and no test row's outcome runs past the end
    assert oos.index.max() <= idx[-1] - pd.Timedelta(hours=horizon)
    assert set(["y", "ret", "base rate so far"]).issubset(oos.columns)
