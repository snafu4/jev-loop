"""Regression tests for the first live paper run, where:

1. a two-sided quote on a flat cash account placed the buy, had the sell
   rejected (no BTC to sell), left the buy untracked, and re-posted a new
   buy every tick, and
2. those quote fills never reached the loop's inventory, so risk.py's
   max_position_usd never saw a ~$220 position against a $50 cap.
"""

import pytest

from jevloop import loop
from jevloop.assets import resolve_symbol
from jevloop.execution.alpaca import AlpacaAPIError, AlpacaPaperClient
from jevloop.limits import Limits
from jevloop.policy import QUOTE_WIDE, Action
from jevloop.state import InventoryState

BTC = resolve_symbol("BTC/USD")
MID = 86_000.0


class FakeAlpaca:
    """Records every order call. No network."""

    def __init__(self, position_qty=0.0, sell_error=False):
        self.spec = BTC
        self.position_qty = position_qty
        self.sell_error = sell_error
        self.limit_orders = []
        self.market_orders = []
        self.cancel_all_calls = 0
        self.close_calls = 0
        self.calls = []  # every order-side call, in order

    def is_market_open(self):
        return True

    def get_orderbook(self):
        return {
            "b": [{"p": MID - 5, "s": 1.0}, {"p": MID - 10, "s": 1.0}],
            "a": [{"p": MID + 5, "s": 1.0}, {"p": MID + 10, "s": 1.0}],
        }

    latest_trade_calls = 0

    def get_latest_trade(self):
        self.latest_trade_calls += 1
        return {"p": MID}

    def get_recent_trades(self, since_s=2100.0, limit=1000):
        return [
            {"t": "2026-01-01T00:00:00.123456789Z", "p": MID, "s": 0.01, "tks": "B"}
        ] * 5

    def get_position(self):
        if not self.position_qty:
            return {}
        return {"qty": str(self.position_qty), "avg_entry_price": str(MID)}

    def submit_limit_order(self, side, qty, limit_price, tif="gtc"):
        if side == "sell" and self.sell_error:
            raise AlpacaAPIError(403, "insufficient balance")
        self.limit_orders.append((side, qty, limit_price))
        self.calls.append(f"limit_{side}")
        return {"id": str(len(self.limit_orders))}

    def submit_market_order(self, side, qty):
        self.market_orders.append((side, qty))
        self.calls.append(f"market_{side}")
        return {"id": "m"}

    def cancel_all_orders(self):
        self.cancel_all_calls += 1
        self.calls.append("cancel_all")

    def close_position(self):
        self.close_calls += 1
        self.position_qty = 0.0


def _snapshot(inventory=0.0, position_age_s=0.0):
    return dict(
        drawdown_pct=0.0,
        inventory=inventory,
        mid=MID,
        daily_loss_usd=0.0,
        position_age_s=position_age_s,
        data_age_s=0.0,
        leverage=1.0,
    )


def _execute(
    alpaca, inv, action, resting=None, rest_counter=0, limits=None, position_age_s=0.0
):
    limits = limits or Limits()
    return loop._execute_action(
        alpaca=alpaca,
        spec=BTC,
        action=action,
        bid_px=MID - 20,
        ask_px=MID + 20,
        mid=MID,
        quote_notional=limits.quote_notional_usd,
        directional_notional=limits.directional_notional_usd,
        snapshot=_snapshot(inv.inventory, position_age_s),
        limits=limits,
        inv=inv,
        api_error_streak=0,
        decision_latency_ms=100.0,
        resting_quotes=resting,
        rest_counter=rest_counter,
        now=0.0,
    )


# -- side gating ---------------------------------------------------------


def test_flat_cash_account_quotes_bid_only():
    fake = FakeAlpaca()
    inv = InventoryState()
    _, fill_txt, _, _, resting, _ = _execute(fake, inv, Action(QUOTE_WIDE, "t"))
    assert [o[0] for o in fake.limit_orders] == ["buy"]
    assert "bid" in resting and "ask" not in resting
    assert "bid" in fill_txt and "ask" not in fill_txt


def test_held_inventory_allows_the_ask():
    fake = FakeAlpaca()
    inv = InventoryState(inventory=0.0003)  # ~$26 held: covers a $20 ask
    _execute(fake, inv, Action(QUOTE_WIDE, "t"))
    assert "sell" in [o[0] for o in fake.limit_orders]


def test_bid_skipped_when_fill_would_breach_max_position():
    fake = FakeAlpaca()
    inv = InventoryState(inventory=0.0005)  # ~$43 held + $20 bid > $50 cap
    _execute(fake, inv, Action(QUOTE_WIDE, "t"))
    assert "buy" not in [o[0] for o in fake.limit_orders]


def test_resting_bid_counts_toward_the_cap_at_refresh():
    # $20 held + $20 bid still resting (its cancel may not land before it
    # fills) + $20 replacement = $60 > $50: the replacement bid is skipped.
    # This is the race that took the hour run to ~$55 at tick 128.
    fake = FakeAlpaca()
    inv = InventoryState(inventory=0.000233)  # ~$20
    _execute(
        fake, inv, Action(QUOTE_WIDE, "t"), resting={"bid": MID - 20}, rest_counter=2
    )
    assert "buy" not in [o[0] for o in fake.limit_orders]


def test_ask_sized_down_to_exactly_what_is_held():
    # Held a hair under the $20 ask size, but well over the $10 minimum.
    fake = FakeAlpaca()
    inv = InventoryState(inventory=0.00023)
    _execute(fake, inv, Action(QUOTE_WIDE, "t"))
    sells = [o for o in fake.limit_orders if o[0] == "sell"]
    assert sells and sells[0][1] == 0.00023


def test_dust_below_the_venue_minimum_is_not_offered():
    fake = FakeAlpaca()
    inv = InventoryState(inventory=0.00005)  # ~$4, under the $10 floor
    _execute(fake, inv, Action(QUOTE_WIDE, "t"))
    assert "sell" not in [o[0] for o in fake.limit_orders]


# -- the re-post bug -----------------------------------------------------


def test_failed_second_side_still_tracks_the_first():
    fake = FakeAlpaca(sell_error=True)
    inv = InventoryState(inventory=0.0003)  # held, so the ask is attempted
    line, _, _, _, resting, _ = _execute(fake, inv, Action(QUOTE_WIDE, "t"))
    assert "order error" in line
    assert "bid" in resting and "ask" not in resting  # the placed buy is not forgotten


def test_resting_bid_is_not_reposted_every_tick():
    fake = FakeAlpaca()
    inv = InventoryState()
    resting, counter = None, 0
    for _ in range(3):  # rest_ticks defaults to 3
        _, _, _, _, resting, counter = _execute(
            fake, inv, Action(QUOTE_WIDE, "t"), resting, counter
        )
    assert len(fake.limit_orders) == 1


# -- directional leg -----------------------------------------------------


def test_sell_leg_skipped_without_inventory():
    fake = FakeAlpaca()
    inv = InventoryState()
    _, fill_txt, _, _, _, _ = _execute(
        fake, inv, Action(QUOTE_WIDE, "t", direction_leg="down")
    )
    assert fake.market_orders == []
    assert "no shorting" in fill_txt


def test_buy_leg_skipped_when_it_would_breach_max_position():
    fake = FakeAlpaca()
    inv = InventoryState()  # flat: $20 bid + $20 leg = $40, ok
    _execute(fake, inv, Action(QUOTE_WIDE, "t", direction_leg="up"))
    assert fake.market_orders == [("buy", pytest.approx(20 / (MID + 20), rel=1e-3))]

    fake2 = FakeAlpaca()
    inv2 = InventoryState(inventory=0.0002)  # ~$17 + $20 bid + $20 leg > $50
    _, fill_txt, _, _, _, _ = _execute(
        fake2, inv2, Action(QUOTE_WIDE, "t", direction_leg="up")
    )
    assert fake2.market_orders == []
    assert "max_position_usd" in fill_txt


# -- inventory age: reduce-only, not a freeze ------------------------------

AGED = 99_999.0  # well past max_inventory_age_s


def test_aged_inventory_still_sells_but_never_buys():
    fake = FakeAlpaca()
    inv = InventoryState(inventory=0.00025)  # ~$21 held too long
    line, _, _, _, _, _ = _execute(
        fake, inv, Action(QUOTE_WIDE, "t"), position_age_s=AGED
    )
    assert [o[0] for o in fake.limit_orders] == ["sell"]
    assert "reduce-only" in line


def test_aged_inventory_skips_a_buy_leg_but_takes_a_sell_leg():
    fake = FakeAlpaca()
    inv = InventoryState(inventory=0.0005)
    _, fill_txt, _, _, _, _ = _execute(
        fake, inv, Action(QUOTE_WIDE, "t", direction_leg="up"), position_age_s=AGED
    )
    assert fake.market_orders == []
    assert "reduce-only" in fill_txt

    fake2 = FakeAlpaca()
    inv2 = InventoryState(inventory=0.0005)
    _execute(
        fake2, inv2, Action(QUOTE_WIDE, "t", direction_leg="down"), position_age_s=AGED
    )
    assert [o[0] for o in fake2.market_orders] == ["sell"]


def test_entering_reduce_only_pulls_a_resting_bid_immediately():
    fake = FakeAlpaca()
    inv = InventoryState(inventory=0.00025)
    _, _, _, _, resting, _ = _execute(
        fake,
        inv,
        Action(QUOTE_WIDE, "t"),
        resting={"bid": MID - 20},
        rest_counter=0,  # would normally rest another two ticks
        position_age_s=AGED,
    )
    assert fake.cancel_all_calls == 1
    assert "bid" not in (resting or {})


# -- reconciliation --------------------------------------------------------


def test_reconcile_treats_unsellable_dust_as_flat():
    # The 0.000000001 BTC that froze the second hour run: ~$0.0001.
    inv = InventoryState(position_opened_at=1.0)
    loop._reconcile_inventory(
        inv, {"qty": "0.000000001", "avg_entry_price": "86077"}, 5.0, MID, 10.0
    )
    assert inv.inventory == 0.0
    assert inv.position_opened_at is None


def test_latest_trade_is_only_fetched_without_a_book(wired):
    fake = wired(FakeAlpaca())
    loop.run("BTC/USD", ticks=3, mock=False, limits=Limits(tick_seconds=0.0))
    assert fake.latest_trade_calls == 0


def test_reconcile_takes_the_broker_position():
    inv = InventoryState()
    loop._reconcile_inventory(
        inv, {"qty": "0.0025", "avg_entry_price": "86000"}, 5.0, MID, 10.0
    )
    assert inv.inventory == 0.0025
    assert inv.entry_price == 86000.0
    assert inv.position_opened_at == 5.0


def test_reconcile_flat_clears_state():
    inv = InventoryState(inventory=0.1, entry_price=1.0, position_opened_at=1.0)
    loop._reconcile_inventory(inv, {}, 5.0, MID, 10.0)
    assert inv.inventory == 0.0
    assert inv.position_opened_at is None


def test_get_position_404_means_flat():
    client = AlpacaPaperClient(api_key="x", secret_key="y", spec=BTC)

    def _404(method, url, **kwargs):
        raise AlpacaAPIError(404, "position does not exist")

    client._request = _404
    assert client.get_position() == {}


def test_position_endpoint_uses_slashless_symbol():
    client = AlpacaPaperClient(api_key="x", secret_key="y", spec=BTC)
    seen = []
    client._request = lambda method, url, **kwargs: seen.append(url) or {}
    client.get_position()
    assert seen[0].endswith("/v2/positions/BTCUSD")


# -- the whole loop, against the fake broker ---------------------------------


class StubDecisionClient:
    name = "STUB"
    model = "stub"

    def ask(self, state, questions, timeout):
        score = lambda s: {  # noqa: E731
            "type": "score",
            "score": s,
            "legend": {},
            "probabilities": {},
            "confidence": 0.9,
        }
        choice = lambda c: {  # noqa: E731
            "type": "choice",
            "choice": c,
            "probabilities": {},
            "confidence": 0.9,
        }
        answers = {
            "regime": choice("mean_reverting"),
            "direction": choice("neutral"),
            "toxic_flow": {"type": "noul", "noul": 0.2},
            "liquidity_stressed": {"type": "noul", "noul": 0.1},
            "quote_environment": score(1.5),  # QUOTE_WIDE
            "inventory_pressure": score(0.0),
            "execution_health": score(2.8),
        }
        return answers, {"route": self.name, "model": self.model, "latency_ms": 10.0}


@pytest.fixture
def wired(monkeypatch, tmp_path):
    def _wire(fake):
        monkeypatch.setattr(loop, "LOG_DIR", tmp_path)
        monkeypatch.setattr(loop, "LOG_FILE", tmp_path / "log.jsonl")
        monkeypatch.setattr(loop, "LATEST_FILE", tmp_path / "latest.json")
        monkeypatch.setattr(loop, "client_from_env", lambda **kw: fake)
        monkeypatch.setattr(
            loop, "resolve_decision_client", lambda mock=False: StubDecisionClient()
        )
        return fake

    return _wire


def test_run_on_flat_account_never_sells_and_cancels_on_exit(wired):
    fake = wired(FakeAlpaca())
    rc = loop.run("BTC/USD", ticks=5, mock=False, limits=Limits(tick_seconds=0.0))
    assert rc == 0
    assert all(o[0] == "buy" for o in fake.limit_orders)
    assert len(fake.limit_orders) == 2  # tick 1, then the tick-4 refresh
    assert fake.cancel_all_calls >= 1  # the bounded run leaves nothing open


def test_locked_dashboard_file_does_not_crash_the_loop(wired, monkeypatch):
    # Windows: replacing latest.json fails while the dashboard has it open.
    fake = wired(FakeAlpaca())

    def locked(self, target):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(loop.Path, "replace", locked)
    rc = loop.run("BTC/USD", ticks=3, mock=False, limits=Limits(tick_seconds=0.0))
    assert rc == 0
    assert len(loop.LOG_FILE.read_text().strip().splitlines()) == 3


def test_unexpected_crash_still_cancels_open_orders(wired, monkeypatch):
    fake = wired(FakeAlpaca())

    def boom(*a, **kw):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(loop, "_append_log", boom)
    with pytest.raises(RuntimeError):
        loop.run("BTC/USD", ticks=3, mock=False, limits=Limits(tick_seconds=0.0))
    assert fake.cancel_all_calls >= 1


def test_rate_cap_from_limits_reaches_the_alpaca_client(monkeypatch, tmp_path):
    # limits.py's max_alpaca_calls_per_minute used to be printed but never
    # passed on: the client always ran on its own hard-coded 90.
    seen = {}

    def fake_client_from_env(**kwargs):
        seen.update(kwargs)
        raise loop.AlpacaConfigError("stop here")

    monkeypatch.setattr(loop, "LOG_DIR", tmp_path)
    monkeypatch.setattr(loop, "client_from_env", fake_client_from_env)
    loop.run("BTC/USD", ticks=1, mock=True, limits=Limits(max_alpaca_calls_per_minute=150))
    assert seen["calls_per_minute"] == 150


def test_client_from_env_builds_the_limiter_with_the_cap(monkeypatch):
    from jevloop.execution.alpaca import client_from_env

    monkeypatch.setenv("ALPACA_API_KEY", "x")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "y")
    monkeypatch.delenv("ALPACA_BASE_URL", raising=False)
    client = client_from_env(symbol="BTC/USD", calls_per_minute=150)
    assert client._limiter.calls_per_minute == 150


def test_dashboard_feed_carries_run_totals_not_window_counts(wired, monkeypatch):
    # latest.json keeps only the last LATEST_WINDOW ticks, so counters read
    # from it capped at 120 and a full hour showed as "120 decisions".
    import json as _json

    monkeypatch.setattr(loop, "LATEST_WINDOW", 5)
    wired(FakeAlpaca())
    loop.run("BTC/USD", ticks=12, mock=False, limits=Limits(tick_seconds=0.0))
    stats = _json.loads(loop.LATEST_FILE.read_text())["stats"]
    assert stats["calls"] == 5  # the window
    assert stats["ticks_total"] == 12
    assert stats["decisions_total"] == 12
    assert stats["late_total"] == 0


def test_run_kills_and_flattens_an_oversized_broker_position(wired):
    fake = wired(FakeAlpaca(position_qty=0.0025))  # ~$215 vs a $50 cap
    rc = loop.run("BTC/USD", ticks=10, mock=False, limits=Limits(tick_seconds=0.0))
    assert rc == 0
    assert fake.close_calls == 1
    assert fake.limit_orders == []  # no new orders once the breach is seen
    last = (loop.LOG_FILE).read_text().strip().splitlines()
    assert len(last) == 1  # stopped on the first tick
