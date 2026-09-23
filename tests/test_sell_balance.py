"""Sells sized from a stale position, as in the 2s hour run: Alpaca's
positions endpoint reported 0.0002901 BTC for ~25s after a resting ask had
filled, while the real balance was 0.000173205, so ten sells were rejected
with "insufficient balance" (available == balance, nothing reserved)."""

import json

from test_execution_safety import MID, FakeAlpaca, _execute

from jevloop import loop
from jevloop.execution.alpaca import AlpacaAPIError, insufficient_balance_available
from jevloop.policy import QUOTE_WIDE, Action
from jevloop.state import InventoryState

REAL = 0.000173205  # ~$14.9: sellable, over the $10 floor
STALE = 0.0002901  # what the positions endpoint still said
SENT = 0.0001732  # REAL floored to the 8-decimal order precision, never rounded up


def _reject(available, requested):
    body = json.dumps(
        {
            "available": str(available),
            "balance": str(available),
            "code": 40310000,
            "message": f"insufficient balance for BTC (requested: {requested}, available: {available})",
            "symbol": "USD",
        }
    )
    return AlpacaAPIError(403, body)


class BalanceFake(FakeAlpaca):
    """Rejects any sell above the real balance, exactly as Alpaca does."""

    def __init__(self, real_balance, **kw):
        super().__init__(**kw)
        self.real_balance = real_balance
        self.rejected_sells = 0

    def _check(self, qty):
        if qty > self.real_balance + 1e-12:
            self.rejected_sells += 1
            raise _reject(self.real_balance, qty)

    def submit_limit_order(self, side, qty, limit_price, tif="gtc"):
        if side == "sell":
            self._check(qty)
        return super().submit_limit_order(side, qty, limit_price, tif)

    def submit_market_order(self, side, qty):
        if side == "sell":
            self._check(qty)
        return super().submit_market_order(side, qty)


def _stale_inv():
    inv = InventoryState()
    loop._reconcile_inventory(inv, {"qty": str(STALE)}, 0.0, MID, 10.0)
    return inv


# -- parsing ---------------------------------------------------------------


def test_available_is_read_from_the_rejection():
    assert insufficient_balance_available(_reject(REAL, 0.00023)) == REAL


def test_other_errors_are_not_mistaken_for_a_balance():
    assert insufficient_balance_available(AlpacaAPIError(403, "potential wash trade")) is None
    assert insufficient_balance_available(AlpacaAPIError(500, "{}")) is None


def test_floor_never_rounds_a_sell_above_the_balance():
    spec = loop.resolve_symbol("BTC/USD")
    assert loop._floor_qty(REAL, spec) == SENT  # nearest would give 0.00017321
    assert loop._floor_qty(0.00023, spec) == 0.00023  # exact values survive


# -- quote sells -------------------------------------------------------------


def test_rejected_ask_is_resent_at_the_real_balance():
    fake = BalanceFake(REAL)
    inv = _stale_inv()
    line, fill_txt, _, _, resting, _ = _execute(fake, inv, Action(QUOTE_WIDE, "t"))
    sells = [o for o in fake.limit_orders if o[0] == "sell"]
    assert fake.rejected_sells == 1
    assert sells and sells[0][1] == SENT
    assert "ask" in resting
    assert "order error" not in line


def test_real_balance_below_the_minimum_skips_the_ask_quietly():
    fake = BalanceFake(0.00005)  # ~$4: unsellable
    inv = _stale_inv()
    line, fill_txt, _, _, resting, _ = _execute(fake, inv, Action(QUOTE_WIDE, "t"))
    assert [o[0] for o in fake.limit_orders if o[0] == "sell"] == []
    assert "ask" not in (resting or {})
    assert "order error" not in line


def test_the_balance_is_remembered_while_the_endpoint_stays_stale():
    fake = BalanceFake(REAL)
    inv = _stale_inv()
    _execute(fake, inv, Action(QUOTE_WIDE, "t"))
    assert inv.known_balance == REAL

    # next tick: the endpoint still says STALE, so the cap holds...
    loop._reconcile_inventory(inv, {"qty": str(STALE)}, 1.0, MID, 10.0)
    assert inv.inventory == REAL
    fake.rejected_sells = 0
    _execute(fake, inv, Action(QUOTE_WIDE, "t"))
    assert fake.rejected_sells == 0  # ...and no repeat rejection (was 12 ticks)

    # once it moves, the endpoint is trusted again
    loop._reconcile_inventory(inv, {"qty": "0.0001"}, 2.0, MID, 10.0)
    assert inv.known_balance is None
    assert inv.inventory == 0.0  # 0.0001 BTC ~ $8.6: under $10, so flat


# -- directional sells -------------------------------------------------------


def test_rejected_sell_leg_is_resent_at_the_real_balance():
    fake = BalanceFake(REAL)
    inv = _stale_inv()
    _, fill_txt, fill_qty, _, _, _ = _execute(
        fake, inv, Action(QUOTE_WIDE, "t", direction_leg="down")
    )
    assert fake.market_orders == [("sell", SENT)]
    assert fill_qty == SENT
    assert "rejected" not in fill_txt


def test_sell_leg_with_unsellable_real_balance_is_skipped_not_rejected():
    fake = BalanceFake(0.00005)
    inv = _stale_inv()
    _, fill_txt, _, _, _, _ = _execute(
        fake, inv, Action(QUOTE_WIDE, "t", direction_leg="down")
    )
    assert fake.market_orders == []
    assert "leg skipped" in fill_txt
