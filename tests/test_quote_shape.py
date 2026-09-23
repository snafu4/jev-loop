"""Quote shape and the directional leg, after the third paper run showed:

- quotes sat ~$1.29 either side of mid, well inside BTC's ~$29 spread,
- QUOTE_WIDE priced exactly like QUOTE_BOTH_SIDES, WIDEN placed nothing,
  and Jev's inventory skew never reached a price,
- 118 directional legs were rejected as "potential wash trade" (a market
  order hitting our own resting quote), and ~20 as "insufficient balance"
  (a $-sized sell a hair above what was held).
"""

import pytest
from test_execution_safety import MID, FakeAlpaca, _execute

from jevloop import loop
from jevloop.policy import QUOTE_BOTH_SIDES, QUOTE_WIDE, WIDEN, Action
from jevloop.state import InventoryState
from jevloop.strategy import THRESHOLDS

BOOK_BID, BOOK_ASK = MID - 15, MID + 15  # a real ~$30 spread
AS_BID, AS_ASK = MID - 1.29, MID + 1.29  # A-S quotes, far inside it


def _shape(kind, skew=0.0):
    return loop._shape_quotes(kind, skew, AS_BID, AS_ASK, BOOK_BID, BOOK_ASK)


# -- shape -------------------------------------------------------------------


def test_both_sides_joins_the_touch_instead_of_quoting_inside_it():
    bid, ask = _shape(QUOTE_BOTH_SIDES)
    assert bid == pytest.approx(BOOK_BID)
    assert ask == pytest.approx(BOOK_ASK)


def test_as_spread_wins_when_it_is_already_wider_than_the_book():
    bid, ask = loop._shape_quotes(
        QUOTE_BOTH_SIDES, 0.0, MID - 50, MID + 50, BOOK_BID, BOOK_ASK
    )
    assert (bid, ask) == (MID - 50, MID + 50)


def test_wide_and_widen_really_are_wider():
    both = _shape(QUOTE_BOTH_SIDES)
    wide = _shape(QUOTE_WIDE)
    widen = _shape(WIDEN)
    width = lambda q: q[1] - q[0]  # noqa: E731
    assert width(wide) == pytest.approx(width(both) * THRESHOLDS.quote_wide_spread_mult)
    assert width(widen) == pytest.approx(width(both) * THRESHOLDS.widen_spread_mult)
    assert width(both) < width(wide) < width(widen)


def test_skew_shifts_both_quotes():
    flat_bid, flat_ask = _shape(QUOTE_BOTH_SIDES)
    long_bid, long_ask = _shape(QUOTE_BOTH_SIDES, skew=-1.0)  # long, cut it
    shift = 15 * THRESHOLDS.skew_price_weight
    assert long_bid == pytest.approx(flat_bid - shift)
    assert long_ask == pytest.approx(flat_ask - shift)  # ask closer to mid


def test_no_book_falls_back_to_as_quotes():
    assert loop._shape_quotes(QUOTE_BOTH_SIDES, 0.0, AS_BID, AS_ASK, None, None) == (
        pytest.approx(AS_BID),
        pytest.approx(AS_ASK),
    )


# -- WIDEN now places quotes -------------------------------------------------


def test_widen_places_quotes_and_replaces_narrower_ones_at_once():
    fake = FakeAlpaca()
    inv = InventoryState(inventory=0.0003)  # ~$26: both sides backed
    # ask only: a resting bid would count as filled ($26 + $20 + $20 > $50)
    resting = {"ask": MID + 20, "kind": QUOTE_WIDE}
    _, fill_txt, _, _, new_resting, _ = _execute(
        fake, inv, Action(WIDEN, "t"), resting=resting, rest_counter=0
    )
    assert fake.calls[0] == "cancel_all"  # shape changed: no waiting 3 ticks
    assert {o[0] for o in fake.limit_orders} == {"buy", "sell"}
    assert new_resting["kind"] == WIDEN
    assert "quoted bid/ask" in fill_txt


def test_same_shape_still_rests_between_refreshes():
    fake = FakeAlpaca()
    inv = InventoryState(inventory=0.0003)
    resting = {"bid": MID - 40, "ask": MID + 40, "kind": QUOTE_WIDE}
    _execute(fake, inv, Action(QUOTE_WIDE, "t"), resting=resting, rest_counter=0)
    assert fake.calls == []


# -- directional leg -----------------------------------------------------------


def test_leg_pulls_our_resting_quotes_before_the_market_order():
    fake = FakeAlpaca()
    inv = InventoryState(inventory=0.0003)
    resting = {"bid": MID - 40, "ask": MID + 40, "kind": QUOTE_WIDE}
    _execute(
        fake,
        inv,
        Action(QUOTE_WIDE, "t", direction_leg="down"),
        resting=resting,
        rest_counter=0,
    )
    first_market = fake.calls.index("market_sell")
    assert "cancel_all" in fake.calls[:first_market]
    # and quotes come back after the leg, not before it
    assert any(c.startswith("limit_") for c in fake.calls[first_market:])


def test_leg_with_nothing_resting_does_not_cancel():
    fake = FakeAlpaca()
    inv = InventoryState(inventory=0.0003)
    _execute(fake, inv, Action(QUOTE_WIDE, "t", direction_leg="down"))
    assert fake.calls[0] == "market_sell"


def test_sell_leg_sized_to_exactly_what_is_held():
    # The ~20 "insufficient balance" rejections: $20 at the bid came out a
    # hair above the BTC actually held.
    fake = FakeAlpaca()
    inv = InventoryState(inventory=0.00023)  # ~$19.8, over the $10 floor
    _execute(fake, inv, Action(QUOTE_WIDE, "t", direction_leg="down"))
    assert fake.market_orders == [("sell", 0.00023)]


def test_sellable_qty_leaves_dust_and_full_size_alone():
    spec = loop.resolve_symbol("BTC/USD")
    assert loop._sellable_qty(0.00023, 0.00005, MID, spec) == 0.00023  # dust: no
    assert loop._sellable_qty(0.00023, 0.001, MID, spec) == 0.00023  # plenty held
    assert loop._sellable_qty(0.00023, 0.0002, MID, spec) == 0.0002  # sell held
