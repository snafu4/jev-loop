"""strategy.py -- the file you edit to change how this loop trades.

This is the harness the video talks about, not the edge. Everything it
shipped with is a generic strategy pulled out of thin air, wired in only
so the demo has something to trade. Real strategies are hard to build
properly; this file is where yours goes.

Two things live here:

1. `StrategyThresholds`, one number per action in `compose_action()`
   (policy.py): when to pull quotes, when to widen, when the quote
   environment is good enough to quote both sides or just quote wide, how
   much inventory pressure skews sizing, and how confident Jev has to be
   about direction before a directional leg is taken. Change a number,
   restart the loop, see different behaviour on the next tick.

2. `apply_strategy()`, a hook called once per tick with the action
   `compose_action()` already produced from the thresholds above. Return
   it unchanged (the default) and nothing changes from what shipped in
   the video. Return a different action to override it, or an action
   with `kind=STAND_DOWN` to veto the tick outright. This is the one
   function a real strategy plugs into.

Shipped default: every decision threshold below matches what the video
ran, and `apply_strategy()` is a no-op. The quote-shape settings at the
bottom were added after the first paper runs: the video's quotes sat a
fixed ~$1.29 either side of mid, well inside BTC's real ~$29 spread, and
QUOTE_WIDE / WIDEN / the inventory skew did not change prices at all.

The hard risk caps (max position, max daily loss, max drawdown, and so
on) do NOT live here: they live in `jevloop/limits.py`, checked in
`risk.py` before every order, and this file cannot raise them. A
strategy can make the loop more conservative than the risk engine
allows; it can never make it less conservative than the risk engine
allows.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StrategyThresholds:
    """One threshold per action in compose_action(). These are exactly the
    numbers the video shipped with."""

    toxic_flow_pull_threshold: float = 0.6  # ans["toxic_flow"] > this -> PULL_QUOTES
    liquidity_stressed_widen_threshold: float = (
        0.7  # ans["liquidity_stressed"] > this -> WIDEN
    )
    quote_env_full_score: float = 2.0  # env >= this and confident -> quote both sides
    quote_env_full_confidence: float = 0.80
    quote_env_wide_score: float = 1.0  # env >= this (but below full) -> quote wide
    inventory_pressure_max_score: float = 3.0  # denominator for the skew calculation

    # The directional leg: a market order in Jev's called direction, bolted
    # on by the video so the demo shows fills. OFF by default: over ~7,900
    # real-data ticks (2026-09-22/23), Jev's "down" calls were right ~50% of
    # the time at 1, 5 and 15 minutes, and "up" calls below 50%, while each
    # leg pays the spread plus a 0.25% taker fee. Turn it back on only once
    # `jevloop calibrate` shows the direction call has signal.
    directional_leg_enabled: bool = False
    direction_confidence_threshold: float = (
        0.55  # when enabled: direction.confidence above this -> take the leg
    )

    # quote shape (loop._shape_quotes). The half-spread is never narrower
    # than the market's own: QUOTE_BOTH_SIDES joins the best bid/ask, and
    # the other actions sit further out by these multiples of it.
    quote_wide_spread_mult: float = 2.0  # QUOTE_WIDE: twice the half-spread
    widen_spread_mult: float = 3.0  # WIDEN: three times, for a stressed book
    # How far Jev's inventory-pressure skew moves both quotes, as a fraction
    # of the half-spread. Long + high pressure (skew -1) shifts both quotes
    # down by this much, so the ask is likelier to fill than the bid.
    skew_price_weight: float = 0.5


THRESHOLDS = StrategyThresholds()


def apply_strategy(action, answers: dict, snapshot: dict, limits) -> object:
    """The strategy hook. Called once per tick, after compose_action() has
    already turned Jev's answers into an action using THRESHOLDS above.

    Default: return the action unchanged. That is the whole strategy the
    video ran, a generic one, pulled out of thin air, applied only so the
    demo shows real fills.

    To plug in your own strategy, edit this function. You can:
      - inspect `answers` (the seven Jev judgments this tick) or
        `snapshot` (the deterministic state) and return a different
        Action than the one compose_action() chose
      - veto the tick outright by returning
        `Action(STAND_DOWN, reason="my strategy said no")`
      - leave `action` alone most of the time and only step in on
        specific conditions

    `risk.py` still runs after this and still has the final veto, so
    nothing returned here can bypass a hard limit in limits.py, only add
    more caution on top of it.
    """
    return action
