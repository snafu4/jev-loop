"""The nine-stage block loop.

    1. block event            (tick clock, "block" = one tick, default 2s)
    2. read the book          (Alpaca paper market data: L2 depth on crypto,
                               best bid/ask on equities, and recent trades)
    3. state snapshot         (deterministic, state.py)
    4. battery                (seven Jev judgments, one call, battery.py)
    5. policy engine          (compose_action, policy.py: code, not Jev)
    6. pricing                (Avellaneda-Stoikov, pricing.py: code)
    7. risk veto              (hard limits, risk.py: code, absolute veto)
    8. execute                (directional leg, then cancel-replace limit quotes)
    9. log, fills, inventory  (JSONL and latest.json for the dashboard)

Paper by default. `--mock` forces the mock decision client even if a real
key is present, useful for a clean demo run that can never fail on network
or billing grounds. `--dry-execution` reads real market data and fires a
real Jev battery but never submits an order to Alpaca; every fill line
reads "dry" instead. Useful for testing the full pipeline without touching
the paper account. `--ticks 0` or `--forever` runs continuously until
stopped (Ctrl+C, or a SIGTERM if it is running in the background), and
cancels any resting orders before it exits rather than leaving them
behind. `--live` is a separate, deliberately awkward opt-in documented in
execution/alpaca.py and SKILL.md; paper is what every default here
resolves to unless a caller goes out of its way to ask for live.

Any asset `jevloop/assets.py` resolves: a crypto pair runs 24/7; a US
equity ticker only trades while the market is open, and the loop holds
(never orders) while it is closed, per `execution/alpaca.py`'s market-hours
guard.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

from .assets import AssetSpec, UnknownSymbolError, resolve_symbol, size_order
from .battery import run_battery
from .client import (
    DecisionClientError,
    GatewayVerificationRequired,
    resolve_decision_client,
)
from .execution.alpaca import (
    LIVE_CONFIRMATION_PHRASE,
    AlpacaAPIError,
    AlpacaConfigError,
    LiveTradingRefused,
    MarketClosedError,
    client_from_env,
    insufficient_balance_available,
)
from .ladder import Rung, select_rung
from .limits import Limits
from .policy import (
    KILL,
    PULL_QUOTES,
    QUOTE_BOTH_SIDES,
    QUOTE_WIDE,
    STAND_DOWN,
    WIDEN,
    compose_action,
    fallback_action,
)
from .pricing import quote_prices
from .strategy import THRESHOLDS
from .state import InventoryState, build_snapshot, record_fill_slippage, update_vwap

LOG_DIR = Path(os.environ.get("JEV_LOOP_HOME", str(Path.home() / ".jev-loop")))
LOG_FILE = LOG_DIR / "log.jsonl"
LATEST_FILE = LOG_DIR / "latest.json"
LATEST_WINDOW = 120


def _fmt_money(x: float) -> str:
    return f"{x:,.1f}"


class _LegSkipped(Exception):
    """A directional leg that turned out not to be placeable."""


class _StopRequested(Exception):
    """Raised by the SIGTERM handler so a run stopped from the background
    (`kill <pid>`) shuts down exactly as cleanly as a foreground Ctrl+C
    (KeyboardInterrupt) does: cancel resting orders, then exit."""


def _handle_sigterm(signum, frame) -> None:
    raise _StopRequested()


def run(
    symbol: str,
    ticks: int | None,
    mock: bool,
    limits: Limits,
    dry_execution: bool = False,
    live: bool = False,
    confirmation: str | None = None,
) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    try:
        spec = resolve_symbol(symbol)
    except UnknownSymbolError as exc:
        print(f"cannot start: {exc}")
        return 1

    try:
        alpaca = client_from_env(
            symbol=spec.symbol,
            live=live,
            confirmation=confirmation,
            calls_per_minute=limits.max_alpaca_calls_per_minute,
        )
    except AlpacaConfigError as exc:
        print(f"cannot start: {exc}")
        return 1
    except LiveTradingRefused as exc:
        print(f"cannot start: {exc}")
        return 1

    client = resolve_decision_client(mock=mock)
    session_txt = "24/7" if spec.is_24_7 else "market hours only"
    print(
        f"asset: {spec.symbol} ({spec.asset_class}, {session_txt}, "
        f"min order ${spec.min_notional_usd:.2f}, depth: {'yes' if spec.has_depth else 'best bid/ask only'})"
    )
    print(
        f"tick cadence: one block every {limits.tick_seconds:.1f}s "
        f"(Alpaca calls capped at {limits.max_alpaca_calls_per_minute}/min)"
    )
    print(
        "your strategy lives in strategy.py (edit thresholds, or the apply_strategy hook)"
    )
    if ticks is None:
        print(
            "running continuously until stopped (Ctrl+C, or `kill <pid>` if this is "
            "in the background). Resting orders are cancelled on shutdown."
        )
    if dry_execution:
        print(
            "dry execution: real market data and a real Jev battery, no orders sent to Alpaca."
        )

    inv = InventoryState(equity_usd=0.0, high_water_mark_usd=0.0)
    api_error_streak = 0
    jev_down = False
    rest_counter = 0
    resting_quotes: dict | None = None
    recent_ticks: list[dict] = []
    block = 0
    started_at = time.time()
    totals = {"decisions": 0, "late": 0}

    n = 0
    previous_sigterm_handler = None
    try:
        previous_sigterm_handler = signal.signal(signal.SIGTERM, _handle_sigterm)
    except (ValueError, AttributeError, OSError):
        pass  # not the main thread, or a platform without SIGTERM

    try:
        while ticks is None or n < ticks:
            tick_start = time.monotonic()
            block += 1
            now = time.time()

            # Market-hours guard for equities: never even ask Jev about a
            # frozen market, and never place an order while it is closed.
            if not spec.is_24_7:
                try:
                    market_open = alpaca.is_market_open()
                except AlpacaAPIError as exc:
                    api_error_streak += 1
                    print(f"tick {block} | alpaca error checking market hours: {exc}")
                    _sleep_remaining(tick_start, limits.tick_seconds)
                    n += 1
                    continue
                if not market_open:
                    record = _closed_market_record(block, now, spec.symbol)
                    _append_log(record)
                    recent_ticks.append(record)
                    if len(recent_ticks) > LATEST_WINDOW:
                        recent_ticks = recent_ticks[-LATEST_WINDOW:]
                    _write_latest(
                        spec.symbol,
                        block,
                        recent_ticks,
                        {"route": None, "model": None},
                        started_at,
                        api_error_streak,
                        limits.tick_seconds,
                        totals,
                    )
                    print(
                        f"tick {block} | {spec.symbol} market is closed, prices are stale | HOLD"
                    )
                    n += 1
                    _sleep_remaining(tick_start, limits.tick_seconds)
                    continue

            # 2. read the book
            try:
                bids, asks = _read_top_of_book(alpaca, spec)
                # Only a fallback for mid: skip the call when the book has
                # both sides (keeps each tick under the Alpaca call budget).
                trade = alpaca.get_latest_trade() if not (bids and asks) else {}
                recent = alpaca.get_recent_trades()
                position = alpaca.get_position()
                api_error_streak = 0
            except AlpacaAPIError as exc:
                api_error_streak += 1
                print(f"tick {block} | alpaca error: {exc}")
                _sleep_remaining(tick_start, limits.tick_seconds)
                n += 1
                continue

            mid = (
                (bids[0][0] + asks[0][0]) / 2
                if bids and asks and bids[0][0] and asks[0][0]
                else float(trade.get("p", 0.0))
            )
            # The broker's position is the source of truth: resting quotes
            # can fill between ticks, and risk.py must see those fills.
            _reconcile_inventory(inv, position, now, mid, spec.min_notional_usd)

            microprice = _microprice(bids, asks, mid)
            spread_bps = (
                (asks[0][0] - bids[0][0]) / mid * 10_000
                if (mid and bids and asks)
                else 0.0
            )

            data_ts = now
            # Real prints only: real timestamps, real prices, real taker side.
            tape = _parse_trades(recent, now)
            # The current mid is the latest price point, so returns reflect
            # the live book even when the tape is quiet (~1 print/min).
            trade_prices = [(ts, px) for ts, px, _, _ in tape] + [(now, mid)]
            trade_sides = [(ts, side) for ts, _, _, side in tape if side]
            update_vwap(inv, [(ts, px, sz) for ts, px, sz, _ in tape])

            # 3. state snapshot
            snapshot = build_snapshot(
                as_of=now,
                mid=mid,
                microprice=microprice,
                spread_bps=spread_bps,
                bid_depth=bids,
                ask_depth=asks,
                trade_prices=trade_prices,
                trade_sides=trade_sides,
                inv=inv,
                data_timestamp=data_ts,
                has_depth=spec.has_depth,
            )

            # 4. battery (respecting the block deadline)
            elapsed = time.monotonic() - tick_start
            budget = max(0.05, limits.tick_seconds - elapsed - 0.15)
            decision_late = False
            answers = None
            meta = {"model": None, "latency_ms": None, "route": None}
            try:
                answers, meta = run_battery(client, snapshot, timeout=budget)
                jev_down = False
            except GatewayVerificationRequired as exc:
                print(
                    f"tick {block} | gateway needs a card on file: {exc}\n"
                    f"           add one at https://vercel.com/d?to=%2F%5Bteam%5D%2F%7E%2Fai%3Fmodal%3Dadd-credit-card, "
                    f"falling back to the mock decision client for the rest of this run."
                )
                client = resolve_decision_client(mock=True)
                jev_down = True
            except DecisionClientError as exc:
                msg = str(exc)
                if "deadline" in msg:
                    decision_late = True
                else:
                    jev_down = True
                    print(f"tick {block} | decision client error: {exc}")

            # 5. policy engine (code) + 6. pricing (code)
            if decision_late:
                action = None
            elif jev_down or answers is None:
                action = fallback_action(snapshot, limits)
            else:
                action = compose_action(answers, snapshot, limits)

            sigma = snapshot["realised_vol_short"] or 0.001
            bid_px, ask_px = quote_prices(
                mid=mid,
                inventory=snapshot["inventory"],
                sigma=sigma,
                gamma=limits.as_gamma,
                kappa=limits.as_kappa,
                time_left_s=limits.as_horizon_s,
            )

            # ladder
            decision_conf = None
            if action is not None and answers is not None:
                decision_conf = answers.get("quote_environment", {}).get("confidence")
            execution_health_score = (
                answers["execution_health"]["score"] if answers else None
            )
            risk_kill_pre = snapshot["drawdown_pct"] > limits.max_drawdown_pct
            rung = select_rung(
                risk_kill=risk_kill_pre,
                decision_late=decision_late,
                jev_down=jev_down and not decision_late,
                decision_confidence=decision_conf,
                low_confidence_threshold=limits.low_confidence_threshold,
                execution_health_score=execution_health_score,
            )

            size_factor = limits.reduce_size_factor if rung == Rung.REDUCE else 1.0
            quote_notional = limits.quote_notional_usd * size_factor
            directional_notional = limits.directional_notional_usd * size_factor

            fill_txt = "-"
            fill_qty = None
            fill_price = None
            kill_now = rung == Rung.KILL or (action is not None and action.kind == KILL)
            if kill_now:
                line_action = "KILL (flatten)"
            elif rung == Rung.HOLD_LATE or action is None:
                line_action = "HOLD (late)"
            else:
                # 7. risk veto happens inside execute_action via risk.check
                (
                    line_action,
                    fill_txt,
                    fill_qty,
                    fill_price,
                    resting_quotes,
                    rest_counter,
                ) = _execute_action(
                    alpaca=alpaca,
                    spec=spec,
                    action=action,
                    bid_px=bid_px,
                    ask_px=ask_px,
                    mid=mid,
                    quote_notional=quote_notional,
                    directional_notional=directional_notional,
                    snapshot=snapshot,
                    limits=limits,
                    inv=inv,
                    api_error_streak=api_error_streak,
                    decision_latency_ms=meta.get("latency_ms"),
                    resting_quotes=resting_quotes,
                    rest_counter=rest_counter,
                    now=now,
                    dry=dry_execution,
                    book_bid=bids[0][0] if bids else None,
                    book_ask=asks[0][0] if asks else None,
                )
                # A risk-engine kill verdict stops the loop, not just this order.
                kill_now = line_action.startswith("KILL")

            if kill_now:
                rung = Rung.KILL
                resting_quotes = None
                _flatten(alpaca, dry_execution, block)
                inv.inventory = 0.0

            # 9. log
            record = {
                "tick": block,
                "ts": now,
                "symbol": spec.symbol,
                "mid": mid,
                "vwap": snapshot["vwap"],
                "spread_bps": round(spread_bps, 2),
                "has_depth": spec.has_depth,
                "regime": answers["regime"]["choice"] if answers else None,
                "regime_conf": answers["regime"]["confidence"] if answers else None,
                "direction": answers["direction"]["choice"] if answers else None,
                # confidence + probabilities: what calibrate.py scores
                "direction_conf": (
                    answers["direction"].get("confidence") if answers else None
                ),
                "direction_probs": (
                    answers["direction"].get("probabilities") if answers else None
                ),
                "toxic_flow": answers["toxic_flow"]["noul"] if answers else None,
                "liquidity_stressed": (
                    answers["liquidity_stressed"]["noul"] if answers else None
                ),
                "quote_environment": (
                    answers["quote_environment"]["score"] if answers else None
                ),
                "quote_environment_conf": (
                    answers["quote_environment"]["confidence"] if answers else None
                ),
                "inventory_pressure": (
                    answers["inventory_pressure"]["score"] if answers else None
                ),
                "execution_health": (
                    answers["execution_health"]["score"] if answers else None
                ),
                "action": action.kind if action else "HOLD_LATE",
                "action_reason": action.reason if action else "block deadline exceeded",
                "direction_leg": action.direction_leg if action else None,
                "skew": action.skew if action else 0.0,
                # which of our quotes rest on the book after this tick; the
                # dashboard labels ticks from this and the position change,
                # not from skew (which made every quoting tick read "BUY")
                "quoted": "/".join(
                    k for k in ("bid", "ask") if k in (resting_quotes or {})
                )
                or None,
                "rung": rung.value,
                "late": rung == Rung.HOLD_LATE,
                "latency_ms": meta.get("latency_ms"),
                "model": meta.get("model"),
                "route": meta.get("route"),
                "inventory": inv.inventory,
                "unrealised_pnl_usd": snapshot["unrealised_pnl_usd"],
                "drawdown_pct": snapshot["drawdown_pct"],
                "fill": fill_txt,
                "fill_qty": fill_qty,
                "fill_price": fill_price,
            }
            totals["decisions"] += 1 if answers else 0
            totals["late"] += 1 if rung == Rung.HOLD_LATE else 0
            _append_log(record)
            recent_ticks.append(record)
            if len(recent_ticks) > LATEST_WINDOW:
                recent_ticks = recent_ticks[-LATEST_WINDOW:]
            _write_latest(
                spec.symbol,
                block,
                recent_ticks,
                meta,
                started_at,
                api_error_streak,
                limits.tick_seconds,
                totals,
            )

            regime_txt = (
                f"{answers['regime']['choice']} {answers['regime']['confidence']*100:.0f}%"
                if answers
                else "n/a"
            )
            tox_txt = f"{answers['toxic_flow']['noul']:.2f}" if answers else "n/a"
            env_txt = (
                f"{answers['quote_environment']['score']:.1f}" if answers else "n/a"
            )
            xh_txt = f"{answers['execution_health']['score']:.1f}" if answers else "n/a"
            ms_txt = (
                f"{meta.get('latency_ms'):.0f} ms" if meta.get("latency_ms") else "late"
            )
            print(
                f"tick {block} | mid {_fmt_money(mid)} | regime {regime_txt} | "
                f"tox {tox_txt} | env {env_txt} | xh {xh_txt} | {ms_txt} | {line_action} | {fill_txt}"
            )

            if rung == Rung.KILL:
                print(f"tick {block} | KILL: hard limit breached, flattened, stopping.")
                break

            n += 1
            _sleep_remaining(tick_start, limits.tick_seconds)

        _cancel_on_exit(alpaca, dry_execution, block)
        return 0
    except (KeyboardInterrupt, _StopRequested):
        print(f"\ntick {block} | stopping: interrupt received.")
        _cancel_on_exit(alpaca, dry_execution, block)
        return 0
    except Exception:
        # Any unexpected crash still leaves no open orders behind.
        print(f"\ntick {block} | crashed, cancelling open orders before exiting.")
        _cancel_on_exit(alpaca, dry_execution, block)
        raise
    finally:
        if previous_sigterm_handler is not None:
            signal.signal(signal.SIGTERM, previous_sigterm_handler)


def _parse_ts(raw: str) -> float | None:
    """RFC 3339 with up to nanosecond precision ('...T21:44:44.638490131Z')
    to epoch seconds. Returns None on anything unparseable."""
    from datetime import datetime, timezone

    if not raw:
        return None
    s = raw.strip().replace("Z", "+00:00")
    if "." in s:
        head, rest = s.split(".", 1)
        i = 0
        while i < len(rest) and rest[i].isdigit():
            i += 1
        s = f"{head}.{rest[:i][:6].ljust(6, '0')}{rest[i:]}"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _parse_trades(raw: list, now: float) -> list[tuple[float, float, float, str | None]]:
    """(timestamp, price, size, taker side) for each print at or before
    `now`, oldest first. Side is None where the feed has no taker flag
    (US equities), so those prints never bias the buy ratio."""
    out = []
    for t in raw:
        ts = _parse_ts(t.get("t", ""))
        try:
            px, sz = float(t.get("p", 0.0)), float(t.get("s", 0.0))
        except (TypeError, ValueError):
            continue
        if ts is None or ts > now or px <= 0:
            continue
        side = {"B": "buy", "S": "sell"}.get(t.get("tks"))
        out.append((ts, px, sz, side))
    out.sort(key=lambda x: x[0])
    return out


def _microprice(
    bids: list[tuple[float, float]], asks: list[tuple[float, float]], mid: float
) -> float:
    """Size-weighted top of book: leans toward the side with less size,
    since that is the side more likely to be taken out next."""
    if not bids or not asks:
        return mid
    (bp, bs), (ap, as_) = bids[0], asks[0]
    if bs + as_ <= 0:
        return mid
    return (bp * as_ + ap * bs) / (bs + as_)


def _reconcile_inventory(
    inv: InventoryState,
    position: dict,
    now: float,
    mid: float,
    min_notional_usd: float,
) -> None:
    """Overwrite the loop's inventory with the broker's position. `position`
    is Alpaca's /v2/positions/{symbol} payload, or {} when flat.

    Dust worth less than the venue's minimum order counts as flat: it can
    never be sold on its own, so treating it as a live position would start
    (and never stop) the inventory-age clock. A leftover 0.000000001 BTC did
    exactly that and froze a whole hour-long run."""
    qty = float(position.get("qty", 0.0) or 0.0) if position else 0.0
    inv.reported_qty = qty
    if inv.known_balance is not None:
        if inv.known_balance_for is not None and abs(qty - inv.known_balance_for) < 1e-12:
            # still the stale figure the rejection contradicted: trust the balance
            qty = min(qty, inv.known_balance)
        else:
            inv.known_balance = inv.known_balance_for = None  # endpoint caught up
    if qty == 0.0 or abs(qty) * mid < min_notional_usd:
        inv.inventory = 0.0
        inv.entry_price = 0.0
        inv.position_opened_at = None
        return
    inv.inventory = qty
    inv.entry_price = float(position.get("avg_entry_price", 0.0) or 0.0)
    if inv.position_opened_at is None:
        inv.position_opened_at = now


def _floor_qty(qty: float, spec: AssetSpec) -> float:
    """Round DOWN to the asset's precision. Alpaca holds BTC to 9 decimals
    but orders here use 8: rounding 0.000173205 to nearest gives 0.00017321,
    a hair over the balance, and the sell is rejected again."""
    import math

    step = 10**spec.qty_precision
    return math.floor(qty * step + 1e-9) / step


def _sell_with_balance_retry(submit, qty: float, price: float, inv, spec) -> float | None:
    """Submit a sell via `submit(qty)`. If Alpaca rejects it for insufficient
    balance, remember the balance it reports, and retry once at exactly that
    amount if it clears the venue minimum. Returns the qty actually sent, or
    None if the real balance was too small to sell. Other errors propagate."""
    try:
        submit(qty)
        return qty
    except AlpacaAPIError as exc:
        available = insufficient_balance_available(exc)
        if available is None:
            raise
        inv.known_balance = available
        inv.known_balance_for = inv.reported_qty
        inv.inventory = min(inv.inventory, available)
        retry_qty = _floor_qty(available, spec)
        if retry_qty <= 0 or retry_qty * price < spec.min_notional_usd:
            return None
        submit(retry_qty)
        return retry_qty


def _flatten(alpaca, dry: bool, block: int) -> None:
    """KILL means flatten: cancel every open order, then close the position."""
    if dry:
        print(f"tick {block} | KILL (dry): would cancel open orders and close the position.")
        return
    try:
        alpaca.cancel_all_orders()
        alpaca.close_position()
        print(f"tick {block} | KILL: open orders cancelled, position closed at market.")
    except AlpacaAPIError as exc:
        print(f"tick {block} | KILL: could not flatten cleanly, check the account: {exc}")


def _cancel_on_exit(alpaca, dry: bool, block: int) -> None:
    """Every exit path, bounded or interrupted, leaves no open orders behind.
    The position itself is kept: the next run reconciles against it."""
    if dry:
        print(f"tick {block} | shut down cleanly, dry run placed no orders.")
        return
    try:
        alpaca.cancel_all_orders()
        print(f"tick {block} | open orders cancelled, shut down cleanly.")
    except AlpacaAPIError as exc:
        print(f"tick {block} | could not cancel open orders cleanly: {exc}")


def _can_buy(inventory: float, mid: float, notional_usd: float, limits: Limits) -> bool:
    """Would a buy of `notional_usd` keep the position within max_position_usd?"""
    return abs(inventory) * mid + notional_usd <= limits.max_position_usd


def _can_sell(inventory: float, qty: float, spec: AssetSpec) -> bool:
    """A cash account cannot short: only sell what is actually held."""
    return spec.shorting_allowed or inventory >= qty - 1e-12


def _shape_quotes(
    kind: str,
    skew: float,
    bid_px: float,
    ask_px: float,
    book_bid: float | None,
    book_ask: float | None,
) -> tuple[float, float]:
    """Turn the Avellaneda-Stoikov quotes into the ones actually posted.

    - The half-spread is never narrower than the market's own, so the loop
      provides liquidity at or outside the touch instead of inside it.
    - QUOTE_WIDE and WIDEN sit further out (strategy.py multipliers).
    - Jev's inventory-pressure skew shifts both quotes: long with pressure
      (skew < 0) moves them down, so the ask is likelier to fill."""
    center = (bid_px + ask_px) / 2
    half = (ask_px - bid_px) / 2
    if book_bid and book_ask and book_ask > book_bid:
        half = max(half, (book_ask - book_bid) / 2)
    mult = {
        QUOTE_WIDE: THRESHOLDS.quote_wide_spread_mult,
        WIDEN: THRESHOLDS.widen_spread_mult,
    }.get(kind, 1.0)
    center += skew * half * THRESHOLDS.skew_price_weight
    return center - half * mult, center + half * mult


def _sellable_qty(qty: float, held: float, price: float, spec: AssetSpec) -> float:
    """A $-sized sell can come out a hair above what is held (it was bought
    at a different price). If what is held still clears the venue minimum,
    sell exactly that instead of having the order rejected."""
    if (
        not spec.shorting_allowed
        and 0 < held < qty
        and held * price >= spec.min_notional_usd
    ):
        return _floor_qty(held, spec)
    return qty


def _read_top_of_book(
    alpaca, spec: AssetSpec
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Crypto: real L2 depth from the order book. Equities: best bid/ask
    from the latest quote only, wrapped in the same (price, size) shape so
    the rest of the loop never has to know the difference. Returns empty
    lists, never fabricated levels, when nothing is available."""
    if spec.has_depth:
        book = alpaca.get_orderbook()
        bids = [(float(l["p"]), float(l["s"])) for l in book.get("b", [])]
        asks = [(float(l["p"]), float(l["s"])) for l in book.get("a", [])]
        return bids, asks

    quote = alpaca.get_latest_quote()
    bids = (
        [(float(quote["bp"]), float(quote.get("bs", 0.0)))] if quote.get("bp") else []
    )
    asks = (
        [(float(quote["ap"]), float(quote.get("as", 0.0)))] if quote.get("ap") else []
    )
    return bids, asks


def _closed_market_record(block: int, now: float, symbol: str) -> dict:
    return {
        "tick": block,
        "ts": now,
        "symbol": symbol,
        "mid": None,
        "vwap": None,
        "spread_bps": None,
        "has_depth": False,
        "regime": None,
        "regime_conf": None,
        "direction": None,
        "direction_conf": None,
        "direction_probs": None,
        "toxic_flow": None,
        "liquidity_stressed": None,
        "quote_environment": None,
        "quote_environment_conf": None,
        "inventory_pressure": None,
        "execution_health": None,
        "action": "MARKET_CLOSED",
        "action_reason": "market closed, prices are stale",
        "direction_leg": None,
        "skew": 0.0,
        "quoted": None,
        "rung": "hold_late",
        "late": False,
        "latency_ms": None,
        "model": None,
        "route": None,
        "inventory": 0.0,
        "unrealised_pnl_usd": 0.0,
        "drawdown_pct": 0.0,
        "fill": "-",
        "fill_qty": None,
        "fill_price": None,
    }


def _execute_action(
    *,
    alpaca,
    spec: AssetSpec,
    action,
    bid_px,
    ask_px,
    mid,
    quote_notional,
    directional_notional,
    snapshot,
    limits,
    inv,
    api_error_streak,
    decision_latency_ms,
    resting_quotes,
    rest_counter,
    now,
    dry: bool = False,
    book_bid: float | None = None,
    book_ask: float | None = None,
):
    """Runs the risk check, then places (or, if `dry` is true, only logs)
    the order implied by `action`. `dry` never calls cancel_all_orders,
    submit_limit_order, or submit_market_order: it reads real market data
    and gets a real Jev battery answer, but never touches the Alpaca order
    book. Every fill line in dry mode starts with "dry:". Order sizes are
    computed from a dollar target (assets.size_order), not a fixed
    quantity, so the same limits work across any asset."""
    from .risk import check as risk_check

    order_notional_usd = max(quote_notional, directional_notional)
    verdict = risk_check(
        snapshot, order_notional_usd, limits, api_error_streak, decision_latency_ms
    )
    reduce_only = False
    if not verdict.ok:
        if verdict.kill:
            return f"KILL ({verdict.veto})", "-", None, None, None, 0
        if not verdict.reduce_only:
            return (
                f"VETOED ({verdict.veto})",
                "-",
                None,
                None,
                resting_quotes,
                rest_counter,
            )
        reduce_only = True

    fill_txt = "-"
    line_action = action.kind

    if action.kind in (PULL_QUOTES, STAND_DOWN):
        if resting_quotes and not dry:
            try:
                alpaca.cancel_all_orders()
            except AlpacaAPIError:
                pass
        return line_action, fill_txt, None, None, None, 0

    if action.kind in (QUOTE_BOTH_SIDES, QUOTE_WIDE, WIDEN):
        fill_qty, fill_price = None, None
        bid_px, ask_px = _shape_quotes(
            action.kind, action.skew, bid_px, ask_px, book_bid, book_ask
        )
        leg_txt = None

        # 1. Directional leg first. It is a market order, so any resting
        # quote of ours on the other side must come off the book before it
        # goes out, or Alpaca rejects it as a potential wash trade. The
        # quotes are re-posted in step 2, after the leg has filled.
        if action.direction_leg in ("up", "down"):
            side = "buy" if action.direction_leg == "up" else "sell"
            leg_px = (book_ask if side == "buy" else book_bid) or (
                ask_px if side == "buy" else bid_px
            )
            leg_qty = size_order(directional_notional, leg_px, spec)
            if side == "sell":
                leg_qty = _sellable_qty(leg_qty, inv.inventory, leg_px, spec)
            if side == "buy" and reduce_only:
                leg_txt = "leg skipped: reduce-only (inventory held too long)"
            elif side == "buy" and not _can_buy(
                # count the bid that step 2 may post as already filled
                inv.inventory, mid, directional_notional + quote_notional, limits
            ):
                leg_txt = "leg skipped: would exceed max_position_usd"
            elif side == "sell" and not _can_sell(inv.inventory, leg_qty, spec):
                leg_txt = "leg skipped: no inventory to sell (no shorting)"
            elif dry:
                leg_txt = f"dry: would {side} {leg_qty} @ {leg_px:,.2f}"
                line_action = f"{action.kind} + {side} leg (dry)"
            else:
                try:
                    if resting_quotes:
                        alpaca.cancel_all_orders()
                        resting_quotes = None  # forces a fresh post below
                    if side == "sell":
                        sent = _sell_with_balance_retry(
                            lambda q: alpaca.submit_market_order("sell", q),
                            leg_qty,
                            leg_px,
                            inv,
                            spec,
                        )
                    else:
                        alpaca.submit_market_order("buy", leg_qty)
                        sent = leg_qty
                    if sent is None:
                        raise _LegSkipped("real balance below the venue minimum")
                    leg_qty = sent
                    inv.inventory += leg_qty if side == "buy" else -leg_qty
                    inv.fills += 1
                    inv.orders_submitted += 1
                    if inv.position_opened_at is None:
                        inv.position_opened_at = now
                    inv.entry_price = leg_px
                    record_fill_slippage(
                        inv, expected_price=mid, fill_price=leg_px, side=side
                    )
                    fill_qty, fill_price = leg_qty, leg_px
                    leg_txt = f"filled {fill_qty} @ {leg_px:,.2f}"
                    line_action = f"{action.kind} + {side} leg"
                except (MarketClosedError, _LegSkipped) as exc:
                    leg_txt = f"leg skipped: {exc}"
                except AlpacaAPIError as exc:
                    inv.orders_rejected += 1
                    leg_txt = f"leg rejected: {exc}"

        # 2. Quotes. Only the sides the account can back: a buy only while
        # the filled position would stay under max_position_usd (counting a
        # resting bid as filled: Alpaca cancels asynchronously, so it can
        # fill between cancel_all_orders() and its replacement landing), a
        # sell only against inventory actually held (cash account).
        buy_qty = size_order(quote_notional, bid_px, spec)
        sell_qty = _sellable_qty(
            size_order(quote_notional, ask_px, spec), inv.inventory, ask_px, spec
        )
        resting_bid_usd = quote_notional if (resting_quotes or {}).get("bid") else 0.0
        quote_buy = not reduce_only and _can_buy(
            inv.inventory, mid, quote_notional + resting_bid_usd, limits
        )
        quote_sell = _can_sell(inv.inventory, sell_qty, spec)
        sides_txt = "/".join(
            s for s, ok in (("bid", quote_buy), ("ask", quote_sell)) if ok
        ) or "no sides"
        # Gas-honesty rule: only cancel-replace every `rest_ticks` ticks,
        # except when the quote shape changed (e.g. QUOTE_WIDE -> WIDEN) or
        # reduce-only has to pull a resting bid right away.
        rest_counter += 1
        stale_bid = reduce_only and bool((resting_quotes or {}).get("bid"))
        shape_changed = (resting_quotes or {}).get("kind") != action.kind
        quote_txt = None
        if (
            resting_quotes is None
            or rest_counter >= limits.rest_ticks
            or stale_bid
            or shape_changed
        ):
            if dry:
                # record only the sides that would really be placed, so the
                # dashboard labels a dry run the same way as a live one
                would = {
                    k: v
                    for k, v, ok in (("bid", bid_px, quote_buy), ("ask", ask_px, quote_sell))
                    if ok
                }
                resting_quotes = {**would, "kind": action.kind} if would else None
                rest_counter = 0
                quote_txt = (
                    f"dry: would quote {sides_txt} {buy_qty}/{sell_qty} "
                    f"@ {bid_px:,.2f}/{ask_px:,.2f}"
                )
            else:
                # `placed` always records what actually reached the book, so
                # a failure on the second side never leaves the first side
                # untracked (which used to re-post a buy every tick).
                placed: dict = {}
                error = None
                try:
                    if resting_quotes:
                        alpaca.cancel_all_orders()
                    if quote_buy:
                        alpaca.submit_limit_order("buy", buy_qty, bid_px)
                        placed["bid"] = bid_px
                    if quote_sell:
                        sent = _sell_with_balance_retry(
                            lambda q: alpaca.submit_limit_order("sell", q, ask_px),
                            sell_qty,
                            ask_px,
                            inv,
                            spec,
                        )
                        if sent is not None:
                            placed["ask"] = ask_px
                except MarketClosedError as exc:
                    error = f"({exc})"
                except AlpacaAPIError as exc:
                    error = f"(order error: {exc})"
                resting_quotes = {**placed, "kind": action.kind} if placed else None
                rest_counter = 0
                if error:
                    line_action = f"{line_action} {error}"
                else:
                    placed_txt = "/".join(k for k in ("bid", "ask") if k in placed)
                    quote_txt = f"quoted {placed_txt or 'no sides'} @ {bid_px:,.2f}/{ask_px:,.2f}"

        fill_txt = "; ".join(t for t in (leg_txt, quote_txt) if t) or "-"
        if action.skew:
            line_action += f" skew {action.skew:+.2f}"
        if reduce_only:
            line_action += " (reduce-only: inventory held too long)"
        return line_action, fill_txt, fill_qty, fill_price, resting_quotes, rest_counter

    return line_action, fill_txt, None, None, resting_quotes, rest_counter


def _sleep_remaining(tick_start: float, tick_seconds: float) -> None:
    elapsed = time.monotonic() - tick_start
    remaining = tick_seconds - elapsed
    if remaining > 0:
        time.sleep(remaining)


def _append_log(record: dict) -> None:
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _write_latest(
    symbol,
    block,
    ticks,
    meta,
    started_at,
    api_error_streak,
    tick_seconds=None,
    totals: dict | None = None,
) -> None:
    """`ticks` is only the last LATEST_WINDOW ticks, for the charts. Run-wide
    counts come from `totals`: the window-based `calls` / `late_count` cap
    at 120, which made a full hour look like four minutes on the dashboard."""
    calls = len(ticks)
    late = sum(1 for t in ticks if t["action"] in ("HOLD_LATE", "MARKET_CLOSED"))
    latencies = [t["latency_ms"] for t in ticks if t.get("latency_ms")]
    avg_ms = sum(latencies) / len(latencies) if latencies else None
    payload = {
        "generated_at": time.time(),
        "symbol": symbol,
        "block": block,
        "ticks": ticks,
        "stats": {
            "avg_ms": avg_ms,
            "calls": calls,
            "late_count": late,
            "uptime_s": time.time() - started_at,
            "decision_client": meta.get("route"),
            "model": meta.get("model"),
            "api_error_streak": api_error_streak,
            "tick_seconds": tick_seconds,
            "ticks_total": block,
            "decisions_total": (totals or {}).get("decisions"),
            "late_total": (totals or {}).get("late"),
        },
    }
    tmp = LATEST_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    # On Windows the replace fails while the dashboard server has the file
    # open. The dashboard is a view, not the trading loop: retry briefly,
    # then skip this tick's write rather than crash (log.jsonl has it all).
    for attempt in range(5):
        try:
            tmp.replace(LATEST_FILE)
            return
        except PermissionError:
            time.sleep(0.02 * (attempt + 1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jev-loop run")
    parser.add_argument(
        "--paper",
        action="store_true",
        default=True,
        help="paper mode (the default, and the only mode unless --live is used)",
    )
    parser.add_argument(
        "--mock", action="store_true", help="force the mock decision client"
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=None,
        help="stop after N ticks; 0 means run forever (same as --forever)",
    )
    parser.add_argument(
        "--forever",
        action="store_true",
        help="run continuously until stopped (Ctrl+C, or SIGTERM in the background), same as --ticks 0",
    )
    parser.add_argument("--symbol", default=os.environ.get("DEFAULT_SYMBOL", "BTC/USD"))
    parser.add_argument(
        "--dry-execution",
        action="store_true",
        help="real market data and a real Jev battery, but never submit an order to Alpaca",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "trade real money on Alpaca's live endpoint instead of paper. Also "
            "requires JEV_LOOP_ALLOW_LIVE=i-understand-the-risk in the environment "
            "and a typed confirmation at startup; missing either refuses to start. "
            "See SKILL.md / README.md before ever using this."
        ),
    )
    args = parser.parse_args(argv)

    ticks = None if (args.forever or args.ticks == 0) else args.ticks

    confirmation = None
    if args.live:
        print("\n" + "!" * 70)
        print("! --live was passed. This is not paper trading.")
        print("! Every order this places spends real money in your real Alpaca")
        print("! account, with no paper safety net.")
        print("!" * 70)
        try:
            confirmation = input(
                f"Type exactly '{LIVE_CONFIRMATION_PHRASE}' to continue, anything else cancels: "
            )
        except EOFError:
            confirmation = None

    limits = Limits()
    return run(
        symbol=args.symbol,
        ticks=ticks,
        mock=args.mock,
        limits=limits,
        dry_execution=args.dry_execution,
        live=args.live,
        confirmation=confirmation,
    )


if __name__ == "__main__":
    sys.exit(main())
