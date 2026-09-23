---
name: jev-loop
description: A 24/7 paper-trading loop for Alpaca, any crypto pair or US equity. Every tick computes a deterministic state snapshot, fires a seven-question judgment battery at TypeSafe's Jev decision model (the Vercel AI Gateway is the normal route, a direct TypeSafe key is a faster optional extra, or a labelled mock), composes an action from your own strategy.py thresholds, prices with Avellaneda-Stoikov, checks nine hard risk limits, and executes on Alpaca paper by default. Live trading exists behind a deliberately awkward three-gate opt-in, off unless all three are set. Includes a live HTML dashboard and a calibration report.
---

# jev-loop

Install location: `~/.claude/skills/jev-loop/`.
Framework: Roan (@RohOnChain), "How to Use Jev to Build a 24/7 HFT Trading System". Installed as a Claude Code skill by Lewis Jackson.

## The split

Deterministic layer (your code): exact arithmetic (mid-price, spread, book
imbalance), hard metrics (inventory, drawdown, session VWAP), safety and
policy (stop-losses, risk vetoes, routing orders to the book).

Probabilistic layer (Jev): fuzzy conditions (trending, mean reverting,
chaotic), order quality (is flow toxic or noise), execution health (is
the setup optimal or degrading). Seven typed questions, one call, one
latency, none of them arithmetic and none of them "what should I do."

```
uv run python -m jevloop explain-split
```

prints the full two-column table with the file that owns each row.

## Invocation

Natural language, or directly:

```
cd ~/.claude/skills/jev-loop
uv run python -m jevloop run --paper --ticks 60 --symbol BTC/USD
uv run python -m jevloop run --paper --mock            # force the mock client
uv run python -m jevloop run --paper --dry-execution   # real data and a real battery, no orders sent
uv run python -m jevloop run --paper --forever         # run continuously, Ctrl+C or `kill <pid>` to stop
uv run python -m jevloop validate-symbol AAPL          # resolve any symbol before running on it
uv run python -m jevloop explain-split                 # the deterministic vs probabilistic table
uv run python -m jevloop calibrate                     # Brier score + reliability table
uv run python -m jevloop replay                        # score Jev on 30 days of history (see Testing Jev on history)
uv run python -m jevloop replay --score-only           # re-score saved answers, no Jev calls
uv run python -m jevloop serve                         # dashboard at http://127.0.0.1:8765
```

`--ticks 0` means the same thing as `--forever`. Every exit cancels open
orders: a bounded run finishing, Ctrl+C / `kill`, a KILL, or a crash. The
position itself is kept and the next run picks it up from Alpaca.

One tick is 2 seconds (`tick_seconds` in `limits.py`), so an hour is
`--ticks 1800`. Ticks that also send orders run slightly over 2s, so 1800
ticks take about 63 minutes. On Windows, stopping a background
`--forever` run from outside may skip the clean shutdown; a bounded
`--ticks N` run always exits cleanly.

Ask Claude things like:

- "run the jev-loop skill for 100 ticks on ETH/USD"
- "run jev-loop on AAPL and tell me if the market's open"
- "run jev-loop in mock mode so I can see the dashboard without spending a key"
- "keep jev-loop running continuously in the background"
- "calibrate the jev-loop and show me the Brier score"
- "explain the split in jev-loop"

## Any asset

`--symbol` accepts any crypto pair (`BTC/USD`, `ETH/USD`, 24/7) or any US
equity ticker (`AAPL`, `SPY`, `TSLA`, `NVDA`, market hours only). The
default stays `BTC/USD` because it never closes. `jevloop/assets.py`
resolves the symbol into a spec: which Alpaca endpoints to call, the
minimum order notional, quantity precision, whether shorting is allowed,
and whether the venue is open right now. Orders are sized from a dollar
target (`notional_usd` in `limits.py`), not a fixed quantity, so the same
config works whether the asset is an $85,000 coin or a $30 stock.

Equities have no Level 2 depth on the basic feed: the state snapshot uses
best bid/ask and recent trades only, and marks the missing depth fields
absent rather than inventing them. When the equity market is closed, the
loop holds and never places an order; it says so on the tick line rather
than silently doing nothing.

## Your strategy lives in strategy.py

Everything this loop ships with is a harness, not an edge: a generic
strategy pulled out of thin air, wired in only so a demo run shows real
fills. `jevloop/strategy.py` is the one file built for you to edit --
it owns the seven tunable thresholds `compose_action()` reads (when to
pull quotes, widen, quote both sides or wide, and how confident Jev has
to be before a directional leg is taken), plus an `apply_strategy()` hook
called on every tick with the action already chosen, free to change it
or veto it outright. The decision thresholds match what the video ran.
The three quote-shape settings do not: they were added after paper runs
showed the video's quotes sat a fixed ~$1.29 either side of mid (well
inside BTC's ~$25-30 spread) and that QUOTE_WIDE, WIDEN and the inventory
skew never changed a price. The hard risk caps stay separate, in
`jevloop/limits.py`, and a strategy can never raise them, only add more
caution on top.

**The directional leg is off** (`directional_leg_enabled = False` in
`strategy.py`). Over ~7,900 real-data ticks (2026-09-22/23), Jev's "down"
calls were right about half the time at 1, 5 and 15 minutes and its "up"
calls less than half, while every leg pays the spread and a taker fee.
Turn it back on only once `jevloop calibrate` shows the call has skill.

## Testing Jev on history

`jevloop replay` tests Jev on history, so each question needn't wait hours
live. For one decision point per hour over the last 30 days it builds a
market-context state from the 24h before (`state_v2.py`, Coinbase data),
asks battery v2 once (`build_questions_v2()` in `battery.py`: where the
price will be in an hour, which of +0.5% / -0.5% comes first, and whether a
0.5% move happens), and labels what really happened on Alpaca over the next
hour. About 530 decision points cost about 530 Jev calls.

- **No look-ahead, no leaks:** the state uses only bars that closed before
  the decision minute, and holds no timestamps or absolute prices, so Jev
  cannot recognise a date and remember what came next.
- **Holdout:** the last 7 days are not sampled or scored unless
  `--include-holdout` is passed. Look at them once, at the end.
- **Baselines:** every Jev score is printed next to free code signals
  (momentum, the Alpaca-Coinbase price gap, trailing volatility). Jev adds
  value only if it beats them.
- **Resumable:** answers are saved to `~/.jev-loop/replay/<variant>.jsonl`
  as they arrive; rerunning skips what is done. History is cached per day
  in `~/.jev-loop/history/`.

**Result, variant `v2` (455 hourly samples, 2026-08-25 .. 09-15, jev-1.13.0,
holdout untouched):** no usable signal. Direction AUC 0.49 and first-touch
AUC 0.46, both no better than chance; P(up) Brier skill -0.38. Big-move AUC
0.55 lost to trailing 1h volatility (0.67), which is free. Buying on Jev
P(up) >= 0.4/0.5/0.6 and holding an hour lost about as much after fees as
buying every hour (-0.44% to -0.52% per trade). Consistent with the live
calibration run: Jev's answers here add nothing that code does not.

## Fees (why this strategy loses money)

Alpaca charges crypto fees, on paper too: 0.15% maker, 0.25% taker at the
lowest volume tier (https://docs.alpaca.markets/docs/crypto-fees), taken
from what you receive. Paper fees are posted in batches, not per fill: on
2026-09-23 the $17.50 posted by 05:34 UTC covered only the fills up to
then (0.122% of $14,340 traded), and later fills were still unposted hours
afterwards, so the account's equity can overstate results until they land.
A quote round trip costs
0.30% in fees against a BTC spread of about 0.03%, so quoting cannot pay
for itself at this tier; a directional round trip needs a move above
0.50%. Any strategy here has to clear those bars first.

## How orders are placed

Code in `jevloop/loop.py`, never Jev, and all of it runs after `risk.py`:

- **Quote prices.** Avellaneda-Stoikov gives the centre; the half-spread
  is never narrower than the market's own. QUOTE_BOTH_SIDES joins the
  best bid/ask, QUOTE_WIDE sits at 2x the half-spread, WIDEN at 3x, and
  the inventory skew shifts both quotes by up to half a half-spread.
  Quotes rest `rest_ticks` (3) ticks, or are replaced at once when the
  action changes shape.
- **Only sides the account can back.** A buy only if the position would
  stay under `max_position_usd` counting any resting bid as filled (Alpaca
  cancels asynchronously, so an old bid can fill after its cancel). A sell
  only against BTC actually held: cash account, no shorting. Sells are
  rounded down to 8 decimals, never up.
- **Directional leg first.** When Jev's direction clears the threshold,
  the loop pulls its own resting quotes, sends the market order, then
  re-posts quotes, so the order can never hit its own quote (Alpaca
  rejects that as a potential wash trade).
- **Position from the broker.** Inventory is read from Alpaca every tick.
  A position worth under the $10 venue minimum counts as flat. Alpaca's
  positions endpoint can lag a fill by ~25s; if a sell is then rejected
  for insufficient balance, the loop re-sends it at the balance Alpaca
  reports and trusts that balance until the endpoint catches up.
- **Market data.** Order book plus the last 35 minutes of real trades
  (real prices, timestamps and taker side). Alpaca's crypto feed prints
  roughly once a minute, so trade intensity is often 0.
- **Rate limit.** One limiter covers all Alpaca calls at
  `max_alpaca_calls_per_minute` (150). Alpaca allows 200/min per account
  for trading and 200/min for market data, counted separately. A tick uses
  about 4 calls, so 2s ticks need about 120/min.
- **Reused connections.** The Alpaca client and the Jev client each hold
  one `requests.Session`, so calls reuse a kept-alive connection. A new
  connection per call measured ~240 ms against ~26 ms reused; before this,
  ~40% of ticks ran late after two hours and a six-hour run crashed with
  `WinError 10055` (Windows socket buffers exhausted).

## What each file does

| File | Job |
|---|---|
| `jevloop/strategy.py` | The file you edit: the seven decision thresholds behind `compose_action()`, the directional-leg switch (off), the three quote-shape settings (wide and widen multipliers, skew weight), and the `apply_strategy()` hook to override or veto an action. |
| `jevloop/limits.py` | The hard risk caps and operational numbers (2s ticks, 150 Alpaca calls/min, $20 order size, $50 position cap). Never overridable by a strategy. |
| `jevloop/assets.py` | Resolves any symbol into a spec: endpoints, notional floor, precision, shorting, market hours. |
| `jevloop/state.py` | Deterministic state snapshot, under ~400 tokens, strict timestamp discipline, session VWAP, honest depth degradation. |
| `jevloop/split.py` | The allow-list of battery questions and the guard that refuses any question that looks like arithmetic. |
| `jevloop/battery.py` | The seven-question Jev battery (regime, direction, toxic flow, liquidity stress, quote environment, inventory pressure, execution health). |
| `jevloop/client.py` | Resolves the decision client: Vercel AI Gateway (the normal route) or a direct TypeSafe key (a faster optional extra) or a mock, prints which one won, pins and logs the model per response. |
| `jevloop/policy.py` | `compose_action()`, code, not Jev, turns seven answers into KILL / PULL_QUOTES / WIDEN / QUOTE_BOTH_SIDES / QUOTE_WIDE / STAND_DOWN using strategy.py's thresholds, plus a directional leg, then hands the result to strategy.py's hook. |
| `jevloop/pricing.py` | Avellaneda-Stoikov reservation price and half spread. |
| `jevloop/risk.py` | Nine hard limits, checked before every order, never delegated. Position, daily loss, drawdown, API-error and leverage breaches KILL (cancel, close the position, stop). Order size, stale data and slow decisions veto the tick. Inventory held past 15 minutes is reduce-only: sells still go, buys don't. |
| `jevloop/ladder.py` | The five-rung fallback ladder (RUN / REDUCE / HOLD_LATE / RULES_ONLY / KILL). |
| `jevloop/execution/alpaca.py` | Alpaca execution and market data, crypto or equities: orders, position read and close, recent-trades window, rate limiter. Paper by default; live trading exists only behind the three-gate opt-in (see Live trading below). Refuses to place an equity order while the market is closed. |
| `jevloop/loop.py` | The nine-stage block loop plus order placement (see How orders are placed). One JSON line per tick to `~/.jev-loop/log.jsonl`, and `~/.jev-loop/latest.json` for the dashboard. Cancels open orders on every exit. |
| `jevloop/replay.py` | `jevloop replay`: samples history, asks battery v2, labels outcomes, and scores Jev against code baselines (AUC with 95% intervals, Brier skill, and one-hour trades after fees). Resumable; `--score-only` makes no Jev calls. |
| `jevloop/state_v2.py` | Replay state (returns, ranges, volatility over 15m/1h/4h/24h, range position, distance from VWAP, volume, Alpaca-Coinbase gap) and outcome labelling. No timestamps or price levels. |
| `jevloop/history.py` | Downloads and caches 1-minute BTC/USD bars per day: Coinbase (public, the real market) for the state, Alpaca for outcomes. |
| `jevloop/calibrate.py` | Scores Jev's direction P(up) against the mid `--horizon` seconds later (default 300): Brier score vs a base-rate baseline (skill score), hit rate per call, share of moves big enough to pay the round-trip taker fee, and a 10-bin reliability table; `reliability.png` if matplotlib is present. Needs ticks logged with `direction_probs` (from 2026-09-23). |
| `jevloop/serve.py` | Tiny static server for `dashboard/index.html` and `dashboard/wall.html`. Sends `Cache-Control: no-cache` so a browser picks up dashboard changes on reload. The page only re-fetches `latest.json` by itself, so after changing the dashboard, reload an open tab (Ctrl+F5) and restart `serve` after changing `serve.py`. |
| `dashboard/*.html` | Two live dashboards, polling `latest.json`. No simulation. Tick, decision and late counts and uptime are for the whole run; charts show the last 90 ticks. Shows "stopped" once the loop stops writing. Each tick is labelled from logged facts: QUOTE BOTH / BID / ASK (blue) from the `quoted` field, FILLED BUY / SELL when the broker position moved (a quote filled), BUY / SELL for a directional leg, and LATE / STAND DOWN / PULL QUOTES (amber). |
| `tests/` | pytest, no network: policy thresholds, risk vetoes, the ladder, state maths, the asset resolver, the split guard, the mock client, the paper-URL guard, the three-gate live check, plus order placement against a fake broker (`test_execution_safety.py`, `test_quote_shape.py`, `test_sell_balance.py`) and real-tape parsing (`test_market_data.py`). |

## Setup

1. Copy `.env.example` to `.env` in this folder and fill in what you have.
2. `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` are required, paper keys only,
   from https://app.alpaca.markets/paper/dashboard/overview. Paper keys
   start with `PK`.
3. `AI_GATEWAY_API_KEY` is the normal way to reach Jev: works today, no
   invite or waitlist, but the Vercel team needs a card on file before
   the gateway will serve requests, even the free credits. A few dollars
   covers roughly a month at this tick rate. `TYPESAFE_API_KEY` is an
   optional extra, slightly faster (one less hop), useful only if you are
   already off TypeSafe's waitlist. Without either, the loop runs on a
   clearly-labelled mock decision client so nothing above blocks on a
   billing page or a waitlist.
4. `DEFAULT_SYMBOL` is optional; it sets what `--symbol` defaults to if
   you don't pass one.

## Live trading (opt-in, off by default)

Paper is the default everywhere in this skill. Turning on live trading
needs all three of the following, together, every time:

1. the `--live` flag on the command line
2. `JEV_LOOP_ALLOW_LIVE=i-understand-the-risk` in the environment
3. typing the exact confirmation phrase the CLI asks for at startup

Missing any one of the three refuses to start; it never falls back to
paper silently. All three present routes every order to Alpaca's live
endpoint instead of paper, with real money and no safety net beyond the
risk caps in `limits.py`, which still apply and are not raised by going
live. This is the "flip of a switch" the video describes, made
deliberately awkward on purpose so it only ever happens on purpose. See
README.md for the exact commands.

## Dependencies

`uv`-managed virtual environment under `.venv/` with Python 3.10+ and:

- `requests>=2.31`
- `python-dotenv>=1.0`
- `matplotlib>=3.8` (optional, only for `reliability.png`)
- `pytest>=8.0` (dev only, for `tests/`)

## What this is not

Paper by default, everywhere. Live trading exists only behind the
three-gate opt-in above and starts with the same small dollar caps as
paper. This does not claim a profit. It cannot turn a bad strategy into
a good one: Jev makes seven judgments cheap and fast, edge is still
yours, and yours lives in strategy.py.
