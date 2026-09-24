# jev-loop

A 24/7 paper-trading loop built around a Jev decision battery, for the video
*How to Use Jev to Build a 24/7 HFT Trading System*.

The split, in one sentence: code computes the state, Jev answers seven typed
questions about it in one call, your code composes those answers into an
action using thresholds you set in `strategy.py`, a risk engine can veto
any of it, and execution defaults to Alpaca's paper API everywhere.

## Quick start

```bash
cd ~/.claude/skills/jev-loop
cp .env.example .env        # fill in ALPACA_API_KEY / ALPACA_SECRET_KEY at minimum
uv run pytest -q             # tests, no network needed
uv run python -m jevloop explain-split         # the split, as a table
uv run python -m jevloop validate-symbol AAPL  # resolve any symbol first
uv run python -m jevloop run --paper --ticks 30 --symbol BTC/USD
uv run python -m jevloop serve   # open http://127.0.0.1:8765
```

`--symbol` takes any crypto pair (24/7) or US equity ticker (market hours
only); `jevloop/assets.py` resolves it into endpoints, order-size rules,
and session rules. Orders are sized from a dollar target, not a fixed
quantity, so the same defaults work across assets.

No `AI_GATEWAY_API_KEY` or `TYPESAFE_API_KEY`? The loop runs on a
clearly-labelled mock decision client. It says so on the first line.

## Running continuously

A plain run stops after `--ticks N`. To keep it going:

```bash
uv run python -m jevloop run --paper --forever --symbol BTC/USD
# or, equivalently:
uv run python -m jevloop run --paper --ticks 0 --symbol BTC/USD
```

Stop it with Ctrl+C in the foreground. In the background:

```bash
nohup uv run python -m jevloop run --paper --forever --symbol BTC/USD \
  > ~/.jev-loop/continuous.log 2>&1 &
echo $! > ~/.jev-loop/continuous.pid
# ...later...
kill "$(cat ~/.jev-loop/continuous.pid)"
```

Every exit cancels open orders: a bounded run finishing, Ctrl+C, `kill`,
a KILL, or a crash. (On Windows, `kill` from outside may skip the clean
shutdown; a bounded `--ticks N` run always exits cleanly.) One tick is 2s,
so an hour is `--ticks 1800`, about 63 minutes of wall time. The dashboard (`jevloop
serve`) keeps reading the same `~/.jev-loop/latest.json` regardless of
whether the run is bounded or continuous.

## Your strategy lives in strategy.py

Everything this loop ships with is a harness, not an edge: a generic
strategy pulled out of thin air, wired in only so a demo run shows real
fills. `jevloop/strategy.py` owns the seven tunable thresholds behind
`compose_action()` and an `apply_strategy()` hook that gets one last
look at every action before it goes near an order, free to change or
veto it. The directional leg is switched off there (Jev's direction
calls showed no signal in paper runs, and Alpaca's fees make each leg
costly; see SKILL.md, "Fees"). The decision thresholds match what the
video ran; the quote-shape
settings (how far out QUOTE_WIDE and WIDEN sit, how far the inventory skew
moves prices) were added after paper runs. See SKILL.md, "How orders are
placed".

## The nine-stage loop

```
block event -> read the book -> state snapshot -> battery (seven Jev judgments)
  -> policy engine (strategy.py's thresholds + hook) -> pricing (Avellaneda-Stoikov, your code)
  -> risk veto (your code, absolute) -> directional leg, then limit quotes
  -> log, fills, inventory
```

## The fallback ladder

```
healthy + high confidence -> RUN
healthy + low confidence  -> REDUCE
late past the tick budget -> HOLD_LATE (never quote on stale state)
Jev unavailable            -> RULES_ONLY (deterministic fallback, no model)
hard limit breached        -> KILL (flatten, stop)
```

## Live trading (opt-in, off by default)

Paper is the default everywhere: `execution/alpaca.py` will not reach
Alpaca's live endpoint unless all three of the following are true at
once.

```bash
export JEV_LOOP_ALLOW_LIVE=i-understand-the-risk
uv run python -m jevloop run --live --ticks 30 --symbol BTC/USD
# then type the exact confirmation phrase the CLI prints and asks for
```

Missing the environment variable, missing `--live`, or typing anything
other than the exact confirmation phrase refuses to start; it never
falls back to paper silently. This is real money with no paper safety
net, at the same small dollar caps `limits.py` enforces on paper -- a
strategy can never raise them. Treat `--live` as what it is: a flip of
a switch made deliberately awkward so it only ever happens on purpose.

## Findings: is Jev useful for trading?

Tested 2026-09-22 .. 09-24 with `jev-1.13.0`. The full numbers are in
SKILL.md ("Testing Jev on history", "Indicators without Jev").

**What was tested**

- BTC live: paper trading on Alpaca and a 4.5-hour dry run, Jev answering
  from a live market snapshot every 2 seconds.
- BTC replay (`jevloop replay`): 455 hourly decision points over 23 days,
  given a 14-number market summary (v2) and then the price path as well
  (v3), asked 1-hour questions.
- EURUSD replay: the same, 290 hourly decision points, thresholds scaled to
  EURUSD's smaller moves.
- Indicators without Jev (`jevloop phase1`): 38-39 derived indicators in a
  walk-forward model on BTC (since 2021), SPY (since 2019) and EURUSD
  (since 2019). Every test keeps a holdout that has not been looked at.

**Results**

| | BTC | SPY | EURUSD |
|---|---|---|---|
| Jev direction call (AUC, 0.5 = coin flip) | 0.48-0.49 | not run | 0.46 |
| Jev "big move" call vs free trailing volatility | 0.58 vs 0.67 | not run | 0.67 vs 0.86 |
| Indicators, ~4h AUC | 0.549, steady every year | 0.50-0.51 | 0.515 |
| Round-trip trading cost | 0.50% | ~0.02% | ~0.02% |
| Tradable edge after costs | no (signal ~10x smaller than costs) | no (no signal) | no (signal ~7x smaller than costs) |

Adding the price path (v3) did not change Jev's answers: its P(up)
correlated 0.97 (BTC) and 0.98 (EURUSD) with the summary-only version.

**Conclusion**

- **Supported:** Jev is of no use for predicting prices from market data.
  Given price or indicator data and asked about price or volatility, it
  added nothing that plain code does not; where its answers carried
  information, a single free volatility measure carried more. This fits
  TypeSafe's own documentation, which says Jev "is not a calculator" and
  compares numbers poorly, so numeric market states are close to its worst
  case.
- **Not supported:** "Jev is of no use for trading." Untested:
  - Jev on text, which is what TypeSafe says it is built for: news
    headlines, central bank statements, earnings releases, company filings.
    Not run for lack of a free news source with reliable timestamps.
  - Jev as a feature extractor feeding a statistical model, the
    architecture TypeSafe recommends for forecasting; these tests only
    asked Jev to forecast directly.
  - Trading jobs that are not prediction: filtering news or announcements
    for relevance, flagging unusual execution or fills, checking orders
    against rules, routing information. These are close to the
    classification and routing uses TypeSafe documents.
- **Caution on the untested part:** trading signals from news are hard
  too. Such edges tend to be small, fade within minutes, and depend on
  exact timestamps. Jev on text is an open question, not a likely win.

## Safety

- Paper by default, everywhere. Live trading exists only behind the
  three-gate opt-in above.
- The hard risk caps live in `jevloop/limits.py` and never change between
  paper and live. The tunable strategy thresholds live in `jevloop/strategy.py`
  instead -- change a number, restart, see different behaviour.
- The risk engine (`jevloop/risk.py`) never calls Jev and never delegates,
  and runs after `strategy.py`'s hook has had its say, not before.
- This is not investment advice and it does not claim a profit. It is a
  scaffold for a decision battery, a policy engine, and a risk layer around
  a fast model. Edge is still your job, and it lives in `strategy.py`.
