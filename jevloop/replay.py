"""`jevloop replay`: score Jev on history instead of waiting hours live.

For decision minutes spaced `--every` minutes apart over the last `--days`
days, build state_v2 from the 24h before (Coinbase), ask Jev battery v2
once, and label what actually happened over the next hour on Alpaca. The
live runs gave ~54 independent 5-minute windows in 4.5 hours; hourly
samples over 23 days give ~550 near-independent ones for ~550 Jev calls.

Guards against fooling ourselves:
- the last `--holdout-days` are never sampled or scored unless
  `--include-holdout` is passed: look at them once, at the end;
- every Jev score is printed next to free code baselines (momentum, the
  Alpaca-Coinbase gap, trailing volatility); Jev adds value only if it
  beats them;
- no timestamps or absolute prices reach Jev (see state_v2.py).

Results go to ~/.jev-loop/replay/<variant>.jsonl, one line per sample,
written as each answer arrives, so an interrupted run resumes without
paying twice. `--score-only` scores that file without calling Jev.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from pathlib import Path

from .battery import build_questions_v2, validate_answers
from .client import DecisionClientError, resolve_decision_client
from .history import LOG_DIR, load_bars
from .state_v2 import MOVE_PCT, build_state_v2, label_outcome

REPLAY_DIR = LOG_DIR / "replay"
ROUND_TRIP_TAKER_PCT = 0.50  # Alpaca crypto, lowest tier: 0.25% each way


# ------------------------------------------------------------ sampling ----


def sample_times(first_day: dt.date, last_day: dt.date, every_min: int, holdout_start: int, include_holdout: bool) -> list[int]:
    """Decision minutes in [first_day, last_day]: at least 24h after the
    start (history) and 1h before the end (outcome)."""
    start = int(dt.datetime.combine(first_day, dt.time(), tzinfo=dt.UTC).timestamp()) + 86400
    end = int(dt.datetime.combine(last_day, dt.time(), tzinfo=dt.UTC).timestamp()) + 86400 - 3600
    now = int(dt.datetime.now(dt.UTC).timestamp()) // 60 * 60 - 3600
    end = min(end, now)
    stop = end if include_holdout else min(end, holdout_start)
    return list(range(start, stop, every_min * 60))


def _mock_answers(questions: dict, rng: random.Random) -> dict:
    """Uninformative, clearly-fake answers for --mock (plumbing tests)."""
    out = {}
    for qid, q in questions.items():
        if q["type"] == "noul":
            out[qid] = {"type": "noul", "noul": round(rng.random(), 3)}
        else:
            opts = list(q["criteria"])
            w = [rng.random() for _ in opts]
            probs = {o: x / sum(w) for o, x in zip(opts, w)}
            out[qid] = {
                "type": "choice",
                "choice": max(probs, key=probs.get),
                "probabilities": probs,
                "confidence": 0.0,
            }
    return out


# ------------------------------------------------------------- running ----


def run_replay(args) -> Path:
    today = dt.datetime.now(dt.UTC).date()
    first_day = today - dt.timedelta(days=args.days)
    holdout_start = int(dt.datetime.combine(today - dt.timedelta(days=args.holdout_days), dt.time(), tzinfo=dt.UTC).timestamp())
    times = sample_times(first_day, today, args.every, holdout_start, args.include_holdout)

    out_path = REPLAY_DIR / f"{args.variant}.jsonl"
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["t"] for l in out_path.read_text().splitlines() if l.strip()}
    todo = [t for t in times if t not in done]
    if args.max_calls is not None:
        todo = todo[: args.max_calls]

    print(f"replay '{args.variant}': {len(times)} decision points "
          f"({first_day} .. {today}, every {args.every} min, "
          f"{'including' if args.include_holdout else 'excluding'} the last {args.holdout_days} days), "
          f"{len(done)} already done, {len(todo)} to run = {len(todo)} Jev calls")
    if not todo:
        return out_path

    print("loading history (cached per day)...")
    coinbase = load_bars("coinbase", first_day, today)
    alpaca = load_bars("alpaca", first_day, today)

    client = resolve_decision_client(mock=args.mock)
    questions = build_questions_v2()
    rng = random.Random(0)
    errors_in_a_row = skipped = written = 0
    with out_path.open("a") as f:
        for i, t in enumerate(todo, 1):
            state = build_state_v2(coinbase, alpaca, t)
            outcome = label_outcome(alpaca, t)
            if state is None or outcome is None:
                skipped += 1
                continue
            try:
                if args.mock:
                    answers, meta = _mock_answers(questions, rng), {"model": "mock", "latency_ms": 0}
                else:
                    answers, meta = client.ask(state=state, questions=questions, timeout=20.0)
                validate_answers(answers, questions)
                errors_in_a_row = 0
            except (DecisionClientError, ValueError, KeyError) as exc:
                errors_in_a_row += 1
                print(f"  {i}/{len(todo)} error: {exc}")
                if errors_in_a_row >= 5:
                    print("  5 errors in a row, stopping; rerun to resume")
                    break
                continue
            f.write(json.dumps({
                "t": t, "state": state, "answers": answers, "outcome": outcome,
                "model": meta.get("model"), "latency_ms": meta.get("latency_ms"),
            }) + "\n")
            f.flush()
            written += 1
            if i % 50 == 0:
                print(f"  {i}/{len(todo)} done")
    print(f"wrote {written} samples to {out_path} ({skipped} skipped for missing data)")
    return out_path


# ------------------------------------------------------------- scoring ----


def auc(scores: list[float], labels: list[int]) -> float:
    """P(score of a random positive > score of a random negative); 0.5 = no information."""
    import bisect

    neg = sorted(s for s, y in zip(scores, labels) if not y)
    pos = [s for s, y in zip(scores, labels) if y]
    if not pos or not neg:
        return float("nan")
    total = 0.0
    for x in pos:
        lo, hi = bisect.bisect_left(neg, x), bisect.bisect_right(neg, x)
        total += lo + 0.5 * (hi - lo)
    return total / (len(pos) * len(neg))


def _ci(scores, labels, n_boot=500, seed=1) -> tuple[float, float]:
    rng = random.Random(seed)
    idx = list(range(len(scores)))
    vals = []
    for _ in range(n_boot):
        pick = [rng.choice(idx) for _ in idx]
        v = auc([scores[i] for i in pick], [labels[i] for i in pick])
        if v == v:
            vals.append(v)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def _row(name: str, scores, labels) -> str:
    a = auc(scores, labels)
    if a != a:  # NaN: every sample had the same outcome
        return f"  {name:44s} n/a (all {len(labels)} samples had the same outcome)"
    lo, hi = _ci(scores, labels)
    verdict = "above chance" if lo > 0.5 else "below chance" if hi < 0.5 else "no better than chance"
    return f"  {name:44s} AUC {a:.3f}  (95% {lo:.3f}-{hi:.3f})  {verdict}"


def brier_skill(p: list[float], y: list[int]) -> float:
    if not p:
        return float("nan")
    rate = sum(y) / len(y)
    base = sum((rate - v) ** 2 for v in y) / len(y)
    b = sum((a - v) ** 2 for a, v in zip(p, y)) / len(y)
    return 1 - b / base if base else float("nan")


def score(path: Path, holdout_start: int | None) -> dict:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if holdout_start is not None:
        rows = [r for r in rows if r["t"] < holdout_start]
    if not rows:
        print("no samples to score")
        return {}
    first = dt.datetime.fromtimestamp(min(r["t"] for r in rows), dt.UTC)
    last = dt.datetime.fromtimestamp(max(r["t"] for r in rows), dt.UTC)
    models = {r.get("model") for r in rows}
    print(f"\n{len(rows)} samples, {first:%Y-%m-%d %H:%M} .. {last:%Y-%m-%d %H:%M} UTC, model(s): {', '.join(sorted(map(str, models)))}")

    S = lambda r, k: r["state"][k]  # noqa: E731
    out = {}

    # --- direction_1h: P(up) vs "ended the hour up" ---------------------
    p_up = [r["answers"]["direction_1h"]["probabilities"].get("up", 0.0) for r in rows]
    went_up = [1 if r["outcome"]["return_1h_pct"] > 0 else 0 for r in rows]
    print(f"\ndirection_1h: did the price end the hour higher?  (base rate {100 * sum(went_up) / len(rows):.1f}% up)")
    print(_row("Jev P(up)", p_up, went_up))
    print(_row("baseline: momentum (return_1h)", [S(r, "return_1h_pct") for r in rows], went_up))
    print(_row("baseline: momentum (return_24h)", [S(r, "return_24h_pct") for r in rows], went_up))
    print(_row("baseline: Alpaca below Coinbase (-gap)", [-S(r, "alpaca_vs_coinbase_price_pct") for r in rows], went_up))
    print(f"  Jev P(up) Brier skill vs base rate: {brier_skill(p_up, went_up):+.3f} (> 0 beats the base rate)")
    out["direction_auc"] = auc(p_up, went_up)

    # --- first_touch_1h: which of +/-0.5% came first ---------------------
    decided = [r for r in rows if r["outcome"]["first_touch_1h"] in ("up_first", "down_first")]
    counts = {k: sum(1 for r in rows if r["outcome"]["first_touch_1h"] == k) for k in ("up_first", "down_first", "neither", "ambiguous")}
    print(f"\nfirst_touch_1h: +{MOVE_PCT}% before -{MOVE_PCT}%?  outcomes {counts}")
    if len(decided) >= 20:
        edge = [r["answers"]["first_touch_1h"]["probabilities"].get("up_first", 0) - r["answers"]["first_touch_1h"]["probabilities"].get("down_first", 0) for r in decided]
        up_first = [1 if r["outcome"]["first_touch_1h"] == "up_first" else 0 for r in decided]
        print(_row("Jev P(up_first) - P(down_first)", edge, up_first))
        print(_row("baseline: momentum (return_1h)", [S(r, "return_1h_pct") for r in decided], up_first))
        out["first_touch_auc"] = auc(edge, up_first)
    else:
        print("  too few decided hours to score")

    # --- big_move_1h: any 0.5% move within the hour ----------------------
    moved = [1 if r["outcome"]["big_move_1h"] else 0 for r in rows]
    print(f"\nbig_move_1h: a {MOVE_PCT}% move either way within the hour?  (base rate {100 * sum(moved) / len(rows):.1f}%)")
    print(_row("Jev P(big move)", [r["answers"]["big_move_1h"]["noul"] for r in rows], moved))
    print(_row("baseline: volatility_last_1h", [S(r, "volatility_last_1h_pct_per_hour") for r in rows], moved))
    print(_row("baseline: volatility_last_24h", [S(r, "volatility_last_24h_pct_per_hour") for r in rows], moved))
    out["big_move_auc"] = auc([r["answers"]["big_move_1h"]["noul"] for r in rows], moved)

    # --- what trading it would have made, after fees -----------------------
    print(f"\ntrading it: buy, hold one hour, sell; {ROUND_TRIP_TAKER_PCT}% round-trip taker fees")
    all_net = [r["outcome"]["return_1h_pct"] - ROUND_TRIP_TAKER_PCT for r in rows]
    print(f"  every hour (no filter)          n={len(rows):4d}  mean net {sum(all_net) / len(all_net):+.3f}% per trade")
    for thr in (0.4, 0.5, 0.6):
        pick = [r["outcome"]["return_1h_pct"] - ROUND_TRIP_TAKER_PCT for r, p in zip(rows, p_up) if p >= thr]
        if pick:
            print(f"  only when Jev P(up) >= {thr:.1f}     n={len(pick):4d}  mean net {sum(pick) / len(pick):+.3f}% per trade")
        else:
            print(f"  only when Jev P(up) >= {thr:.1f}     n=   0")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jev-loop replay")
    parser.add_argument("--days", type=int, default=30, help="history window (days)")
    parser.add_argument("--every", type=int, default=60, help="minutes between decision points")
    parser.add_argument("--holdout-days", type=int, default=7, help="most recent days kept out")
    parser.add_argument("--include-holdout", action="store_true", help="sample and score the held-out days too")
    parser.add_argument("--variant", default="v2", help="results file name under ~/.jev-loop/replay/")
    parser.add_argument("--max-calls", type=int, default=None, help="cap Jev calls this run")
    parser.add_argument("--mock", action="store_true", help="uninformative fake answers, no Jev calls")
    parser.add_argument("--score-only", action="store_true", help="score the results file, no Jev calls")
    args = parser.parse_args(argv)

    today = dt.datetime.now(dt.UTC).date()
    holdout_start = int(dt.datetime.combine(today - dt.timedelta(days=args.holdout_days), dt.time(), tzinfo=dt.UTC).timestamp())
    path = REPLAY_DIR / f"{args.variant}.jsonl"
    if not args.score_only:
        path = run_replay(args)
    if not path.exists():
        print(f"no results at {path}")
        return 1
    score(path, None if args.include_holdout else holdout_start)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
