"""Calibration: does Jev's direction call carry signal on this venue?

Reads the tick log and, for every tick where Jev gave direction
probabilities, compares its P(up) with whether the mid was actually higher
`--horizon` seconds later. Reports:

- the Brier score of P(up), next to a baseline that always predicts the
  observed up-rate, and the skill score between them (> 0 means Jev beat
  the baseline, <= 0 means it did not);
- a 10-bin reliability table (predicted P(up) vs how often up happened);
- the hit rate of each call (up / down / neutral);
- how often the price moved far enough over the horizon to pay the
  round-trip taker fee, which is the bar a directional trade has to clear.

Writes reliability.png if matplotlib is installed.

Ticks are 2-3s apart, so neighbouring ticks share most of their future:
the tick count overstates the evidence. The number of independent
horizon-length windows is printed alongside it.

Only ticks logged with `direction_probs` count (logging added 2026-09-23);
older ticks are skipped rather than guessed at.
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
from pathlib import Path

LOG_DIR = Path(os.environ.get("JEV_LOOP_HOME", str(Path.home() / ".jev-loop")))
LOG_FILE = LOG_DIR / "log.jsonl"

# Alpaca crypto, lowest volume tier: 0.25% taker each way.
# https://docs.alpaca.markets/docs/crypto-fees
ROUND_TRIP_TAKER_FEE_BPS = 50.0


def load_ticks() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    ticks = []
    with LOG_FILE.open() as f:
        for line in f:
            line = line.strip()
            if line:
                ticks.append(json.loads(line))
    return [t for t in ticks if t.get("mid") and t.get("ts")]


def _forward_bps(ticks: list[dict], times: list[float], i: int, horizon_s: float) -> float | None:
    """Mid move in bps from tick i to the first tick at least horizon_s later,
    or None if that tick is missing or lies across a gap between runs."""
    j = bisect.bisect_left(times, times[i] + horizon_s)
    if j >= len(ticks) or times[j] - times[i] > horizon_s + 15:
        return None
    return (ticks[j]["mid"] - ticks[i]["mid"]) / ticks[i]["mid"] * 10_000


def pair_predictions(
    ticks: list[dict], horizon_s: float = 300.0
) -> list[tuple[float, int, str, float]]:
    """(P(up), went up 1/0, the call, forward move in bps) for each tick with
    direction probabilities and a usable forward price."""
    ticks = sorted(ticks, key=lambda t: t["ts"])
    times = [t["ts"] for t in ticks]
    out = []
    for i, t in enumerate(ticks):
        probs = t.get("direction_probs")
        if not probs or "up" not in probs:
            continue
        move = _forward_bps(ticks, times, i, horizon_s)
        if move is None:
            continue
        out.append((float(probs["up"]), 1 if move > 0 else 0, t.get("direction"), move))
    return out


def brier_score(pairs) -> float:
    if not pairs:
        return float("nan")
    return sum((p[0] - p[1]) ** 2 for p in pairs) / len(pairs)


def baseline_brier(pairs) -> float:
    """Brier score of always predicting the observed up-rate."""
    if not pairs:
        return float("nan")
    rate = sum(p[1] for p in pairs) / len(pairs)
    return sum((rate - p[1]) ** 2 for p in pairs) / len(pairs)


def skill_score(pairs) -> float:
    """1 - Brier / baseline Brier. > 0: better than the base rate."""
    base = baseline_brier(pairs)
    if not pairs or base == 0:
        return float("nan")
    return 1 - brier_score(pairs) / base


def reliability_table(pairs, n_bins: int = 10) -> list[dict]:
    bins = [[] for _ in range(n_bins)]
    for p in pairs:
        bins[min(n_bins - 1, int(p[0] * n_bins))].append(p)
    rows = []
    for i, b in enumerate(bins):
        rows.append(
            {
                "bin": f"{i / n_bins:.1f}-{(i + 1) / n_bins:.1f}",
                "n": len(b),
                "mean_predicted": sum(p[0] for p in b) / len(b) if b else float("nan"),
                "empirical": sum(p[1] for p in b) / len(b) if b else float("nan"),
            }
        )
    return rows


def hit_rates(pairs) -> dict[str, tuple[int, float]]:
    """Per call: (n, fraction right). 'neutral' counts as right when the
    move stayed within 1 bp."""
    out = {}
    for call in ("up", "down", "neutral"):
        moves = [p[3] for p in pairs if p[2] == call]
        if not moves:
            continue
        if call == "up":
            right = sum(m > 0 for m in moves)
        elif call == "down":
            right = sum(m < 0 for m in moves)
        else:
            right = sum(abs(m) < 1 for m in moves)
        out[call] = (len(moves), right / len(moves))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jev-loop calibrate")
    parser.add_argument(
        "--horizon", type=float, default=300.0, help="seconds ahead to check the outcome"
    )
    args = parser.parse_args(argv)

    ticks = load_ticks()
    if not ticks:
        print(f"No log found at {LOG_FILE}. Run `jev-loop run --ticks 60` first.")
        return 1

    pairs = pair_predictions(ticks, horizon_s=args.horizon)
    with_probs = sum(1 for t in ticks if t.get("direction_probs"))
    if not pairs:
        print(
            f"No ticks with direction probabilities and a price {args.horizon:.0f}s later "
            f"({with_probs} ticks have probabilities; logging them started 2026-09-23). "
            "Run a longer session."
        )
        return 1

    windows = int(
        (max(t["ts"] for t in ticks) - min(t["ts"] for t in ticks)) // args.horizon
    )
    print(f"\ndirection calibration, horizon {args.horizon:.0f}s")
    print(
        f"{len(pairs)} ticks scored ({len(ticks) - with_probs} older ticks without "
        f"probabilities skipped); at most ~{windows} independent {args.horizon:.0f}s windows"
    )
    b, base, skill = brier_score(pairs), baseline_brier(pairs), skill_score(pairs)
    print(f"\nBrier P(up):        {b:.4f}")
    print(f"baseline (up-rate): {base:.4f}")
    print(
        f"skill score:        {skill:+.3f}  "
        + ("(beats the base rate)" if skill > 0 else "(no better than the base rate)")
    )

    print("\nhit rate by call:")
    for call, (n, rate) in hit_rates(pairs).items():
        note = "  (right = |move| < 1 bp)" if call == "neutral" else ""
        print(f"  {call:8s} n={n:5d}  right {100 * rate:5.1f}%{note}")

    moves = [abs(p[3]) for p in pairs]
    big = sum(m > ROUND_TRIP_TAKER_FEE_BPS for m in moves)
    print(
        f"\nmoves over the {ROUND_TRIP_TAKER_FEE_BPS:.0f} bp round-trip taker fee: "
        f"{big}/{len(moves)} ({100 * big / len(moves):.1f}%); "
        f"median |move| {sorted(moves)[len(moves) // 2]:.1f} bp"
    )

    print(f"\n{'bin':>10} {'n':>5} {'mean predicted':>15} {'empirical':>10}")
    rows = reliability_table(pairs)
    for row in rows:
        mp = f"{row['mean_predicted']:.2f}" if row["n"] else "-"
        emp = f"{row['empirical']:.2f}" if row["n"] else "-"
        print(f"{row['bin']:>10} {row['n']:>5} {mp:>15} {emp:>10}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [r["mean_predicted"] for r in rows if r["n"]]
        ys = [r["empirical"] for r in rows if r["n"]]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfectly calibrated")
        ax.plot(xs, ys, marker="o", label="Jev P(up)")
        ax.set_xlabel("predicted P(up)")
        ax.set_ylabel(f"fraction up after {args.horizon:.0f}s")
        ax.set_title("Reliability: does 80% mean 80%?")
        ax.legend()
        out = LOG_DIR / "reliability.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nwrote {out}")
    except ImportError:
        print("\n(matplotlib not installed, skipping reliability.png; the table above is the same data)")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
