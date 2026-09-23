"""The seven-question judgment battery.

One call, one latency, seven typed answers. Every question is atomic and
evaluated in isolation, and none of them ask Jev to weigh several factors
at once, and none of them ask Jev to trade or to calculate. That
composition happens in policy.py, in code, using the thresholds in
limits.py. The seven questions implement the three probabilistic rows in
the split (see split.py): fuzzy market conditions (regime, direction),
order quality (toxic_flow, liquidity_stressed), and execution health
(quote_environment, inventory_pressure, execution_health).

Question shape follows the verified TypeSafe / Vercel AI Gateway request
schema (see reference/systemone-request-shape.json):
  - noul needs only "instructions"
  - choice.criteria is a map of option -> description (or null)
  - score.criteria is an ARRAY of level descriptions, ordered low to high

Every question dict built here is checked against jevloop/split.py before
it is ever sent: if a question is off the allow-list, or its instructions
look like a request for arithmetic, the battery refuses to fire rather
than silently asking Jev to calculate something.
"""

from __future__ import annotations

from .split import assert_split_respected


def build_questions() -> dict:
    questions = {
        "regime": {
            "type": "choice",
            "instructions": "What market regime does this state describe?",
            "criteria": {
                "trending": None,
                "mean_reverting": None,
                "high_vol": None,
                "crisis": None,
            },
        },
        "direction": {
            "type": "choice",
            "instructions": "What is the price bias over the next 10 ticks?",
            "criteria": {"up": None, "down": None, "neutral": None},
        },
        "toxic_flow": {
            "type": "noul",
            "instructions": "Is the aggressive flow in this state likely informed rather than noise?",
        },
        "liquidity_stressed": {
            "type": "noul",
            "instructions": "Is the order book thinner than its recent norm?",
        },
        "quote_environment": {
            "type": "score",
            "instructions": "How favourable is this state for providing liquidity?",
            "criteria": ["Do not quote", "Marginal", "Standard", "Excellent"],
        },
        "inventory_pressure": {
            "type": "score",
            "instructions": "Given the current inventory, how urgent is it to cut the position?",
            "criteria": ["None", "Mild", "Skew hard", "Reduce now"],
        },
        "execution_health": {
            "type": "score",
            "instructions": (
                "Given recent fill ratio, reject count, slippage, and latency in this "
                "state, is execution quality optimal or degrading?"
            ),
            "criteria": ["Broken", "Degraded", "Normal", "Optimal"],
        },
    }
    assert_split_respected(questions)
    return questions


REQUIRED_ANSWER_KEYS = {
    "noul": {"type", "noul"},
    "choice": {"type", "choice", "probabilities", "confidence"},
    "score": {"type", "score", "legend", "probabilities", "confidence"},
}


def build_questions_v2() -> dict:
    """Battery v2, used by `jevloop replay` against state_v2.

    Versus v1: the horizon is an hour (v1 asked about "the next 10 ticks",
    20 seconds, far too short for a move to clear 0.5% of round-trip fees),
    every option is defined (v1 sent empty criteria), and one question is
    the one a trade actually needs answered: which comes first, +0.5% or
    -0.5%. "The current price" is the latest price the state describes."""
    questions = {
        "direction_1h": {
            "type": "choice",
            "instructions": "One hour from now, where will the price be compared with the current price?",
            "criteria": {
                "up": "more than 0.25% above the current price",
                "down": "more than 0.25% below the current price",
                "flat": "within 0.25% of the current price",
            },
        },
        "first_touch_1h": {
            "type": "choice",
            "instructions": "Within the next hour, which of these happens first?",
            "criteria": {
                "up_first": "the price rises 0.5% above the current price before it falls 0.5% below it",
                "down_first": "the price falls 0.5% below the current price before it rises 0.5% above it",
                "neither": "the price stays within 0.5% of the current price for the whole hour",
            },
        },
        "big_move_1h": {
            "type": "noul",
            "instructions": (
                "Will the price move at least 0.5% away from the current price, in "
                "either direction, at some point within the next hour?"
            ),
        },
    }
    assert_split_respected(questions)
    return questions


def validate_answers(answers: dict, questions: dict | None = None) -> None:
    """Raise if the response is missing a question or a required field.
    Cheap insurance against a schema change on either the direct API or the
    gateway silently breaking policy.py's assumptions."""
    questions = questions or build_questions()
    for key, q in questions.items():
        if key not in answers:
            raise ValueError(f"battery response missing answer for '{key}'")
        ans = answers[key]
        required = REQUIRED_ANSWER_KEYS[q["type"]]
        missing = required - set(ans.keys())
        if missing:
            raise ValueError(f"answer '{key}' missing fields {missing}")


def run_battery(client, state: dict, timeout: float) -> tuple[dict, dict]:
    """Fire the battery once. Returns (answers, meta) where meta carries the
    model name that answered and the round-trip latency in milliseconds."""
    questions = build_questions()
    answers, meta = client.ask(state=state, questions=questions, timeout=timeout)
    validate_answers(answers)
    return answers, meta
