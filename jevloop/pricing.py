"""Avellaneda-Stoikov reservation price and half spread.

Fifty-year-old market-making maths. Stays in code. No model call, ever.
Jev never sees this file's output as a question. It only answers whether
the state is worth quoting into at all (policy.py). This file answers
where to quote.

    reservation r = mid - inventory * gamma * sigma^2 * (T - t)
    half spread   = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / kappa)
"""

from __future__ import annotations

import math


def reservation_price(
    mid: float,
    inventory: float,
    gamma: float,
    sigma: float,
    time_left_s: float,
) -> float:
    """Reservation price: mid, skewed away from the side that grows inventory.

    A positive inventory (long) pulls the reservation price below mid so the
    quoting logic favours selling; a negative inventory pulls it above mid.
    """
    return mid - inventory * gamma * (sigma**2) * time_left_s


def half_spread(gamma: float, sigma: float, time_left_s: float, kappa: float) -> float:
    """Half the total quoted spread around the reservation price."""
    inventory_term = gamma * (sigma**2) * time_left_s
    liquidity_term = (2.0 / gamma) * math.log1p(gamma / kappa)
    return inventory_term + liquidity_term


def quote_prices(
    mid: float,
    inventory: float,
    sigma: float,
    gamma: float,
    kappa: float,
    time_left_s: float,
) -> tuple[float, float]:
    """Convenience wrapper: returns (bid, ask) around the reservation price."""
    r = reservation_price(mid, inventory, gamma, sigma, time_left_s)
    h = half_spread(gamma, sigma, time_left_s, kappa)
    return r - h, r + h
