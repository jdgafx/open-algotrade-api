"""Fractional-Kelly position sizing helpers.

We size at HALF-Kelly: ~75% of the max compounded growth rate at far lower drawdown
and ruin probability than full Kelly. A non-positive edge yields a zero fraction
(never bet a losing or break-even edge), per docs/adr/0001.
"""
from __future__ import annotations


def half_kelly_fraction(win_rate: float, payoff_ratio: float) -> float:
    """Return the half-Kelly fraction of bankroll to risk.

    win_rate: probability of a winning trade in [0, 1].
    payoff_ratio: average win / average loss (b), > 0.
    Full Kelly f* = p - (1 - p) / b. We return max(0, f*) / 2, capped at 0.5.
    """
    if payoff_ratio <= 0:
        return 0.0
    p = max(0.0, min(1.0, win_rate))
    full_kelly = p - (1.0 - p) / payoff_ratio
    if full_kelly <= 0:
        return 0.0
    return min(full_kelly / 2.0, 0.5)
