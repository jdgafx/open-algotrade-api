"""Confidence-scaled leverage ladder (docs/adr/0001).

A strategy earns leverage as a function of confidence in its LIVE edge. Three tiers
by live OOS trade count:
  - < 10 trades  : OBSERVATION  -> leverage 1x, minimal size (cannot harm the account)
  - 10-29 trades : RAMP         -> leverage scales with edge toward the asset cap
  - >= 30 trades : FULL         -> half-Kelly-implied leverage, capped by the asset max
A non-positive edge always yields 1x regardless of trade count.
"""
from __future__ import annotations
from .kelly import half_kelly_fraction

# Live Hyperliquid per-asset maxima (meta endpoint, 2026-05-29). Update from the API.
HL_MAX_LEVERAGE = {"BTC": 40, "ETH": 25, "SOL": 20, "AVAX": 10, "DOGE": 10}
_DEFAULT_MAX_LEVERAGE = 5

OBSERVATION_MAX_TRADES = 10
FULL_MIN_TRADES = 30


def edge_confidence(total_trades: int, win_rate: float, payoff_ratio: float) -> float:
    """Confidence in [0,1]: how much we trust this strategy's live edge.

    Combines a sample-size factor (more live trades -> more trust) with the
    half-Kelly edge magnitude (a thicker edge -> more trust). Both must be present.
    """
    hk = half_kelly_fraction(win_rate, payoff_ratio)        # 0..0.5
    if hk <= 0:
        return 0.0
    sample_factor = min(total_trades / float(FULL_MIN_TRADES), 1.0)  # 0..1 at 30 trades
    edge_factor = min(hk / 0.25, 1.0)                                # 0..1 (0.25 hk == strong)
    return round(sample_factor * edge_factor, 6)


def ladder_leverage(symbol: str, total_trades: int, win_rate: float, payoff_ratio: float) -> int:
    """Effective leverage for the next entry, per the ladder. Always >= 1, <= asset cap."""
    cap = HL_MAX_LEVERAGE.get(symbol.upper(), _DEFAULT_MAX_LEVERAGE)
    if half_kelly_fraction(win_rate, payoff_ratio) <= 0:
        return 1
    if total_trades < OBSERVATION_MAX_TRADES:
        return 1
    conf = edge_confidence(total_trades, win_rate, payoff_ratio)  # 0..1
    lev = 1 + int(round(conf * (cap - 1)))
    if total_trades < FULL_MIN_TRADES:
        lev = min(lev, max(1, cap // 2))  # ramp tier cannot exceed half the asset cap
    return max(1, min(lev, cap))
