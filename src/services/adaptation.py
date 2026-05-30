"""Realtime leverage adaptation (ADR-0002). Pure functions; new-entries-only.

adaptation_multiplier scales the confidence-ladder leverage at entry:
  vol-target backbone × regime favourability × funding modifier, clamped.
All constants are tunable and must be validated in paper before gating real leverage.
"""
from __future__ import annotations
from typing import Optional, Set

TARGET_ATR_PCT = 2.0          # "neutral" volatility; below -> lever up, above -> lever down
ATR_FLOOR_PCT = 0.25          # guard against divide-by-tiny
VOL_FACTOR_MIN, VOL_FACTOR_MAX = 0.5, 2.0
REGIME_FAVORABLE, REGIME_ADVERSE, REGIME_NEUTRAL = 1.2, 0.6, 1.0
FUNDING_ADVERSE, FUNDING_NEUTRAL, FUNDING_FAVORABLE = 0.8, 1.0, 1.1
ADAPT_MIN, ADAPT_MAX = 0.3, 2.5


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(x, hi))


def vol_target_factor(atr_pct: Optional[float]) -> float:
    if atr_pct is None or atr_pct <= 0:
        return 1.0
    return _clamp(TARGET_ATR_PCT / max(atr_pct, ATR_FLOOR_PCT), VOL_FACTOR_MIN, VOL_FACTOR_MAX)


def regime_factor(strategy_type: str, current_regime: Optional[str], favorable_types: Set[str]) -> float:
    if current_regime is None:
        return REGIME_NEUTRAL
    return REGIME_FAVORABLE if strategy_type in favorable_types else REGIME_ADVERSE


def funding_factor(side: str, funding_bias: Optional[str]) -> float:
    if not funding_bias or funding_bias == "neutral":
        return FUNDING_NEUTRAL
    crowded_against = (side == "long" and funding_bias == "long_crowded") or \
                      (side == "short" and funding_bias == "short_crowded")
    return FUNDING_ADVERSE if crowded_against else FUNDING_FAVORABLE


def adaptation_multiplier(strategy_type: str, side: str, atr_pct: Optional[float],
                          current_regime: Optional[str], favorable_types: Set[str],
                          funding_bias: Optional[str]) -> float:
    m = (vol_target_factor(atr_pct)
         * regime_factor(strategy_type, current_regime, favorable_types)
         * funding_factor(side, funding_bias))
    return _clamp(m, ADAPT_MIN, ADAPT_MAX)
