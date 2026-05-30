"""Tests for StrategyOrchestrator._compute_entry_adaptation (Task 5a.3).

Tests the helper directly — no loop setup required.
"""
from unittest.mock import MagicMock

import pytest

from src.engine.orchestrator import StrategyOrchestrator


def _make_orch(regime_detector=None, funding_monitor=None, strategy_types=None):
    """Minimal orchestrator: only positional args + deps under test."""
    mock_client = MagicMock()
    mock_executor = MagicMock()
    orch = StrategyOrchestrator(client=mock_client, executor=mock_executor,
                                regime_detector=regime_detector,
                                funding_monitor=funding_monitor)
    if strategy_types is not None:
        orch._strategy_types = strategy_types
    return orch


# ── 1. Favourable regime + calm vol + neutral funding → multiplier > 1.0 ──

def test_favorable_regime_calm_vol_neutral_funding():
    """turtle in trending_up regime, low vol, neutral funding → > 1.0."""
    rd = MagicMock()
    rd.get_current_regime.return_value = {
        "regime": "trending_up",
        "recommended_strategies": ["turtle"],
    }
    rd.get_volatility_status.return_value = {"BTC": {"atr_pct": 1.0}}

    fm = MagicMock()
    fm.get_funding_bias.return_value = "neutral"

    orch = _make_orch(regime_detector=rd, funding_monitor=fm,
                     strategy_types={"t-btc": "turtle"})

    result = orch._compute_entry_adaptation("t-btc", "BTC", "long")
    assert result > 1.0, f"expected > 1.0, got {result}"


# ── 2. Adverse regime + violent vol → multiplier < 1.0 ──

def test_adverse_regime_violent_vol():
    """macd not in trending_up recommended list, very high vol → < 1.0."""
    rd = MagicMock()
    rd.get_current_regime.return_value = {
        "regime": "trending_up",
        "recommended_strategies": ["turtle"],  # macd NOT in list
    }
    rd.get_volatility_status.return_value = {"BTC": {"atr_pct": 5.0}}

    fm = MagicMock()
    fm.get_funding_bias.return_value = "neutral"

    orch = _make_orch(regime_detector=rd, funding_monitor=fm,
                     strategy_types={"m-btc": "macd"})

    result = orch._compute_entry_adaptation("m-btc", "BTC", "long")
    assert result < 1.0, f"expected < 1.0, got {result}"


# ── 3. regime_detector is None → exactly 1.0 ──

def test_no_regime_detector_returns_one():
    """When regime_detector is None, helper must return exactly 1.0."""
    orch = _make_orch(regime_detector=None, strategy_types={"t-btc": "turtle"})
    result = orch._compute_entry_adaptation("t-btc", "BTC", "long")
    assert result == 1.0, f"expected 1.0, got {result}"


# ── Edge: get_current_regime returns None ──

def test_regime_returns_none_safe():
    """get_current_regime returning None must not raise."""
    rd = MagicMock()
    rd.get_current_regime.return_value = None
    rd.get_volatility_status.return_value = {}

    orch = _make_orch(regime_detector=rd, strategy_types={"t-btc": "turtle"})
    result = orch._compute_entry_adaptation("t-btc", "BTC", "long")
    # No regime → REGIME_NEUTRAL factor; no vol info → vol_factor 1.0; no funding → 1.0
    assert result == 1.0, f"expected 1.0, got {result}"


# ── Edge: get_volatility_status raises ──

def test_vol_status_exception_safe():
    """Exception from get_volatility_status must not propagate."""
    rd = MagicMock()
    rd.get_current_regime.return_value = {
        "regime": "trending_up",
        "recommended_strategies": ["turtle"],
    }
    rd.get_volatility_status.side_effect = RuntimeError("network error")

    orch = _make_orch(regime_detector=rd, strategy_types={"t-btc": "turtle"})
    # Should not raise; atr_pct falls back to None → vol_factor 1.0
    result = orch._compute_entry_adaptation("t-btc", "BTC", "long")
    assert isinstance(result, float)


# ── Edge: funding_monitor raises ──

def test_funding_monitor_exception_safe():
    """Exception from get_funding_bias must not propagate."""
    rd = MagicMock()
    rd.get_current_regime.return_value = {
        "regime": "trending_up",
        "recommended_strategies": ["turtle"],
    }
    rd.get_volatility_status.return_value = {"BTC": {"atr_pct": 1.0}}

    fm = MagicMock()
    fm.get_funding_bias.side_effect = RuntimeError("timeout")

    orch = _make_orch(regime_detector=rd, funding_monitor=fm,
                     strategy_types={"t-btc": "turtle"})
    result = orch._compute_entry_adaptation("t-btc", "BTC", "long")
    assert isinstance(result, float)
