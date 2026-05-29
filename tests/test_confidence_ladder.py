import pytest
from src.services.confidence_ladder import (
    edge_confidence, ladder_leverage, HL_MAX_LEVERAGE,
)


def test_observation_tier_under_10_trades():
    assert edge_confidence(total_trades=4, win_rate=0.7, payoff_ratio=2.0) < 0.34
    assert ladder_leverage("BTC", total_trades=4, win_rate=0.7, payoff_ratio=2.0) == 1


def test_ramp_tier_10_to_29_trades():
    lev = ladder_leverage("BTC", total_trades=20, win_rate=0.6, payoff_ratio=1.5)
    assert 1 < lev <= HL_MAX_LEVERAGE["BTC"]


def test_ramp_tier_capped_below_full():
    # ramp tier (<30 trades) must not exceed half the asset cap
    lev = ladder_leverage("BTC", total_trades=20, win_rate=0.9, payoff_ratio=5.0)
    assert lev <= HL_MAX_LEVERAGE["BTC"] // 2


def test_full_tier_caps_at_asset_max():
    assert ladder_leverage("SOL", total_trades=200, win_rate=0.7, payoff_ratio=3.0) <= HL_MAX_LEVERAGE["SOL"]
    assert ladder_leverage("DOGE", total_trades=200, win_rate=0.7, payoff_ratio=3.0) <= 10


def test_no_edge_stays_at_one():
    assert ladder_leverage("BTC", total_trades=200, win_rate=0.45, payoff_ratio=1.0) == 1


def test_unknown_symbol_uses_default_cap():
    lev = ladder_leverage("PEPE", total_trades=200, win_rate=0.7, payoff_ratio=3.0)
    assert 1 <= lev <= 5
