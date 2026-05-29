import pytest
from src.services.kelly import half_kelly_fraction


def test_no_edge_returns_zero():
    # 50% win rate, 1:1 payoff -> zero edge -> never bet
    assert half_kelly_fraction(win_rate=0.5, payoff_ratio=1.0) == 0.0


def test_negative_edge_clamped_to_zero():
    assert half_kelly_fraction(win_rate=0.3, payoff_ratio=1.0) == 0.0


def test_known_kelly_value_is_halved():
    # p=0.6, b=1.0 -> full Kelly f* = p - (1-p)/b = 0.2 ; half = 0.10
    assert half_kelly_fraction(win_rate=0.6, payoff_ratio=1.0) == pytest.approx(0.10, abs=1e-9)


def test_capped_at_half_of_one():
    # extreme inputs must never exceed 0.5 (half of full-bankroll Kelly)
    assert half_kelly_fraction(win_rate=0.99, payoff_ratio=10.0) <= 0.5


def test_zero_or_negative_payoff_returns_zero():
    assert half_kelly_fraction(win_rate=0.9, payoff_ratio=0.0) == 0.0
    assert half_kelly_fraction(win_rate=0.9, payoff_ratio=-1.0) == 0.0


def test_win_rate_out_of_range_is_clamped():
    # robustness: win_rate outside [0,1] must not explode
    assert 0.0 <= half_kelly_fraction(win_rate=1.5, payoff_ratio=2.0) <= 0.5
    assert half_kelly_fraction(win_rate=-0.2, payoff_ratio=2.0) == 0.0
