import pytest
from src.services.adaptation import (
    adaptation_multiplier, vol_target_factor, regime_factor, funding_factor,
    ADAPT_MIN, ADAPT_MAX,
)


def test_vol_target_levers_up_in_calm_down_in_violent():
    assert vol_target_factor(1.0) > 1.0      # calm (atr 1% < 2% target) -> lever up
    assert vol_target_factor(4.0) < 1.0      # violent -> lever down
    assert vol_target_factor(None) == 1.0    # unknown -> neutral


def test_regime_factor_favours_matching_strategy():
    assert regime_factor("turtle", "trending_up", {"turtle", "macd"}) > 1.0
    assert regime_factor("mean_reversion", "trending_up", {"turtle"}) < 1.0
    assert regime_factor("turtle", None, {"turtle"}) == 1.0


def test_funding_factor_trims_crowded_side():
    assert funding_factor("long", "long_crowded") < 1.0
    assert funding_factor("long", "short_crowded") > 1.0
    assert funding_factor("short", "neutral") == 1.0


def test_multiplier_is_clamped():
    hi = adaptation_multiplier("turtle", "long", atr_pct=0.1, current_regime="trending_up",
                               favorable_types={"turtle"}, funding_bias="short_crowded")
    lo = adaptation_multiplier("mean_reversion", "long", atr_pct=10.0, current_regime="trending_up",
                               favorable_types={"turtle"}, funding_bias="long_crowded")
    assert ADAPT_MIN <= lo <= hi <= ADAPT_MAX


def test_all_neutral_is_about_one():
    m = adaptation_multiplier("x", "long", atr_pct=2.0, current_regime=None,
                              favorable_types=set(), funding_bias="neutral")
    assert m == pytest.approx(1.0, abs=1e-9)
