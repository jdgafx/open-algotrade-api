"""Coverage tests for newly-added OHLCV-backtestable MoonDev strategy param spaces.

Verifies (a) each strategy_type is registered in PARAM_SPACES with a non-empty
space, and (b) suggest_params returns a dict keyed by the expected params with
values inside each declared [lo, hi] bound.
"""

import pytest

from src.engine.param_spaces import PARAM_SPACES, suggest_params


# Expected param spaces (mirrors the optimizer tuning spec). Each entry maps a
# strategy_type to {param_name: (kind, lo, hi)}.
EXPECTED_SPACES: dict[str, dict[str, tuple]] = {
    "capitulation_reversal": {
        "rsi_period": ("int", 10, 21),
        "rsi_oversold": ("float", 20, 35),
        "drop_lookback": ("int", 15, 30),
        "drop_pct": ("float", 0.015, 0.06),
        "vol_spike_mult": ("float", 1.2, 2.5),
        "vol_lookback": ("int", 15, 30),
        "take_profit_pct": ("float", 1, 4),
        "stop_loss_pct": ("float", 0.5, 3),
        "hold_cap_bars": ("int", 4, 16),
    },
    "consecutive_down": {
        "n_down_bars": ("int", 2, 6),
        "rsi_period": ("int", 10, 21),
        "rsi_max": ("float", 30, 60),
        "hold_cap_bars": ("int", 4, 15),
        "take_profit_pct": ("float", 1, 5),
        "stop_loss_pct": ("float", 0.8, 3),
    },
    "day_of_week_bias": {
        "rsi_period": ("int", 10, 21),
        "sma_period": ("int", 15, 30),
        "rsi_min": ("int", 35, 60),
        "take_profit_pct": ("float", 1, 4),
        "stop_loss_pct": ("float", 0.5, 3),
        "hold_cap_bars": ("int", 3, 12),
    },
    "donchian_channel": {
        "channel_period": ("int", 10, 50),
        "atr_period": ("int", 8, 25),
        "hold_cap_bars": ("int", 6, 24),
        "take_profit_pct": ("float", 1, 6),
        "stop_loss_pct": ("float", 0.5, 3),
    },
    "ema_bollinger": {
        "short_ema_period": ("int", 10, 35),
        "long_ema_period": ("int", 40, 80),
        "bb_period": ("int", 15, 30),
        "bb_std": ("float", 1.5, 3),
    },
    "flip_flop": {
        "atr_period": ("int", 7, 21),
        "multiplier": ("float", 1.5, 4.5),
    },
    "funding_arb": {
        "funding_threshold": ("float", 0.0001, 0.002),
        "combined_target_pct": ("float", 0.5, 3.5),
        "momentum_threshold": ("float", 0.005, 0.035),
        "arb_max_loss_pct": ("float", -3, -0.5),
    },
    "gap_up_momentum": {
        "gap_threshold_pct": ("float", 0.002, 0.015),
        "uo_period_short": ("int", 5, 12),
        "uo_period_med": ("int", 10, 20),
        "uo_period_long": ("int", 20, 40),
        "uo_oversold": ("float", 15, 40),
        "adx_period": ("int", 10, 20),
        "adx_threshold": ("float", 20, 45),
        "mfi_period": ("int", 10, 20),
        "mfi_oversold": ("float", 25, 50),
        "take_profit_pct": ("float", 2, 10),
        "stop_loss_pct": ("float", 1, 4),
        "hold_cap_bars": ("int", 5, 20),
    },
    "liquidation_adx": {
        "adx_period": ("int", 10, 21),
        "adx_threshold": ("float", 18, 35),
        "exit_threshold": ("float", 12, 25),
        "rsi_period": ("int", 10, 21),
        "divergence_lookback": ("int", 5, 14),
        "vol_spike_mult": ("float", 1.5, 4),
        "vol_lookback": ("int", 14, 30),
        "max_arm_bars": ("int", 3, 12),
    },
    "liquidation_dip": {
        "liq_threshold_usd": ("int", 100000, 2000000),
        "bounce_pct": ("float", 0.002, 0.015),
        "dip_tolerance_pct": ("float", 0.0005, 0.008),
        "max_liq_age_hours": ("int", 1, 24),
        "proximity_pct": ("float", 0.005, 0.08),
        "take_profit_pct": ("float", 0.5, 5),
        "stop_loss_pct": ("float", 3, 25),
        "time_limit_hours": ("int", 2, 72),
    },
    "liquidation_momentum": {
        "vol_spike_mult": ("float", 1.5, 4),
        "vol_lookback": ("int", 10, 35),
        "momentum_candle_pct": ("float", 0.001, 0.008),
        "hold_cap_bars": ("int", 3, 16),
        "take_profit_pct": ("float", 1, 4),
        "stop_loss_pct": ("float", 0.5, 2.5),
    },
    "liquidation_revisit": {
        "liq_candle_pct": ("float", 0.01, 0.05),
        "vol_spike_mult": ("float", 1.2, 2.5),
        "vol_lookback": ("int", 10, 40),
        "revisit_window_bars": ("int", 10, 50),
        "revisit_tolerance_pct": ("float", 0.002, 0.015),
        "rsi_period": ("int", 7, 21),
        "rsi_oversold": ("int", 20, 40),
        "take_profit_pct": ("float", 0.5, 5),
        "stop_loss_pct": ("float", 0.5, 3),
        "hold_cap_bars": ("int", 3, 15),
    },
    "rsi_vwap": {
        "rsi_period": ("int", 10, 21),
        "oversold": ("int", 20, 40),
        "overbought": ("int", 60, 80),
    },
    "sma_adx_bb_vol": {
        "sma_period": ("int", 12, 34),
        "adx_period": ("int", 10, 20),
        "bb_period": ("int", 15, 30),
        "bb_std": ("float", 1.5, 3),
        "min_adx": ("int", 15, 35),
        "volume_multiplier": ("float", 1, 2.5),
    },
    "solana_sniper": {
        "sma_fast": ("int", 5, 20),
        "sma_slow": ("int", 20, 50),
        "sell_at_multiple": ("float", 2, 15),
        "stop_loss_pct": ("float", -1, -0.05),
        "min_score": ("int", 40, 85),
    },
    "supply_demand_zone": {
        "zone_lookback_days": ("int", 10, 40),
        "zone_threshold": ("float", 0.008, 0.03),
    },
    "timeinality": {
        "take_profit_pct": ("float", 0.5, 3),
        "stop_loss_pct": ("float", 0.3, 2),
        "hold_cap_bars": ("int", 1, 12),
        "require_bearish_candle": ("int", 0, 1),
    },
}


class _Trial:
    """Fake Optuna trial whose suggesters return the low bound of each range."""

    def suggest_int(self, name, lo, hi):
        return lo

    def suggest_float(self, name, lo, hi):
        return lo

    def suggest_categorical(self, name, choices):
        return choices[0]


@pytest.mark.parametrize("strategy_type", sorted(EXPECTED_SPACES.keys()))
def test_strategy_registered_with_nonempty_space(strategy_type):
    assert strategy_type in PARAM_SPACES, f"{strategy_type} missing from PARAM_SPACES"
    space = PARAM_SPACES[strategy_type]
    assert isinstance(space, dict)
    assert len(space) > 0, f"{strategy_type} has an empty param space"


@pytest.mark.parametrize("strategy_type", sorted(EXPECTED_SPACES.keys()))
def test_param_space_matches_spec(strategy_type):
    space = PARAM_SPACES[strategy_type]
    expected = EXPECTED_SPACES[strategy_type]
    assert set(space.keys()) == set(expected.keys()), (
        f"{strategy_type} param names mismatch"
    )
    for name, (kind, lo, hi) in expected.items():
        spec = space[name]
        assert spec[0] == kind, f"{strategy_type}.{name} kind mismatch"
        assert spec[1] == lo, f"{strategy_type}.{name} lo mismatch"
        assert spec[2] == hi, f"{strategy_type}.{name} hi mismatch"


@pytest.mark.parametrize("strategy_type", sorted(EXPECTED_SPACES.keys()))
def test_suggest_params_returns_expected_keys_and_bounds(strategy_type):
    expected = EXPECTED_SPACES[strategy_type]
    params = suggest_params(_Trial(), strategy_type)
    assert set(params.keys()) == set(expected.keys()), (
        f"{strategy_type} suggest_params keys mismatch"
    )
    for name, (_kind, lo, hi) in expected.items():
        value = params[name]
        assert lo <= value <= hi, (
            f"{strategy_type}.{name}={value} outside [{lo}, {hi}]"
        )
