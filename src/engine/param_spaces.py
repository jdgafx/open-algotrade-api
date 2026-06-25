from typing import Any


# Search space definition per strategy type.
# Format: param_name -> ("float", low, high) | ("int", low, high) | ("categorical", [values])
PARAM_SPACES: dict[str, dict[str, tuple]] = {
    "mean_reversion": {
        "reversion_target_pct": ("float", 0.008, 0.025),
        "max_loss_pct": ("float", -0.020, -0.005),
        "zscore_entry": ("float", 1.2, 3.0),
        "bb_std": ("float", 1.5, 3.0),
        "sma_entry_period": ("int", 10, 35),
    },
    "adx": {
        "adx_threshold": ("int", 20, 40),
        "exit_threshold": ("int", 12, 25),
        "adx_period": ("int", 10, 20),
        "min_hold_bars": ("int", 2, 8),
    },
    "bollinger": {
        "bb_period": ("int", 12, 30),
        "bb_std": ("float", 1.5, 2.8),
        "adx_threshold": ("int", 15, 25),
        "min_hold_bars": ("int", 2, 7),
    },
    "macd": {
        "fast_period": ("int", 8, 16),
        "slow_period": ("int", 21, 32),
        "signal_period": ("int", 7, 12),
        "max_loss_pct": ("float", -0.030, -0.010),
    },
    "rsi": {
        "rsi_period": ("int", 10, 21),
        "oversold": ("int", 20, 35),
        "overbought": ("int", 65, 80),
        "min_hold_bars": ("int", 3, 8),
    },
    "sma_crossover": {
        "sma_period": ("int", 15, 50),
        "support_lookback": ("int", 15, 30),
        "min_hold_bars": ("int", 3, 8),
    },
    "vwma": {
        "fast": ("int", 15, 30),
        "mid": ("int", 35, 55),
        "slow": ("int", 65, 90),
        "min_signal_strength": ("float", 0.5, 0.8),
    },
    "nadaraya_watson": {
        "kernel_bandwidth": ("float", 5.0, 15.0),
        "overbought": ("int", 70, 85),
        "oversold": ("int", 15, 30),
        "adx_threshold": ("int", 25, 40),
        "min_hold_bars": ("int", 5, 12),
    },
    "turtle": {
        "lookback_period": ("int", 20, 60),
        "atr_multiplier": ("float", 2.0, 4.5),
        "take_profit_pct": ("float", 0.03, 0.10),
    },
    "consolidation_pop": {
        "deviance_threshold": ("float", 0.3, 0.6),
        "range_position_buy": ("float", 0.2, 0.4),
        "range_position_sell": ("float", 0.6, 0.8),
        "tp_pct": ("float", 0.015, 0.035),
        "sl_pct": ("float", 0.008, 0.020),
    },
    "vwap_bot": {
        "vwap_bias_long": ("float", 0.5, 0.9),
        "vwap_bias_short": ("float", 0.1, 0.5),
        "min_vwap_distance": ("float", 0.0005, 0.002),
    },
    "pivot_lines": {
        "pivot_lookback": ("int", 12, 36),
        "min_signal_strength": ("float", 0.4, 0.7),
    },
    "ichimoku": {
        "tenkan_period": ("int", 7, 12),
        "kijun_period": ("int", 22, 30),
        "senkou_b_period": ("int", 44, 60),
        "min_signal_strength": ("float", 0.3, 0.6),
    },
    "grid_fibonacci": {
        "proximity_pct": ("float", 0.8, 2.5),
        "take_profit_fib": ("float", 0.382, 0.618),
    },
    "correlation": {
        "lag_threshold": ("float", 0.003, 0.010),
        "sl_pct": ("float", 0.010, 0.025),
        "tp_pct": ("float", 0.015, 0.040),
    },
    "market_maker": {
        "exit_pct": ("float", 0.005, 0.020),
        "mm_stop_pct": ("float", 0.004, 0.012),
        "max_tr_pct": ("float", 0.010, 0.030),
        "time_limit_minutes": ("int", 120, 600),
    },
    "closed_market_overnight": {
        "momentum_lookback": ("int", 6, 24),
        "breakout_pct": ("float", 0.001, 0.005),
        "tp_pct": ("float", 0.005, 0.020),
        "sl_pct": ("float", 0.003, 0.015),
    },
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


def suggest_params(trial: Any, strategy_type: str, space_override: dict | None = None) -> dict[str, Any]:
    """Suggest a param set for an Optuna trial given a strategy type."""
    space = space_override if space_override is not None else PARAM_SPACES.get(strategy_type, {})
    params: dict[str, Any] = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == "float":
            params[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == "int":
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, list(spec[1]))
    return params
