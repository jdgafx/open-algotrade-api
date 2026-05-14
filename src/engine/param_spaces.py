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
}


def suggest_params(trial: Any, strategy_type: str) -> dict[str, Any]:
    """Suggest a param set for an Optuna trial given a strategy type."""
    space = PARAM_SPACES.get(strategy_type, {})
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
