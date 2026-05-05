from unittest.mock import MagicMock
from src.engine.param_spaces import PARAM_SPACES, suggest_params


def test_param_spaces_covers_key_strategies():
    required = {"mean_reversion", "adx", "bollinger", "macd", "rsi", "sma_crossover"}
    assert required.issubset(set(PARAM_SPACES.keys()))


def test_mean_reversion_has_rr_params():
    space = PARAM_SPACES["mean_reversion"]
    assert "reversion_target_pct" in space
    assert "max_loss_pct" in space
    # TP range must be above 0.008 (spec fix for R:R inversion)
    assert space["reversion_target_pct"][1] >= 0.008


def test_suggest_params_float():
    trial = MagicMock()
    trial.suggest_float.return_value = 0.015
    trial.suggest_int.return_value = 20
    params = suggest_params(trial, "mean_reversion")
    assert "reversion_target_pct" in params
    assert "zscore_entry" in params


def test_suggest_params_int():
    trial = MagicMock()
    trial.suggest_int.return_value = 25
    trial.suggest_float.return_value = 1.5
    params = suggest_params(trial, "adx")
    assert "adx_threshold" in params
    assert "exit_threshold" in params


def test_suggest_params_unknown_strategy_returns_empty():
    trial = MagicMock()
    params = suggest_params(trial, "nonexistent_strategy")
    assert params == {}
