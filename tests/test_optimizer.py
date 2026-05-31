import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
import pytest
from src.engine.optimizer import OptimizationEngine, OptimizationResult


def _make_candles(n=200):
    import numpy as np
    prices = 50000 + np.cumsum(np.random.randn(n) * 100)
    return pd.DataFrame({
        "open": prices, "high": prices * 1.001, "low": prices * 0.999,
        "close": prices, "volume": [1000.0] * n,
    })


@pytest.fixture
def engine():
    return OptimizationEngine(initial_capital=10000.0, commission_pct=0.07)


def test_optimization_result_fields():
    r = OptimizationResult(
        params={"zscore_entry": 1.5},
        in_sample_sharpe=0.8,
        out_sample_sharpe=0.6,
        out_sample_profit_factor=1.3,
        out_sample_win_rate=45.0,
        out_sample_total_trades=20,
        out_sample_max_drawdown=8.0,
        composite_score=0.5,
        passed_walkforward=True,
    )
    assert r.passed_walkforward is True
    assert r.composite_score == 0.5


def test_passes_walkforward_all_good(engine):
    result = MagicMock(
        profit_factor=1.5, win_rate=45.0, total_trades=20, max_drawdown_pct=8.0, sharpe_ratio=0.8,
    )
    assert engine._passes_walkforward(result) is True


def test_passes_walkforward_fails_low_profit_factor(engine):
    result = MagicMock(
        profit_factor=1.0, win_rate=45.0, total_trades=20, max_drawdown_pct=8.0, sharpe_ratio=0.8,
    )
    assert engine._passes_walkforward(result) is False


def test_passes_walkforward_fails_low_win_rate(engine):
    result = MagicMock(
        profit_factor=1.5, win_rate=30.0, total_trades=20, max_drawdown_pct=8.0, sharpe_ratio=0.8,
    )
    assert engine._passes_walkforward(result) is False


def test_passes_walkforward_fails_too_few_trades(engine):
    # MIN_OOS_TRADES is the permissive research screen (currently 5); the strict
    # promotion gate is PROMOTION_MIN_OOS_TRADES (30). Use a count below the screen
    # so this asserts the too-few-trades rejection specifically.
    result = MagicMock(
        profit_factor=1.5, win_rate=45.0, total_trades=4, max_drawdown_pct=8.0, sharpe_ratio=0.8,
    )
    assert engine._passes_walkforward(result) is False


def test_passes_walkforward_fails_high_drawdown(engine):
    result = MagicMock(
        profit_factor=1.5, win_rate=45.0, total_trades=20, max_drawdown_pct=20.0, sharpe_ratio=0.8,
    )
    assert engine._passes_walkforward(result) is False


@pytest.mark.asyncio
async def test_get_data_uses_cache(engine):
    df = _make_candles()
    with patch("src.engine.optimizer.candle_cache") as mock_cache:
        mock_cache.get.return_value = df
        result = await engine._get_data("BTC", "1h", 90)
        assert result is df
        mock_cache.get.assert_called_once_with("BTC", "1h", 90)


@pytest.mark.asyncio
async def test_get_data_fetches_and_caches_on_miss(engine):
    df = _make_candles()
    with patch("src.engine.optimizer.candle_cache") as mock_cache, \
         patch("src.engine.optimizer.Backtester") as mock_bt_cls:
        mock_cache.get.return_value = None
        mock_bt = AsyncMock()
        mock_bt._fetch_data = AsyncMock(return_value=df)
        mock_bt_cls.return_value = mock_bt
        result = await engine._get_data("BTC", "1h", 90)
        assert result is df
        mock_cache.set.assert_called_once()


@pytest.mark.asyncio
async def test_optimize_raises_on_insufficient_data(engine):
    with patch.object(engine, "_get_data", AsyncMock(return_value=None)):
        with pytest.raises(ValueError, match="Insufficient data"):
            await engine.optimize("mean_reversion", "BTC", "1h", lookback_days=90, n_trials=5)
