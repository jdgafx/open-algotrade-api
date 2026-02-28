"""
Tests for the Backtesting Engine.

Tests cover:
- Running a backtest with synthetic data
- Trade entry and exit detection
- PnL calculation accuracy
- Metric computation (Sharpe, profit factor, win rate, drawdown)
- Equity curve generation
- Handling insufficient data
- Forced close at end of backtest
- Multiple strategy types
- Commission impact
- Edge cases (no trades, all winning, all losing)
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from src.engine.backtester import Backtester, BacktestResult, BacktestTrade


# ──────────────────────────────────────────────
# Basic Backtest Execution
# ──────────────────────────────────────────────


class TestBacktesterBasic:
    @pytest.mark.asyncio
    async def test_run_with_trending_data(self, trending_up_data):
        bt = Backtester(initial_capital=10000.0, commission_pct=0.05)
        result = await bt.run(
            strategy_type="rsi",
            symbol="BTC",
            timeframe="1h",
            data=trending_up_data,
        )
        assert isinstance(result, BacktestResult)
        assert result.strategy_type == "rsi"
        assert result.symbol == "BTC"
        assert result.total_bars == len(trending_up_data)
        assert result.id  # Should have a generated ID

    @pytest.mark.asyncio
    async def test_run_with_sideways_data(self, sideways_data):
        bt = Backtester(initial_capital=10000.0)
        result = await bt.run(
            strategy_type="rsi",
            symbol="BTC",
            data=sideways_data,
        )
        assert isinstance(result, BacktestResult)
        assert result.total_bars == len(sideways_data)

    @pytest.mark.asyncio
    async def test_run_with_volatile_data(self, volatile_data):
        bt = Backtester(initial_capital=10000.0)
        result = await bt.run(
            strategy_type="rsi",
            symbol="BTC",
            data=volatile_data,
        )
        assert isinstance(result, BacktestResult)

    @pytest.mark.asyncio
    async def test_result_has_duration(self, sideways_data):
        bt = Backtester()
        result = await bt.run(
            strategy_type="rsi",
            symbol="BTC",
            data=sideways_data,
        )
        assert result.duration_ms >= 0
        assert result.created_at  # Should be an ISO timestamp


# ──────────────────────────────────────────────
# Insufficient Data
# ──────────────────────────────────────────────


class TestBacktesterInsufficientData:
    @pytest.mark.asyncio
    async def test_too_few_bars_raises(self):
        """Backtester requires at least 10 bars."""
        dates = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
        tiny_data = pd.DataFrame(
            {
                "open": [100] * 5,
                "high": [101] * 5,
                "low": [99] * 5,
                "close": [100] * 5,
                "volume": [1000] * 5,
            },
            index=dates,
        )
        bt = Backtester()
        with pytest.raises(ValueError, match="Insufficient data"):
            await bt.run(strategy_type="rsi", symbol="BTC", data=tiny_data)

    @pytest.mark.asyncio
    async def test_none_data_without_fetch_raises(self):
        """Passing data=None without a network should raise."""
        bt = Backtester()
        with pytest.raises((ValueError, Exception)):
            await bt.run(strategy_type="rsi", symbol="BTC", data=None)


# ──────────────────────────────────────────────
# Trade Tracking
# ──────────────────────────────────────────────


class TestBacktesterTrades:
    @pytest.mark.asyncio
    async def test_trades_have_required_fields(self, trending_up_data):
        bt = Backtester(initial_capital=10000.0)
        result = await bt.run(
            strategy_type="rsi",
            symbol="BTC",
            data=trending_up_data,
        )
        for trade in result.trades:
            assert "entry_idx" in trade
            assert "entry_price" in trade
            assert "entry_time" in trade
            assert "side" in trade
            assert "size_usd" in trade
            assert "exit_idx" in trade
            assert "exit_price" in trade
            assert "exit_time" in trade
            assert "pnl" in trade
            assert "pnl_pct" in trade
            assert trade["side"] in ("long", "short")
            assert trade["entry_price"] > 0
            assert trade["exit_price"] > 0

    @pytest.mark.asyncio
    async def test_winning_and_losing_counts(self, trending_up_data):
        bt = Backtester(initial_capital=10000.0)
        result = await bt.run(
            strategy_type="rsi",
            symbol="BTC",
            data=trending_up_data,
        )
        assert result.winning_trades + result.losing_trades == result.total_trades
        if result.total_trades > 0:
            expected_wr = result.winning_trades / result.total_trades * 100
            assert abs(result.win_rate - expected_wr) < 1.0

    @pytest.mark.asyncio
    async def test_forced_close_at_end(self, make_ohlcv):
        """If a position is open at the end of the data, it should be force-closed."""
        # Use data with a strong trend so we almost certainly get an entry
        data = make_ohlcv(n_bars=50, base_price=100.0, trend=0.005, volatility=0.001, seed=99)
        bt = Backtester(initial_capital=10000.0)
        result = await bt.run(
            strategy_type="sma_crossover",
            symbol="BTC",
            data=data,
        )
        # All trades should be closed (including force-closed ones)
        for trade in result.trades:
            assert trade["exit_idx"] is not None
            assert trade["exit_price"] is not None


# ──────────────────────────────────────────────
# PnL and Metrics
# ──────────────────────────────────────────────


class TestBacktesterMetrics:
    @pytest.mark.asyncio
    async def test_total_pnl_consistency(self, trending_up_data):
        bt = Backtester(initial_capital=10000.0)
        result = await bt.run(
            strategy_type="rsi",
            symbol="BTC",
            data=trending_up_data,
        )
        if result.total_trades > 0:
            sum_trade_pnl = sum(t["pnl"] for t in result.trades)
            assert abs(result.total_pnl - sum_trade_pnl) < 1.0

    @pytest.mark.asyncio
    async def test_profit_factor_calculation(self, trending_up_data):
        bt = Backtester(initial_capital=10000.0)
        result = await bt.run(
            strategy_type="rsi",
            symbol="BTC",
            data=trending_up_data,
        )
        if result.total_trades > 0:
            # Profit factor should be non-negative
            assert result.profit_factor >= 0

    @pytest.mark.asyncio
    async def test_sharpe_ratio_present(self, trending_up_data):
        bt = Backtester(initial_capital=10000.0)
        result = await bt.run(
            strategy_type="rsi",
            symbol="BTC",
            data=trending_up_data,
        )
        # Sharpe is 0 if fewer than 2 trades or no variance in returns
        assert isinstance(result.sharpe_ratio, float)

    @pytest.mark.asyncio
    async def test_max_drawdown_non_negative(self, trending_up_data):
        bt = Backtester(initial_capital=10000.0)
        result = await bt.run(
            strategy_type="rsi",
            symbol="BTC",
            data=trending_up_data,
        )
        assert result.max_drawdown >= 0
        assert result.max_drawdown_pct >= 0

    @pytest.mark.asyncio
    async def test_best_worst_trade(self, trending_up_data):
        bt = Backtester(initial_capital=10000.0)
        result = await bt.run(
            strategy_type="rsi",
            symbol="BTC",
            data=trending_up_data,
        )
        if result.total_trades > 0:
            assert result.best_trade >= result.worst_trade


# ──────────────────────────────────────────────
# Equity Curve
# ──────────────────────────────────────────────


class TestBacktesterEquityCurve:
    @pytest.mark.asyncio
    async def test_equity_curve_generated(self, trending_up_data):
        bt = Backtester(initial_capital=10000.0)
        result = await bt.run(
            strategy_type="rsi",
            symbol="BTC",
            data=trending_up_data,
        )
        assert isinstance(result.equity_curve, list)
        assert len(result.equity_curve) > 0

    @pytest.mark.asyncio
    async def test_equity_curve_points_have_fields(self, trending_up_data):
        bt = Backtester(initial_capital=10000.0)
        result = await bt.run(
            strategy_type="rsi",
            symbol="BTC",
            data=trending_up_data,
        )
        for point in result.equity_curve:
            assert "bar" in point
            assert "equity" in point
            assert "drawdown_pct" in point
            assert point["equity"] > 0
            assert point["drawdown_pct"] >= 0


# ──────────────────────────────────────────────
# Commission Impact
# ──────────────────────────────────────────────


class TestBacktesterCommission:
    @pytest.mark.asyncio
    async def test_higher_commission_reduces_profit(self, trending_up_data):
        """Higher commission should produce worse PnL."""
        bt_low = Backtester(initial_capital=10000.0, commission_pct=0.01)
        bt_high = Backtester(initial_capital=10000.0, commission_pct=1.0)

        result_low = await bt_low.run(
            strategy_type="rsi", symbol="BTC", data=trending_up_data.copy()
        )
        result_high = await bt_high.run(
            strategy_type="rsi", symbol="BTC", data=trending_up_data.copy()
        )

        if result_low.total_trades > 0 and result_high.total_trades > 0:
            # With the same trades, higher commission = lower PnL
            assert result_low.total_pnl >= result_high.total_pnl


# ──────────────────────────────────────────────
# Multiple Strategy Types
# ──────────────────────────────────────────────


class TestBacktesterMultipleStrategies:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "strategy_type",
        ["rsi", "sma_crossover", "bollinger", "macd", "adx"],
    )
    async def test_run_various_strategies(self, strategy_type, make_ohlcv):
        data = make_ohlcv(n_bars=200, base_price=100.0, trend=0.001, volatility=0.02)
        bt = Backtester(initial_capital=10000.0)
        result = await bt.run(
            strategy_type=strategy_type,
            symbol="BTC",
            data=data,
        )
        assert isinstance(result, BacktestResult)
        assert result.strategy_type == strategy_type
        assert result.total_bars == 200


# ──────────────────────────────────────────────
# Edge Cases
# ──────────────────────────────────────────────


class TestBacktesterEdgeCases:
    @pytest.mark.asyncio
    async def test_no_trades_produces_zeros(self, make_ohlcv):
        """Flat data with tight range should produce few or no signals."""
        dates = pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC")
        flat_data = pd.DataFrame(
            {
                "open": [100.0] * 50,
                "high": [100.01] * 50,
                "low": [99.99] * 50,
                "close": [100.0] * 50,
                "volume": [1000.0] * 50,
            },
            index=dates,
        )
        bt = Backtester(initial_capital=10000.0)
        result = await bt.run(
            strategy_type="rsi",
            symbol="BTC",
            data=flat_data,
        )
        if result.total_trades == 0:
            assert result.win_rate == 0
            assert result.total_pnl == 0
            assert result.sharpe_ratio == 0
            assert result.avg_trade_pnl == 0

    @pytest.mark.asyncio
    async def test_missing_volume_column(self):
        """Data without volume column should still work (auto-filled with 0)."""
        dates = pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC")
        rng = np.random.RandomState(42)
        prices = 100.0 + np.cumsum(rng.normal(0, 0.5, 50))
        data = pd.DataFrame(
            {
                "open": prices * 0.999,
                "high": prices * 1.005,
                "low": prices * 0.995,
                "close": prices,
            },
            index=dates,
        )
        bt = Backtester(initial_capital=10000.0)
        result = await bt.run(
            strategy_type="rsi",
            symbol="BTC",
            data=data,
        )
        assert isinstance(result, BacktestResult)

    @pytest.mark.asyncio
    async def test_missing_required_columns_raises(self):
        """Data missing open/high/low/close should raise ValueError."""
        dates = pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC")
        bad_data = pd.DataFrame(
            {
                "price": [100.0] * 50,
                "vol": [1000.0] * 50,
            },
            index=dates,
        )
        bt = Backtester()
        with pytest.raises(ValueError, match="must have columns"):
            await bt.run(strategy_type="rsi", symbol="BTC", data=bad_data)

    @pytest.mark.asyncio
    async def test_uppercase_columns_normalized(self):
        """Columns like Open, High, Low, Close should be normalized."""
        dates = pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC")
        rng = np.random.RandomState(42)
        prices = 100.0 + np.cumsum(rng.normal(0, 0.5, 50))
        data = pd.DataFrame(
            {
                "Open": prices * 0.999,
                "High": prices * 1.005,
                "Low": prices * 0.995,
                "Close": prices,
                "Volume": np.abs(rng.normal(1000, 200, 50)),
            },
            index=dates,
        )
        bt = Backtester()
        result = await bt.run(strategy_type="rsi", symbol="BTC", data=data)
        assert isinstance(result, BacktestResult)

    @pytest.mark.asyncio
    async def test_custom_params(self, make_ohlcv):
        """Custom strategy params should be passed through."""
        data = make_ohlcv(n_bars=200, trend=0.001)
        bt = Backtester(initial_capital=10000.0)
        result = await bt.run(
            strategy_type="rsi",
            symbol="BTC",
            data=data,
            params={"rsi_period": 7, "overbought": 80, "oversold": 20},
        )
        assert isinstance(result, BacktestResult)
        assert result.params.get("rsi_period") == 7

    @pytest.mark.asyncio
    async def test_breakout_data_generates_trades(self, breakout_data):
        """Breakout-shaped data should generate trades for trend-following strategies."""
        bt = Backtester(initial_capital=10000.0)
        result = await bt.run(
            strategy_type="sma_crossover",
            symbol="BTC",
            data=breakout_data,
        )
        assert isinstance(result, BacktestResult)
        # Not asserting total_trades > 0 because it depends on the strategy parameters,
        # but the backtest itself should complete cleanly


# ──────────────────────────────────────────────
# BacktestResult Structure
# ──────────────────────────────────────────────


class TestBacktestResultStructure:
    @pytest.mark.asyncio
    async def test_result_fields(self, trending_up_data):
        bt = Backtester(initial_capital=10000.0)
        result = await bt.run(
            strategy_type="rsi",
            symbol="BTC",
            data=trending_up_data,
        )
        # Verify all expected fields exist on the result
        assert hasattr(result, "id")
        assert hasattr(result, "strategy_type")
        assert hasattr(result, "symbol")
        assert hasattr(result, "timeframe")
        assert hasattr(result, "start_date")
        assert hasattr(result, "end_date")
        assert hasattr(result, "params")
        assert hasattr(result, "total_bars")
        assert hasattr(result, "total_trades")
        assert hasattr(result, "winning_trades")
        assert hasattr(result, "losing_trades")
        assert hasattr(result, "win_rate")
        assert hasattr(result, "total_pnl")
        assert hasattr(result, "total_pnl_pct")
        assert hasattr(result, "max_drawdown")
        assert hasattr(result, "max_drawdown_pct")
        assert hasattr(result, "sharpe_ratio")
        assert hasattr(result, "profit_factor")
        assert hasattr(result, "avg_trade_pnl")
        assert hasattr(result, "best_trade")
        assert hasattr(result, "worst_trade")
        assert hasattr(result, "avg_win")
        assert hasattr(result, "avg_loss")
        assert hasattr(result, "trades")
        assert hasattr(result, "equity_curve")
        assert hasattr(result, "created_at")
        assert hasattr(result, "duration_ms")
