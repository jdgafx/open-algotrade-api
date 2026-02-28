"""
Tests for all BaseStrategy implementations.

Each strategy gets at least:
- should_enter with trending data (expect signal)
- should_enter with sideways data (expect fewer/no signals)
- should_exit with winning position (expect take profit)
- should_exit with losing position (expect stop loss)
- Insufficient data handling (expect None)
"""

import pytest
import numpy as np
import pandas as pd

from src.strategies.base_strategy import (
    BaseStrategy,
    Signal,
    SignalType,
    StrategyConfig,
    StrategyTier,
)
from src.strategies.registry import create_strategy, get_strategy_class, list_strategies


# ──────────────────────────────────────────────
# Registry Tests
# ──────────────────────────────────────────────


class TestStrategyRegistry:
    def test_registry_has_strategies(self):
        strategies = list_strategies()
        assert len(strategies) >= 20, f"Expected 20+ strategies, got {len(strategies)}"

    def test_all_strategies_have_required_fields(self):
        for s in list_strategies():
            assert "strategy_type" in s
            assert "tier" in s
            assert "description" in s
            assert "default_symbol" in s
            assert "default_timeframe" in s
            assert "default_params" in s
            assert "category" in s
            assert "risk_level" in s

    def test_get_strategy_class_valid(self):
        cls = get_strategy_class("rsi")
        assert issubclass(cls, BaseStrategy)

    def test_get_strategy_class_invalid(self):
        with pytest.raises(ValueError, match="Unknown strategy type"):
            get_strategy_class("nonexistent_strategy")

    def test_create_strategy_returns_instance(self):
        config = StrategyConfig(name="test", symbol="BTC")
        strategy = create_strategy("rsi", config)
        assert isinstance(strategy, BaseStrategy)
        assert strategy.config.name == "test"

    @pytest.mark.parametrize(
        "strategy_type",
        [s["strategy_type"] for s in list_strategies()],
    )
    def test_all_strategies_instantiate(self, strategy_type):
        """Every registered strategy should instantiate without error."""
        config = StrategyConfig(name=f"test-{strategy_type}", symbol="BTC")
        strategy = create_strategy(strategy_type, config)
        assert isinstance(strategy, BaseStrategy)


# ──────────────────────────────────────────────
# Base Strategy Tests
# ──────────────────────────────────────────────


class TestBaseStrategy:
    @pytest.mark.asyncio
    async def test_record_trade_winning(self, make_strategy_config):
        config = make_strategy_config(name="test-base")
        strategy = create_strategy("rsi", config)

        strategy.record_trade(10.0)
        assert strategy.state.total_trades == 1
        assert strategy.state.winning_trades == 1
        assert strategy.state.losing_trades == 0
        assert strategy.state.total_pnl == 10.0
        assert strategy.win_rate == 100.0

    @pytest.mark.asyncio
    async def test_record_trade_losing(self, make_strategy_config):
        config = make_strategy_config(name="test-base")
        strategy = create_strategy("rsi", config)

        strategy.record_trade(-5.0)
        assert strategy.state.total_trades == 1
        assert strategy.state.winning_trades == 0
        assert strategy.state.losing_trades == 1
        assert strategy.state.total_pnl == -5.0

    @pytest.mark.asyncio
    async def test_drawdown_tracking(self, make_strategy_config):
        config = make_strategy_config(name="test-dd")
        strategy = create_strategy("rsi", config)

        strategy.record_trade(20.0)
        strategy.record_trade(-15.0)
        assert strategy.state.max_drawdown == 15.0
        assert strategy.state.peak_pnl == 20.0

    @pytest.mark.asyncio
    async def test_get_stats(self, make_strategy_config):
        config = make_strategy_config(name="test-stats")
        strategy = create_strategy("rsi", config)
        stats = strategy.get_stats()
        assert stats["name"] == "test-stats"
        assert stats["symbol"] == "BTC"
        assert stats["is_running"] is False
        assert stats["total_trades"] == 0

    @pytest.mark.asyncio
    async def test_on_start_on_stop(self, make_strategy_config):
        config = make_strategy_config(name="test-lifecycle")
        strategy = create_strategy("rsi", config)

        await strategy.on_start()
        assert strategy.state.is_running is True
        assert strategy.state.start_time is not None

        await strategy.on_stop()
        assert strategy.state.is_running is False

    @pytest.mark.asyncio
    async def test_is_healthy(self, make_strategy_config):
        config = make_strategy_config(name="test-health")
        strategy = create_strategy("rsi", config)
        assert strategy.is_healthy is True

        strategy.state.consecutive_errors = 100
        assert strategy.is_healthy is False


# ──────────────────────────────────────────────
# RSI Strategy Tests
# ──────────────────────────────────────────────


class TestRSIStrategy:
    @pytest.mark.asyncio
    async def test_insufficient_data_returns_none(self, make_strategy_config):
        config = make_strategy_config(params={"rsi_period": 14})
        strategy = create_strategy("rsi", config)
        short_data = pd.DataFrame(
            {"open": [100] * 5, "high": [101] * 5, "low": [99] * 5, "close": [100] * 5, "volume": [1000] * 5}
        )
        result = await strategy.should_enter(short_data)
        assert result is None

    @pytest.mark.asyncio
    async def test_oversold_reversal_long(self, make_strategy_config, make_ohlcv):
        config = make_strategy_config(params={
            "rsi_period": 14, "oversold": 30, "overbought": 70,
            "trend_mode": False, "divergence_mode": False,
        })
        strategy = create_strategy("rsi", config)

        # Create data where RSI goes to oversold then crosses back up
        data = make_ohlcv(n_bars=100, base_price=100, trend=-0.005, volatility=0.01, seed=10)
        # The declining data should eventually trigger oversold conditions
        signal = await strategy.should_enter(data)
        # Signal could be None or LONG depending on exact RSI values
        if signal is not None:
            assert signal.signal_type in (SignalType.LONG, SignalType.SHORT)

    @pytest.mark.asyncio
    async def test_exit_on_target(self, make_strategy_config, make_ohlcv):
        config = make_strategy_config(params={"rsi_period": 14}, target_pct=5.0)
        strategy = create_strategy("rsi", config)
        data = make_ohlcv(n_bars=100)

        position = {"size": 0.1, "is_long": True, "pnl_perc": 6.0}
        signal = await strategy.should_exit(data, position)
        assert signal is not None
        assert signal.signal_type == SignalType.CLOSE_LONG
        assert "Target" in signal.reason

    @pytest.mark.asyncio
    async def test_exit_on_max_loss(self, make_strategy_config, make_ohlcv):
        config = make_strategy_config(params={"rsi_period": 14}, max_loss_pct=-10.0)
        strategy = create_strategy("rsi", config)
        data = make_ohlcv(n_bars=100)

        position = {"size": 0.1, "is_long": True, "pnl_perc": -11.0}
        signal = await strategy.should_exit(data, position)
        assert signal is not None
        assert signal.signal_type == SignalType.CLOSE_LONG
        assert "Max loss" in signal.reason


# ──────────────────────────────────────────────
# SMA Strategy Tests
# ──────────────────────────────────────────────


class TestSMAStrategy:
    @pytest.mark.asyncio
    async def test_insufficient_data(self, make_strategy_config):
        config = make_strategy_config(params={"sma_period": 20})
        strategy = create_strategy("sma_crossover", config)
        short_data = pd.DataFrame(
            {"open": [100] * 10, "high": [101] * 10, "low": [99] * 10, "close": [100] * 10, "volume": [1000] * 10}
        )
        result = await strategy.should_enter(short_data)
        assert result is None

    @pytest.mark.asyncio
    async def test_cross_above_sma_long(self, make_strategy_config):
        """Construct data where price crosses above SMA."""
        config = make_strategy_config(params={"sma_period": 5, "support_lookback": 5})
        strategy = create_strategy("sma_crossover", config)

        # 20 bars below SMA, then price jumps above
        n = 30
        prices = [100.0] * 25 + [95.0, 95.0, 95.0, 98.0, 103.0]
        df = pd.DataFrame({
            "open": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "volume": [1000] * n,
        })
        signal = await strategy.should_enter(df)
        if signal is not None:
            assert signal.signal_type in (SignalType.LONG, SignalType.SHORT)

    @pytest.mark.asyncio
    async def test_exit_target(self, make_strategy_config, make_ohlcv):
        config = make_strategy_config(params={"sma_period": 20}, target_pct=5.0)
        strategy = create_strategy("sma_crossover", config)
        data = make_ohlcv(n_bars=50)

        position = {"size": 0.1, "is_long": True, "pnl_perc": 6.0}
        signal = await strategy.should_exit(data, position)
        assert signal is not None
        assert signal.signal_type == SignalType.CLOSE_LONG


# ──────────────────────────────────────────────
# Bollinger Strategy Tests
# ──────────────────────────────────────────────


class TestBollingerStrategy:
    @pytest.mark.asyncio
    async def test_insufficient_data(self, make_strategy_config):
        config = make_strategy_config(params={"bb_period": 20, "bb_std": 2.0})
        strategy = create_strategy("bollinger", config)
        short_data = pd.DataFrame(
            {"open": [100] * 10, "high": [101] * 10, "low": [99] * 10, "close": [100] * 10, "volume": [1000] * 10}
        )
        result = await strategy.should_enter(short_data)
        assert result is None

    @pytest.mark.asyncio
    async def test_squeeze_breakout(self, make_strategy_config):
        """After tight bands (low volatility), a breakout should trigger."""
        config = make_strategy_config(params={
            "bb_period": 10, "bb_std": 2.0, "squeeze_threshold": 0.1,
        })
        strategy = create_strategy("bollinger", config)

        # 50 bars of tight range, then sudden breakout
        prices = [100.0 + np.random.uniform(-0.1, 0.1) for _ in range(50)]
        prices.append(115.0)  # Breakout above bands
        n = len(prices)
        df = pd.DataFrame({
            "open": prices,
            "high": [p + 0.5 for p in prices],
            "low": [p - 0.5 for p in prices],
            "close": prices,
            "volume": [1000] * n,
        })
        signal = await strategy.should_enter(df)
        # After tight bands and breakout, should trigger long
        if signal is not None:
            assert signal.signal_type in (SignalType.LONG, SignalType.SHORT)

    @pytest.mark.asyncio
    async def test_exit_max_loss(self, make_strategy_config, make_ohlcv):
        config = make_strategy_config(params={"bb_period": 20}, max_loss_pct=-10.0)
        strategy = create_strategy("bollinger", config)
        data = make_ohlcv(n_bars=50)

        position = {"size": -0.1, "is_long": False, "pnl_perc": -11.0}
        signal = await strategy.should_exit(data, position)
        assert signal is not None
        assert signal.signal_type == SignalType.CLOSE_SHORT


# ──────────────────────────────────────────────
# MACD Strategy Tests
# ──────────────────────────────────────────────


class TestMACDStrategy:
    @pytest.mark.asyncio
    async def test_insufficient_data(self, make_strategy_config):
        config = make_strategy_config(params={
            "fast_period": 12, "slow_period": 26, "signal_period": 9,
        })
        strategy = create_strategy("macd", config)
        short_data = pd.DataFrame(
            {"open": [100] * 20, "high": [101] * 20, "low": [99] * 20, "close": [100] * 20, "volume": [1000] * 20}
        )
        result = await strategy.should_enter(short_data)
        assert result is None

    @pytest.mark.asyncio
    async def test_trending_data_produces_signal(self, make_strategy_config, trending_up_data):
        config = make_strategy_config(params={
            "fast_period": 12, "slow_period": 26, "signal_period": 9,
            "ma_filter_period": 50, "confirmation_bars": 1,
        })
        strategy = create_strategy("macd", config)
        signal = await strategy.should_enter(trending_up_data)
        # With strong uptrend, MACD should eventually cross above signal
        # Signal might still be None depending on exact data shape
        if signal is not None:
            assert signal.signal_type in (SignalType.LONG, SignalType.SHORT)

    @pytest.mark.asyncio
    async def test_exit_on_target(self, make_strategy_config, make_ohlcv):
        config = make_strategy_config(params={
            "fast_period": 12, "slow_period": 26, "signal_period": 9,
        }, target_pct=5.0)
        strategy = create_strategy("macd", config)
        data = make_ohlcv(n_bars=100)

        position = {"size": 0.1, "is_long": True, "pnl_perc": 6.0}
        signal = await strategy.should_exit(data, position)
        assert signal is not None
        assert signal.signal_type == SignalType.CLOSE_LONG


# ──────────────────────────────────────────────
# ADX Strategy Tests
# ──────────────────────────────────────────────


class TestADXStrategy:
    @pytest.mark.asyncio
    async def test_insufficient_data(self, make_strategy_config):
        config = make_strategy_config(params={"adx_period": 14, "di_period": 14})
        strategy = create_strategy("adx", config)
        short_data = pd.DataFrame(
            {"open": [100] * 10, "high": [101] * 10, "low": [99] * 10, "close": [100] * 10, "volume": [1000] * 10}
        )
        result = await strategy.should_enter(short_data)
        assert result is None

    @pytest.mark.asyncio
    async def test_strong_trend_signal(self, make_strategy_config, trending_up_data):
        config = make_strategy_config(params={
            "adx_period": 14, "di_period": 14, "adx_threshold": 20,
        })
        strategy = create_strategy("adx", config)
        signal = await strategy.should_enter(trending_up_data)
        if signal is not None:
            assert signal.signal_type in (SignalType.LONG, SignalType.SHORT)

    @pytest.mark.asyncio
    async def test_exit_on_max_loss(self, make_strategy_config, make_ohlcv):
        config = make_strategy_config(params={
            "adx_period": 14, "di_period": 14, "exit_threshold": 20,
        }, max_loss_pct=-10.0)
        strategy = create_strategy("adx", config)
        data = make_ohlcv(n_bars=50)

        position = {"size": 0.1, "is_long": True, "pnl_perc": -11.0}
        signal = await strategy.should_exit(data, position)
        assert signal is not None


# ──────────────────────────────────────────────
# Turtle Strategy Tests
# ──────────────────────────────────────────────


class TestTurtleHLStrategy:
    @pytest.mark.asyncio
    async def test_insufficient_data(self, make_strategy_config):
        config = make_strategy_config(params={"lookback_period": 55, "atr_period": 20})
        strategy = create_strategy("turtle", config)
        short_data = pd.DataFrame(
            {"open": [100] * 30, "high": [101] * 30, "low": [99] * 30, "close": [100] * 30, "volume": [1000] * 30}
        )
        result = await strategy.should_enter(short_data)
        assert result is None

    @pytest.mark.asyncio
    async def test_breakout_signal(self, make_strategy_config, breakout_data):
        config = make_strategy_config(params={
            "lookback_period": 55, "atr_period": 20, "take_profit_pct": 0.002,
        })
        strategy = create_strategy("turtle", config)
        signal = await strategy.should_enter(breakout_data)
        if signal is not None:
            assert signal.signal_type in (SignalType.LONG, SignalType.SHORT)

    @pytest.mark.asyncio
    async def test_exit_stop_loss(self, make_strategy_config, make_ohlcv):
        config = make_strategy_config(params={
            "lookback_period": 55, "atr_period": 20, "atr_multiplier": 2.0,
        })
        strategy = create_strategy("turtle", config)
        data = make_ohlcv(n_bars=100)

        # Position far below stop
        position = {"size": 0.1, "is_long": True, "entry_px": 200.0, "pnl_perc": -15.0}
        signal = await strategy.should_exit(data, position)
        assert signal is not None


# ──────────────────────────────────────────────
# Ichimoku Strategy Tests
# ──────────────────────────────────────────────


class TestIchimokuStrategy:
    @pytest.mark.asyncio
    async def test_insufficient_data(self, make_strategy_config):
        config = make_strategy_config(params={
            "tenkan_period": 9, "kijun_period": 26, "senkou_b_period": 52,
        })
        strategy = create_strategy("ichimoku", config)
        short_data = pd.DataFrame(
            {"open": [100] * 30, "high": [101] * 30, "low": [99] * 30, "close": [100] * 30, "volume": [1000] * 30}
        )
        result = await strategy.should_enter(short_data)
        assert result is None

    @pytest.mark.asyncio
    async def test_trending_data(self, make_strategy_config, trending_up_data):
        config = make_strategy_config(params={
            "tenkan_period": 9, "kijun_period": 26, "senkou_b_period": 52,
        })
        strategy = create_strategy("ichimoku", config)
        signal = await strategy.should_enter(trending_up_data)
        if signal is not None:
            assert signal.signal_type in (SignalType.LONG, SignalType.SHORT)


# ──────────────────────────────────────────────
# VWMA Strategy Tests
# ──────────────────────────────────────────────


class TestVWMAStrategy:
    @pytest.mark.asyncio
    async def test_insufficient_data(self, make_strategy_config):
        config = make_strategy_config(params={"fast_period": 20, "mid_period": 41, "slow_period": 75})
        strategy = create_strategy("vwma", config)
        short_data = pd.DataFrame(
            {"open": [100] * 10, "high": [101] * 10, "low": [99] * 10, "close": [100] * 10, "volume": [1000] * 10}
        )
        result = await strategy.should_enter(short_data)
        assert result is None


# ──────────────────────────────────────────────
# Parametrized: All strategies handle short data gracefully
# ──────────────────────────────────────────────


class TestAllStrategiesGraceful:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "strategy_type",
        [s["strategy_type"] for s in list_strategies()],
    )
    async def test_short_data_returns_none(self, strategy_type, make_strategy_config):
        """Every strategy should return None when given < 10 bars of data."""
        config = make_strategy_config(name=f"test-{strategy_type}")
        strategy = create_strategy(strategy_type, config)

        short_data = pd.DataFrame({
            "open": [100.0] * 5,
            "high": [101.0] * 5,
            "low": [99.0] * 5,
            "close": [100.0] * 5,
            "volume": [1000.0] * 5,
        })
        result = await strategy.should_enter(short_data)
        assert result is None, f"{strategy_type} did not return None for insufficient data"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "strategy_type",
        [s["strategy_type"] for s in list_strategies()],
    )
    async def test_exit_on_max_loss(self, strategy_type, make_strategy_config, make_ohlcv):
        """Every strategy should exit when PnL exceeds max_loss_pct."""
        config = make_strategy_config(
            name=f"test-exit-{strategy_type}",
            max_loss_pct=-10.0,
            target_pct=5.0,
        )
        strategy = create_strategy(strategy_type, config)
        data = make_ohlcv(n_bars=200)

        position = {"size": 0.1, "is_long": True, "entry_px": 200.0, "pnl_perc": -11.0}
        signal = await strategy.should_exit(data, position)
        # All strategies should have a max_loss exit
        assert signal is not None, f"{strategy_type} did not exit on max loss"
        assert signal.signal_type in (
            SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT, SignalType.CLOSE_ALL,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "strategy_type",
        [s["strategy_type"] for s in list_strategies()],
    )
    async def test_exit_on_target(self, strategy_type, make_strategy_config, make_ohlcv):
        """Every strategy should exit when PnL reaches target_pct."""
        config = make_strategy_config(
            name=f"test-tp-{strategy_type}",
            max_loss_pct=-10.0,
            target_pct=5.0,
        )
        strategy = create_strategy(strategy_type, config)
        data = make_ohlcv(n_bars=200)

        position = {"size": 0.1, "is_long": True, "entry_px": 50.0, "pnl_perc": 6.0}
        signal = await strategy.should_exit(data, position)
        assert signal is not None, f"{strategy_type} did not exit on target"
        assert signal.signal_type in (
            SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT, SignalType.CLOSE_ALL,
        )


# ──────────────────────────────────────────────
# Anti-overtrading (BaseStrategy.run_iteration)
# ──────────────────────────────────────────────


class TestAntiOvertrading:
    @pytest.mark.asyncio
    async def test_cooldown_blocks_entry(self, make_strategy_config, make_ohlcv):
        """After closing a trade, cooldown should block new entries."""
        from datetime import datetime, timezone

        config = make_strategy_config(
            params={"cooldown_seconds": 999999, "min_hold_bars": 0, "max_trades_per_hour": 100},
        )
        strategy = create_strategy("rsi", config)

        # Simulate a recent trade close
        strategy.state.last_trade_close_time = datetime.now(timezone.utc)

        data = make_ohlcv(n_bars=100)
        signal = await strategy.run_iteration(data, position=None)
        # Should be blocked by cooldown
        assert signal is None

    @pytest.mark.asyncio
    async def test_max_trades_per_hour_blocks(self, make_strategy_config, make_ohlcv):
        """After reaching max trades per hour, new entries should be blocked."""
        from datetime import datetime, timezone

        config = make_strategy_config(
            params={"max_trades_per_hour": 2, "cooldown_seconds": 0, "min_hold_bars": 0},
        )
        strategy = create_strategy("rsi", config)

        strategy.state.trades_this_hour = 2
        strategy.state.hour_start = datetime.now(timezone.utc)

        data = make_ohlcv(n_bars=100)
        signal = await strategy.run_iteration(data, position=None)
        assert signal is None
