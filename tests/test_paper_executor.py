"""
Tests for the PaperTradingExecutor.

Tests cover:
- Entry/exit execution
- Balance tracking and commission
- Position management
- PnL calculation
- Reset functionality
- Edge cases (insufficient balance, duplicate positions)
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from src.execution.paper_executor import (
    PaperTradingExecutor,
    ExecutionResult,
    PaperOrderResult,
)
from src.strategies.base_strategy import (
    BaseStrategy,
    Signal,
    SignalType,
    StrategyConfig,
    StrategyTier,
)
from src.strategies.registry import create_strategy


@pytest.fixture
def executor():
    """Create a paper trading executor with mocked price fetching."""
    ex = PaperTradingExecutor(
        base_url="https://api.hyperliquid.xyz",
        default_slippage=0.001,
        max_position_usd=10000.0,
        initial_balance=10000.0,
        commission_pct=0.035,
    )
    # Mock the price fetch to avoid network calls
    ex._mid_prices = {"BTC": 50000.0, "ETH": 3000.0, "SOL": 100.0}
    ex._last_price_fetch = 9999999999.0  # Far future so cache is always valid
    return ex


@pytest.fixture
def strategy():
    """Create a simple strategy for testing."""
    config = StrategyConfig(
        name="test-paper",
        symbol="BTC",
        tier=StrategyTier.C,
        leverage=2,
        size_usd=1000.0,
        target_pct=5.0,
        max_loss_pct=-10.0,
    )
    return create_strategy("rsi", config)


class TestPaperExecutorInit:
    def test_initial_state(self, executor):
        assert executor.balance == 10000.0
        assert executor.initial_balance == 10000.0
        assert executor.peak_balance == 10000.0
        assert len(executor._positions) == 0
        assert len(executor._trades) == 0
        assert executor.vault_address == "paper-trading"

    def test_execution_stats_empty(self, executor):
        stats = executor.get_execution_stats()
        assert stats["mode"] == "paper"
        assert stats["total_executions"] == 0
        assert stats["balance"] == 10000.0


class TestPaperExecutorEntry:
    @pytest.mark.asyncio
    async def test_long_entry(self, executor, strategy):
        signal = Signal(
            signal_type=SignalType.LONG,
            symbol="BTC",
            size_usd=1000.0,
            reason="test long",
        )
        result = await executor.execute_signal(signal, strategy)

        assert result.success is True
        assert "BTC" in executor._positions
        pos = executor._positions["BTC"]
        assert pos.side == "long"
        assert pos.size > 0
        assert pos.strategy_name == "test-paper"
        assert len(executor._trades) == 1
        assert executor._trades[0].action == "entry"
        # Balance should have decreased by commission
        assert executor.balance < 10000.0

    @pytest.mark.asyncio
    async def test_short_entry(self, executor, strategy):
        signal = Signal(
            signal_type=SignalType.SHORT,
            symbol="BTC",
            size_usd=500.0,
            reason="test short",
        )
        result = await executor.execute_signal(signal, strategy)

        assert result.success is True
        pos = executor._positions["BTC"]
        assert pos.side == "short"
        assert pos.size < 0

    @pytest.mark.asyncio
    async def test_duplicate_entry_fails(self, executor, strategy):
        signal = Signal(signal_type=SignalType.LONG, symbol="BTC", reason="first")
        await executor.execute_signal(signal, strategy)

        signal2 = Signal(signal_type=SignalType.LONG, symbol="BTC", reason="second")
        result = await executor.execute_signal(signal2, strategy)
        assert result.success is False
        assert "Already in position" in result.error

    @pytest.mark.asyncio
    async def test_insufficient_balance(self, executor, strategy):
        executor.balance = 1.0  # Very low balance

        signal = Signal(
            signal_type=SignalType.LONG,
            symbol="BTC",
            size_usd=1000.0,
            reason="test",
        )
        result = await executor.execute_signal(signal, strategy)
        assert result.success is False
        assert "Insufficient balance" in result.error

    @pytest.mark.asyncio
    async def test_none_signal_succeeds(self, executor, strategy):
        signal = Signal(signal_type=SignalType.NONE, symbol="BTC", reason="no-op")
        result = await executor.execute_signal(signal, strategy)
        assert result.success is True
        assert len(executor._positions) == 0


class TestPaperExecutorExit:
    @pytest.mark.asyncio
    async def test_long_exit_with_profit(self, executor, strategy):
        # Enter long
        entry_signal = Signal(signal_type=SignalType.LONG, symbol="BTC", size_usd=1000.0, reason="enter")
        await executor.execute_signal(entry_signal, strategy)
        balance_after_entry = executor.balance

        # Simulate price increase
        executor._mid_prices["BTC"] = 55000.0

        exit_signal = Signal(signal_type=SignalType.CLOSE_LONG, symbol="BTC", reason="take profit")
        result = await executor.execute_signal(exit_signal, strategy)

        assert result.success is True
        assert result.realized_pnl > 0
        assert executor.balance > balance_after_entry
        assert "BTC" not in executor._positions
        assert len(executor._trades) == 2  # entry + exit

    @pytest.mark.asyncio
    async def test_long_exit_with_loss(self, executor, strategy):
        entry_signal = Signal(signal_type=SignalType.LONG, symbol="BTC", size_usd=1000.0, reason="enter")
        await executor.execute_signal(entry_signal, strategy)
        balance_after_entry = executor.balance

        # Simulate price decrease
        executor._mid_prices["BTC"] = 45000.0

        exit_signal = Signal(signal_type=SignalType.CLOSE_LONG, symbol="BTC", reason="stop loss")
        result = await executor.execute_signal(exit_signal, strategy)

        assert result.success is True
        assert result.realized_pnl < 0
        assert executor.balance < balance_after_entry

    @pytest.mark.asyncio
    async def test_short_exit_with_profit(self, executor, strategy):
        entry_signal = Signal(signal_type=SignalType.SHORT, symbol="BTC", size_usd=1000.0, reason="enter")
        await executor.execute_signal(entry_signal, strategy)

        # Simulate price decrease (profit for short)
        executor._mid_prices["BTC"] = 45000.0

        exit_signal = Signal(signal_type=SignalType.CLOSE_SHORT, symbol="BTC", reason="take profit")
        result = await executor.execute_signal(exit_signal, strategy)

        assert result.success is True
        assert result.realized_pnl > 0

    @pytest.mark.asyncio
    async def test_exit_no_position(self, executor, strategy):
        exit_signal = Signal(signal_type=SignalType.CLOSE_LONG, symbol="BTC", reason="no position")
        result = await executor.execute_signal(exit_signal, strategy)
        assert result.success is True
        assert result.error == "No position to close"


class TestPaperExecutorPosition:
    @pytest.mark.asyncio
    async def test_get_position_empty(self, executor):
        pos = await executor.get_position("BTC")
        assert pos is None

    @pytest.mark.asyncio
    async def test_get_position_with_unrealized_pnl(self, executor, strategy):
        signal = Signal(signal_type=SignalType.LONG, symbol="BTC", size_usd=1000.0, reason="test")
        await executor.execute_signal(signal, strategy)

        # Price unchanged
        pos = await executor.get_position("BTC")
        assert pos is not None
        assert pos["symbol"] == "BTC"
        assert pos["side"] == "long"
        assert "unrealized_pnl" in pos
        assert "pnl_perc" in pos

    @pytest.mark.asyncio
    async def test_get_all_positions(self, executor, strategy):
        # Open BTC position
        signal_btc = Signal(signal_type=SignalType.LONG, symbol="BTC", size_usd=500.0, reason="test")
        await executor.execute_signal(signal_btc, strategy)

        # Create ETH strategy
        eth_config = StrategyConfig(name="test-eth", symbol="ETH", size_usd=500.0)
        eth_strategy = create_strategy("rsi", eth_config)
        signal_eth = Signal(signal_type=SignalType.SHORT, symbol="ETH", size_usd=500.0, reason="test")
        await executor.execute_signal(signal_eth, eth_strategy)

        positions = await executor.get_all_positions()
        assert len(positions) == 2

    @pytest.mark.asyncio
    async def test_get_active_positions_dict(self, executor, strategy):
        signal = Signal(signal_type=SignalType.LONG, symbol="BTC", size_usd=1000.0, reason="test")
        await executor.execute_signal(signal, strategy)

        active = executor.get_active_positions()
        assert "BTC" in active
        assert active["BTC"]["strategy"] == "test-paper"
        assert active["BTC"]["side"] == "long"

    @pytest.mark.asyncio
    async def test_get_account_value(self, executor, strategy):
        value = await executor.get_account_value()
        assert value == 10000.0  # No positions

        signal = Signal(signal_type=SignalType.LONG, symbol="BTC", size_usd=1000.0, reason="test")
        await executor.execute_signal(signal, strategy)
        value = await executor.get_account_value()
        # Should be close to initial balance (minus small commission)
        assert abs(value - 10000.0) < 10.0


class TestPaperExecutorHistory:
    @pytest.mark.asyncio
    async def test_trade_history(self, executor, strategy):
        signal = Signal(signal_type=SignalType.LONG, symbol="BTC", size_usd=1000.0, reason="test")
        await executor.execute_signal(signal, strategy)

        history = executor.get_trade_history()
        assert len(history) == 1
        assert history[0]["symbol"] == "BTC"
        assert history[0]["action"] == "entry"
        assert history[0]["strategy"] == "test-paper"
        assert "timestamp" in history[0]

    @pytest.mark.asyncio
    async def test_equity_curve(self, executor, strategy):
        # Enter and exit to create equity curve points
        signal = Signal(signal_type=SignalType.LONG, symbol="BTC", size_usd=1000.0, reason="enter")
        await executor.execute_signal(signal, strategy)

        executor._mid_prices["BTC"] = 55000.0
        exit_signal = Signal(signal_type=SignalType.CLOSE_LONG, symbol="BTC", reason="exit")
        await executor.execute_signal(exit_signal, strategy)

        curve = executor.get_equity_curve()
        assert len(curve) >= 1
        assert curve[0]["equity"] == 10000.0  # Initial point


class TestPaperExecutorReset:
    @pytest.mark.asyncio
    async def test_reset_clears_state(self, executor, strategy):
        signal = Signal(signal_type=SignalType.LONG, symbol="BTC", size_usd=1000.0, reason="test")
        await executor.execute_signal(signal, strategy)

        assert len(executor._positions) > 0
        assert len(executor._trades) > 0

        executor.reset()

        assert executor.balance == executor.initial_balance
        assert len(executor._positions) == 0
        assert len(executor._trades) == 0
        assert executor._trade_counter == 0
        assert executor.peak_balance == executor.initial_balance


class TestPaperExecutorEmergencyClose:
    @pytest.mark.asyncio
    async def test_emergency_close_all(self, executor, strategy):
        signal = Signal(signal_type=SignalType.LONG, symbol="BTC", size_usd=1000.0, reason="test")
        await executor.execute_signal(signal, strategy)
        assert len(executor._positions) == 1

        results = await executor.emergency_close_all()
        assert len(results) == 1
        assert results[0].success is True
        assert len(executor._positions) == 0


class TestPaperExecutorPnLGuard:
    @pytest.mark.asyncio
    async def test_pnl_guard_no_trigger(self, executor, strategy):
        signal = Signal(signal_type=SignalType.LONG, symbol="BTC", size_usd=1000.0, reason="test")
        await executor.execute_signal(signal, strategy)

        result = await executor.check_pnl_guard("BTC", target_pct=50.0, max_loss_pct=-50.0)
        assert result is None

    @pytest.mark.asyncio
    async def test_pnl_guard_no_position(self, executor):
        result = await executor.check_pnl_guard("BTC")
        assert result is None


class TestPaperExecutorStats:
    @pytest.mark.asyncio
    async def test_stats_after_trades(self, executor, strategy):
        signal = Signal(signal_type=SignalType.LONG, symbol="BTC", size_usd=1000.0, reason="enter")
        await executor.execute_signal(signal, strategy)

        executor._mid_prices["BTC"] = 55000.0
        exit_signal = Signal(signal_type=SignalType.CLOSE_LONG, symbol="BTC", reason="exit")
        await executor.execute_signal(exit_signal, strategy)

        stats = executor.get_execution_stats()
        assert stats["mode"] == "paper"
        assert stats["total_executions"] == 2
        assert stats["successful"] == 2
        assert stats["total_trades"] == 2
        assert stats["balance"] > 10000.0  # Profitable trade
        assert stats["total_return_pct"] > 0
