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
        assert "test-paper:BTC" in executor._positions
        pos = executor._positions["test-paper:BTC"]
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
        pos = executor._positions["test-paper:BTC"]
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
        assert "test-paper:BTC" not in executor._positions
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
        pos = await executor.get_position("BTC", strategy_name="test-paper")
        assert pos is None

    @pytest.mark.asyncio
    async def test_get_position_with_unrealized_pnl(self, executor, strategy):
        signal = Signal(signal_type=SignalType.LONG, symbol="BTC", size_usd=1000.0, reason="test")
        await executor.execute_signal(signal, strategy)

        # Price unchanged
        pos = await executor.get_position("BTC", strategy_name="test-paper")
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

        # Create SOL strategy (a DIFFERENT correlation cluster from BTC, so the
        # U6 cluster cap allows both to open — BTC+ETH would be one 'majors' cluster).
        sol_config = StrategyConfig(name="test-sol", symbol="SOL", size_usd=500.0)
        sol_strategy = create_strategy("rsi", sol_config)
        signal_sol = Signal(signal_type=SignalType.SHORT, symbol="SOL", size_usd=500.0, reason="test")
        await executor.execute_signal(signal_sol, sol_strategy)

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
        # Should be close to initial balance (minus commission: 3.5% of $1000 = $35)
        assert abs(value - 10000.0) < 40.0


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
        # Regression 2026-07-01: total_realized_pnl must be balance-delta, not
        # a sum over self._trades — that undercounts vs true lifetime balance
        # once the trade ledger is older than the durable JSONL seed point.
        assert stats["total_realized_pnl"] == round(stats["balance"] - stats["initial_balance"], 2)
        assert stats["total_pnl"] == stats["total_realized_pnl"]


class TestPaperExecutorOrphanFlush:
    """Closing positions held by a strategy that is being disabled.

    Regression for 2026-05-15: PATCH enabled=false on a strategy leaves
    its open positions orphaned — orchestrator skips the strategy on the
    next polling cycle, so should_exit() is never called and the
    positions drift indefinitely. close_by_strategy() lets the disable
    flow flush positions atomically.
    """

    @pytest.mark.asyncio
    async def test_close_by_strategy_closes_only_named_strategy(self, executor, strategy):
        # Open a position for "test-paper" (the strategy fixture's name)
        signal_a = Signal(signal_type=SignalType.LONG, symbol="BTC", size_usd=500.0, reason="a-entry")
        await executor.execute_signal(signal_a, strategy)

        # Open a position for "other-strat" on SOL (a different correlation cluster
        # from BTC, so the U6 cluster cap permits the second open).
        other_config = StrategyConfig(name="other-strat", symbol="SOL", size_usd=500.0)
        other = create_strategy("rsi", other_config)
        signal_b = Signal(signal_type=SignalType.SHORT, symbol="SOL", size_usd=500.0, reason="b-entry")
        await executor.execute_signal(signal_b, other)

        assert len(executor._positions) == 2

        results = await executor.close_by_strategy("test-paper")

        # test-paper position closed, other-strat untouched
        assert len(results) == 1
        assert results[0].success is True
        remaining = {pos.strategy_name for pos in executor._positions.values()}
        assert remaining == {"other-strat"}

    @pytest.mark.asyncio
    async def test_close_by_strategy_records_exit_trade(self, executor, strategy):
        signal = Signal(signal_type=SignalType.LONG, symbol="BTC", size_usd=500.0, reason="enter")
        await executor.execute_signal(signal, strategy)

        entry_trade_count = sum(1 for t in executor._trades if t.action == "entry")
        await executor.close_by_strategy("test-paper")

        exit_trades = [t for t in executor._trades if t.action == "exit"]
        assert len(exit_trades) == 1
        assert exit_trades[0].strategy_name == "test-paper"
        assert exit_trades[0].reason == "strategy_disabled"

    @pytest.mark.asyncio
    async def test_close_by_strategy_no_positions_is_noop(self, executor):
        results = await executor.close_by_strategy("nonexistent")
        assert results == []
        assert len(executor._positions) == 0


def _seed_full_confidence(executor, strategy_name, n=30, pnl=50.0):
    """Seed n winning exit trades so the strategy reaches FULL tier (kelly_mult=1.0)."""
    from src.execution.paper_executor import PaperTrade
    for _ in range(n):
        executor._trades.append(PaperTrade(
            id=0, symbol="BTC", side="long", action="exit",
            price=100.0, size=1.0, size_usd=50.0,
            pnl=pnl, pnl_pct=pnl / 50.0 * 100,
            reason="seed", strategy_name=strategy_name,
        ))


class TestCompoundSizing:
    """Verify compound position sizing grows with balance.

    Each test seeds >=30 winning trades for the strategy so the confidence
    ladder enters FULL tier (kelly_mult=1.0, effective_leverage=40 for BTC).

    After the leverage/PnL fix (Task 2.5): size_usd is the TRUE leveraged
    notional (base * compound * kelly * leverage), capped at max_position_usd.
    With BTC at 40x and max_position_usd=10000 (fixture), the leveraged
    notional exceeds the cap in all three cases below, so:
      - pos.size_usd == max_position_usd == 10000 (cap binds for all)
      - margin = size_usd / leverage = 10000/40 = 250 (correct; balance check uses this)
      - compound_mult isolation is confirmed via the uncapped pre-leverage computation
        asserted inline.
    """

    @pytest.mark.asyncio
    async def test_compound_mult_at_par(self, executor, strategy):
        """At initial balance, compound_mult=1.0 (after full confidence).
        Leveraged notional = 1000 * 1.0 * 1.0 * 40 = 40000 → capped at 10000."""
        _seed_full_confidence(executor, "test-paper")
        signal = Signal(signal_type=SignalType.LONG, symbol="BTC", size_usd=1000.0, reason="test")
        result = await executor.execute_signal(signal, strategy)
        assert result.success is True
        pos = executor._positions["test-paper:BTC"]
        # leveraged notional (1000 * 40 = 40000) exceeds max_position_usd (10000); cap binds
        assert pos.size_usd == 10000.0  # == max_position_usd (cap)
        assert pos.leverage == 40       # BTC full-confidence leverage
        # margin = notional / leverage = 10000/40 = 250; well within balance of 10000
        assert abs(pos.size_usd / pos.leverage - 250.0) < 0.01

    @pytest.mark.asyncio
    async def test_compound_mult_scales_with_profit(self, executor, strategy):
        """When balance is 20% above initial, compound_mult=1.2 (after full confidence).
        Leveraged notional = 1000 * 1.2 * 1.0 * 40 = 48000 → capped at 10000."""
        _seed_full_confidence(executor, "test-paper")
        executor.balance = 12000.0  # 20% profit
        signal = Signal(signal_type=SignalType.LONG, symbol="BTC", size_usd=1000.0, reason="test")
        result = await executor.execute_signal(signal, strategy)
        assert result.success is True
        pos = executor._positions["test-paper:BTC"]
        # leveraged notional (1200 * 40 = 48000) exceeds max_position_usd (10000); cap binds
        assert pos.size_usd == 10000.0  # == max_position_usd (cap)
        assert pos.leverage == 40
        # margin = 10000/40 = 250; balance 12000 ≫ 250
        assert abs(pos.size_usd / pos.leverage - 250.0) < 0.01

    @pytest.mark.asyncio
    async def test_compound_mult_capped_at_3x(self, executor, strategy):
        """Compound mult caps at 3× regardless of balance growth (after full confidence).
        Leveraged notional = 1000 * 3.0 * 1.0 * 40 = 120000 → capped at 10000."""
        _seed_full_confidence(executor, "test-paper")
        executor.balance = 50000.0  # 5× initial — compound caps at 3×
        signal = Signal(signal_type=SignalType.LONG, symbol="BTC", size_usd=1000.0, reason="test")
        result = await executor.execute_signal(signal, strategy)
        assert result.success is True
        pos = executor._positions["test-paper:BTC"]
        # leveraged notional (3000 * 40 = 120000) exceeds max_position_usd (10000); cap binds
        assert pos.size_usd == 10000.0  # == max_position_usd (cap)
        assert pos.leverage == 40
        # margin = 10000/40 = 250; balance 50000 ≫ 250
        assert abs(pos.size_usd / pos.leverage - 250.0) < 0.01


class TestLeveragePnLInvariant:
    """Task 2.5: verify the single-count PnL model is self-consistent.

    Two key invariants:
    1. No double-count: realized balance delta == price_move_fraction * notional (NOT * leverage again).
    2. Unrealized == realized: get_account_value() unrealized at some mid equals the
       balance change from closing at that exact mid — the guarantee the ruin guards depend on.

    Uses a high max_position_usd so the leverage cap does NOT bind, making
    expected values exact.
    """

    @pytest.fixture
    def ex_uncapped(self):
        """Executor with large position cap and no slippage/commission for clean arithmetic."""
        ex = PaperTradingExecutor(
            base_url="https://api.hyperliquid.xyz",
            default_slippage=0.0,      # no slippage: fill at mid
            max_position_usd=500000.0, # cap won't bind at typical sizes
            initial_balance=100000.0,
            commission_pct=0.0,        # no commission: isolate PnL
        )
        ex._mid_prices = {"BTC": 50000.0}
        ex._last_price_fetch = 9999999999.0
        return ex

    @pytest.fixture
    def strategy_uncapped(self):
        config = StrategyConfig(name="inv-test", symbol="BTC", tier=StrategyTier.C, leverage=1, size_usd=1000.0)
        return create_strategy("rsi", config)

    @pytest.mark.asyncio
    async def test_no_double_count_realized(self, ex_uncapped, strategy_uncapped):
        """Realized balance delta must equal price_move_fraction * leveraged_notional.

        Setup: seed 30 wins so effective_leverage == 40 (BTC full tier).
        base_usd=1000, compound=1.0, kelly=1.0 → pre-leverage=1000, notional=40000.
        Move price +10%: expected balance gain = 0.10 * 40000 = 4000 (NOT 4000*40).
        """
        _seed_full_confidence(ex_uncapped, "inv-test")
        balance_before = ex_uncapped.balance  # 100000.0

        signal = Signal(signal_type=SignalType.LONG, symbol="BTC", size_usd=1000.0, reason="enter")
        result = await ex_uncapped.execute_signal(signal, strategy_uncapped)
        assert result.success, result.error

        pos = ex_uncapped._positions["inv-test:BTC"]
        leveraged_notional = pos.size_usd  # should be 1000 * 40 = 40000 (cap not binding)
        assert abs(leveraged_notional - 40000.0) < 1.0, f"Expected notional 40000, got {leveraged_notional}"
        assert pos.leverage == 40

        balance_after_entry = ex_uncapped.balance  # no commission → unchanged
        assert balance_after_entry == balance_before  # commission=0 so no deduction

        # Move price +10%
        ex_uncapped._mid_prices["BTC"] = 55000.0
        exit_signal = Signal(signal_type=SignalType.CLOSE_LONG, symbol="BTC", reason="exit")
        exit_result = await ex_uncapped.execute_signal(exit_signal, strategy_uncapped)
        assert exit_result.success

        balance_after_exit = ex_uncapped.balance
        balance_delta = balance_after_exit - balance_after_entry
        expected_delta = 0.10 * leveraged_notional  # = 4000.0 — ONE count, no further *leverage
        assert abs(balance_delta - expected_delta) < 0.01, (
            f"Expected balance delta {expected_delta}, got {balance_delta}. "
            "Double-count would produce {expected_delta * 40}."
        )

    @pytest.mark.asyncio
    async def test_unrealized_matches_realized(self, ex_uncapped, strategy_uncapped):
        """Unrealized PnL from get_account_value() must match the realized balance change
        when closing at the same price. This is the ruin-guard consistency guarantee.
        """
        _seed_full_confidence(ex_uncapped, "inv-test")

        signal = Signal(signal_type=SignalType.LONG, symbol="BTC", size_usd=1000.0, reason="enter")
        await ex_uncapped.execute_signal(signal, strategy_uncapped)
        balance_after_entry = ex_uncapped.balance

        # Move price to some mid
        ex_uncapped._mid_prices["BTC"] = 52000.0

        # Capture unrealized account value
        account_value_at_52k = await ex_uncapped.get_account_value()
        unrealized_delta = account_value_at_52k - balance_after_entry  # should be +ve for long

        # Now close at that same price
        exit_signal = Signal(signal_type=SignalType.CLOSE_LONG, symbol="BTC", reason="exit")
        await ex_uncapped.execute_signal(exit_signal, strategy_uncapped)
        balance_after_exit = ex_uncapped.balance
        realized_delta = balance_after_exit - balance_after_entry

        # Unrealized delta at mid must equal realized delta when closed at that mid
        assert abs(unrealized_delta - realized_delta) < 0.01, (
            f"Unrealized={unrealized_delta:.4f} != Realized={realized_delta:.4f}. "
            "Divergence means ruin guard reads wrong account value."
        )
