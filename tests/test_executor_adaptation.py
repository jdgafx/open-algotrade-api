"""TDD tests for Task 5a.2: realtime adaptation multiplier applied at entry.

Signal.metadata["adaptation_multiplier"] scales the confidence-ladder leverage
at entry time. Default 1.0 when absent — existing callers unaffected.
"""

import pytest
from src.execution.paper_executor import PaperTradingExecutor, PaperTrade
from src.services.confidence_ladder import HL_MAX_LEVERAGE


def _winning_exit(strategy: str) -> PaperTrade:
    return PaperTrade(
        id=0, symbol="BTC", side="long", action="exit",
        price=50000.0, size=0.001, size_usd=50.0,
        pnl=50.0, pnl_pct=100.0, reason="seed", strategy_name=strategy,
    )


def _make_full_tier_executor(strategy_name: str) -> PaperTradingExecutor:
    """Seed 30 winning BTC exit trades so the strategy is FULL tier (ladder == 40)."""
    ex = PaperTradingExecutor(initial_balance=50_000.0)
    ex._mid_prices = {"BTC": 50_000.0}
    ex._last_price_fetch = 9_999_999_999.0
    for _ in range(30):
        ex._trades.append(_winning_exit(strategy_name))
    return ex


@pytest.mark.asyncio
async def test_adaptation_multiplier_halves_leverage():
    """metadata={"adaptation_multiplier": 0.5} on a FULL-tier BTC strategy
    must halve the ladder leverage: round(40 * 0.5) = 20."""
    from src.strategies.base_strategy import Signal, SignalType, StrategyConfig, StrategyTier
    from src.strategies.registry import create_strategy

    ex = _make_full_tier_executor("adapt-half-btc")
    config = StrategyConfig(
        name="adapt-half-btc", symbol="BTC", tier=StrategyTier.A,
        leverage=5, size_usd=1000.0,
    )
    strategy = create_strategy("rsi", config)
    signal = Signal(
        signal_type=SignalType.LONG, symbol="BTC", size_usd=1000.0,
        reason="adapt-test",
        metadata={"adaptation_multiplier": 0.5},
    )
    result = await ex.execute_signal(signal, strategy)

    assert result.success is True, f"Entry failed: {result.error}"
    pos = ex._positions["adapt-half-btc:BTC"]
    assert pos.leverage == 20, f"Expected 20 (round(40*0.5)), got {pos.leverage}"


@pytest.mark.asyncio
async def test_adaptation_multiplier_capped_at_hl_max():
    """metadata={"adaptation_multiplier": 2.0} must cap at HL_MAX_LEVERAGE["BTC"]=40,
    not produce 80."""
    from src.strategies.base_strategy import Signal, SignalType, StrategyConfig, StrategyTier
    from src.strategies.registry import create_strategy

    ex = _make_full_tier_executor("adapt-double-btc")
    config = StrategyConfig(
        name="adapt-double-btc", symbol="BTC", tier=StrategyTier.A,
        leverage=5, size_usd=1000.0,
    )
    strategy = create_strategy("rsi", config)
    signal = Signal(
        signal_type=SignalType.LONG, symbol="BTC", size_usd=1000.0,
        reason="adapt-test",
        metadata={"adaptation_multiplier": 2.0},
    )
    result = await ex.execute_signal(signal, strategy)

    assert result.success is True, f"Entry failed: {result.error}"
    pos = ex._positions["adapt-double-btc:BTC"]
    btc_cap = HL_MAX_LEVERAGE.get("BTC", 5)
    assert pos.leverage == btc_cap, (
        f"Expected {btc_cap} (capped at HL max, not 80), got {pos.leverage}"
    )


@pytest.mark.asyncio
async def test_adaptation_multiplier_absent_defaults_to_one():
    """No metadata key at all -> default 1.0 -> leverage == 40 (unchanged FULL tier)."""
    from src.strategies.base_strategy import Signal, SignalType, StrategyConfig, StrategyTier
    from src.strategies.registry import create_strategy

    ex = _make_full_tier_executor("adapt-default-btc")
    config = StrategyConfig(
        name="adapt-default-btc", symbol="BTC", tier=StrategyTier.A,
        leverage=5, size_usd=1000.0,
    )
    strategy = create_strategy("rsi", config)
    # metadata defaults to {} — no adaptation_multiplier key
    signal = Signal(
        signal_type=SignalType.LONG, symbol="BTC", size_usd=1000.0,
        reason="adapt-test",
    )
    result = await ex.execute_signal(signal, strategy)

    assert result.success is True, f"Entry failed: {result.error}"
    pos = ex._positions["adapt-default-btc:BTC"]
    assert pos.leverage == 40, f"Expected 40 (default 1.0 multiplier), got {pos.leverage}"


@pytest.mark.asyncio
async def test_adaptation_multiplier_none_metadata_defaults_to_one():
    """metadata=None (explicit None) -> default 1.0 -> leverage == 40."""
    from src.strategies.base_strategy import Signal, SignalType, StrategyConfig, StrategyTier
    from src.strategies.registry import create_strategy

    ex = _make_full_tier_executor("adapt-none-btc")
    config = StrategyConfig(
        name="adapt-none-btc", symbol="BTC", tier=StrategyTier.A,
        leverage=5, size_usd=1000.0,
    )
    strategy = create_strategy("rsi", config)
    signal = Signal(
        signal_type=SignalType.LONG, symbol="BTC", size_usd=1000.0,
        reason="adapt-test",
    )
    signal.metadata = None  # explicitly None to exercise the `or {}` guard
    result = await ex.execute_signal(signal, strategy)

    assert result.success is True, f"Entry failed: {result.error}"
    pos = ex._positions["adapt-none-btc:BTC"]
    assert pos.leverage == 40, f"Expected 40 (None metadata defaults to 1.0), got {pos.leverage}"
