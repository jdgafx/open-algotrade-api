"""TDD tests for _live_edge_stats helper and confidence-ladder entry sizing.

Step 1: tests for the helper (fail until implemented).
Step 2: new-behavior assertions for the ladder-at-entry integration.
"""

import pytest
from src.execution.paper_executor import PaperTradingExecutor, PaperTrade


def _exit_trade(strategy, pnl):
    return PaperTrade(id=0, symbol="BTC", side="long", action="exit", price=100.0,
                      size=1.0, size_usd=50.0, pnl=pnl, pnl_pct=pnl / 50.0 * 100,
                      reason="t", strategy_name=strategy)


def test_live_edge_stats_from_trade_history():
    ex = PaperTradingExecutor(initial_balance=200.0)
    for _ in range(6):
        ex._trades.append(_exit_trade("donchian-btc", 20.0))   # 6 wins +20
    for _ in range(4):
        ex._trades.append(_exit_trade("donchian-btc", -10.0))  # 4 losses -10
    wr, payoff, n, *_ = ex._live_edge_stats("donchian-btc")
    assert n == 10
    assert wr == pytest.approx(0.6)
    assert payoff == pytest.approx(2.0)


def test_live_edge_stats_ignores_other_strategies_and_entries():
    ex = PaperTradingExecutor(initial_balance=200.0)
    ex._trades.append(_exit_trade("other", 99.0))
    entry = _exit_trade("donchian-btc", 0.0)
    entry.action = "entry"
    ex._trades.append(entry)
    wr, payoff, n, *_ = ex._live_edge_stats("donchian-btc")
    assert n == 0


def test_live_edge_stats_caps_payoff_when_no_losses():
    """Wins + breakevens but ZERO losses must yield a dimensionless capped payoff,
    not a raw dollar avg_win. Otherwise Kelly gets a wrong/negative ratio b."""
    from src.services.kelly import half_kelly_fraction
    ex = PaperTradingExecutor(initial_balance=200.0)
    for _ in range(6):
        ex._trades.append(_exit_trade("breakeven-btc", 0.5))   # 6 wins of $0.50
    for _ in range(4):
        ex._trades.append(_exit_trade("breakeven-btc", 0.0))   # 4 breakevens, no losses
    wr, payoff, n, *_ = ex._live_edge_stats("breakeven-btc")
    assert n == 10
    assert wr == pytest.approx(0.6)
    assert payoff == pytest.approx(10.0)  # capped sentinel, NOT the $0.50 avg_win
    assert half_kelly_fraction(wr, payoff) > 0  # a genuine edge, not 0/negative


def test_fresh_strategy_gets_observation_leverage_and_tiny_size():
    from src.services.confidence_ladder import ladder_leverage
    ex = PaperTradingExecutor(initial_balance=200.0)
    wr, payoff, n, wr_lo, wr_hi = ex._live_edge_stats("fresh-btc")
    assert (wr, payoff, n) == (0.0, 0.0, 0)
    assert (wr_lo, wr_hi) == (0.0, 0.0)
    assert ladder_leverage("BTC", n, wr, payoff) == 1


@pytest.mark.asyncio
async def test_ladder_lifts_leverage_with_winning_history():
    """After >=30 winning BTC trades, effective leverage must exceed 1."""
    from src.strategies.base_strategy import Signal, SignalType, StrategyConfig, StrategyTier
    from src.strategies.registry import create_strategy

    ex = PaperTradingExecutor(initial_balance=50000.0)
    ex._mid_prices = {"BTC": 50000.0}
    ex._last_price_fetch = 9999999999.0

    # Seed 30 winning exit trades for the strategy so it reaches FULL tier
    for _ in range(30):
        ex._trades.append(_exit_trade("ladder-test-btc", 50.0))  # payoff=inf but wr=1.0

    config = StrategyConfig(
        name="ladder-test-btc",
        symbol="BTC",
        tier=StrategyTier.A,
        leverage=5,
        size_usd=1000.0,
    )
    strategy = create_strategy("rsi", config)
    signal = Signal(signal_type=SignalType.LONG, symbol="BTC", size_usd=1000.0, reason="ladder-test")
    result = await ex.execute_signal(signal, strategy)

    assert result.success is True, f"Entry failed: {result.error}"
    pos = ex._positions["ladder-test-btc:BTC"]
    # With 30 wins, win_rate=1.0, payoff high → ladder should lift leverage above 1
    assert pos.leverage > 1, f"Expected leverage > 1 after 30 wins, got {pos.leverage}"


@pytest.mark.asyncio
async def test_ramp_tier_caps_leverage_at_half_asset_cap():
    """In the RAMP tier (10-29 trades), leverage is lifted above 1 but capped
    at half the asset cap, regardless of edge strength."""
    from src.services.confidence_ladder import HL_MAX_LEVERAGE
    from src.strategies.base_strategy import Signal, SignalType, StrategyConfig, StrategyTier
    from src.strategies.registry import create_strategy

    ex = PaperTradingExecutor(initial_balance=50000.0)
    ex._mid_prices = {"BTC": 50000.0}
    ex._last_price_fetch = 9999999999.0

    # Seed 15 winning exit trades → RAMP tier (>=10, <30)
    for _ in range(15):
        ex._trades.append(_exit_trade("ramp-test-btc", 50.0))

    config = StrategyConfig(
        name="ramp-test-btc",
        symbol="BTC",
        tier=StrategyTier.A,
        leverage=5,
        size_usd=1000.0,
    )
    strategy = create_strategy("rsi", config)
    signal = Signal(signal_type=SignalType.LONG, symbol="BTC", size_usd=1000.0, reason="ramp-test")
    result = await ex.execute_signal(signal, strategy)

    assert result.success is True, f"Entry failed: {result.error}"
    pos = ex._positions["ramp-test-btc:BTC"]
    assert 1 < pos.leverage <= HL_MAX_LEVERAGE["BTC"] // 2, (
        f"Ramp tier should lift leverage above 1 but cap at half the asset cap "
        f"({HL_MAX_LEVERAGE['BTC'] // 2}), got {pos.leverage}"
    )


@pytest.mark.asyncio
async def test_market_neutral_signal_skips_observation_floor():
    """A market-neutral sleeve (e.g. xsec_carry) must trade at its configured
    leg size, NOT the 10% directional observation floor. A fresh directional
    strategy with identical config sizes at 10%; the market-neutral one at 100%."""
    from src.strategies.base_strategy import Signal, SignalType, StrategyConfig, StrategyTier
    from src.strategies.registry import create_strategy

    def _fresh_exec():
        ex = PaperTradingExecutor(initial_balance=50000.0)
        ex._mid_prices = {"BTC": 50000.0}
        ex._last_price_fetch = 9999999999.0
        return ex

    cfg = StrategyConfig(name="mn-test", symbol="BTC", tier=StrategyTier.E, size_usd=1000.0)
    strat = create_strategy("rsi", cfg)

    # Directional fresh: observation floor => ~10% of configured size
    ex_dir = _fresh_exec()
    r_dir = await ex_dir.execute_signal(
        Signal(signal_type=SignalType.LONG, symbol="BTC", size_usd=1000.0, reason="dir"), strat
    )
    assert r_dir.success, r_dir.error
    dir_size = ex_dir._positions["mn-test:BTC"].size_usd

    # Market-neutral fresh: full configured size, 1x
    ex_mn = _fresh_exec()
    r_mn = await ex_mn.execute_signal(
        Signal(signal_type=SignalType.LONG, symbol="BTC", size_usd=1000.0,
               reason="mn", metadata={"market_neutral": True}), strat
    )
    assert r_mn.success, r_mn.error
    mn_pos = ex_mn._positions["mn-test:BTC"]

    assert mn_pos.size_usd == pytest.approx(1000.0, rel=0.01), (
        f"market-neutral leg should trade full size, got {mn_pos.size_usd}"
    )
    assert mn_pos.leverage == 1
    assert mn_pos.size_usd > dir_size * 5, (
        f"market-neutral ({mn_pos.size_usd}) must be >>  directional floor ({dir_size})"
    )
