"""F2 (R9) — entry sizing gates on the Wilson LOWER-BOUND win-rate.

A lucky streak (high point win-rate but a wide CI on thin data) must not ramp
Kelly/leverage before the edge's confidence interval clears breakeven. A
genuinely proven edge (lower bound above breakeven) still ramps fully.
"""

import pytest

from src.execution.paper_executor import PaperTradingExecutor, PaperTrade
from src.services.kelly import half_kelly_fraction


def _exit(strategy, pnl):
    return PaperTrade(id=0, symbol="BTC", side="long", action="exit", price=100.0,
                      size=1.0, size_usd=50.0, pnl=pnl, pnl_pct=pnl / 50.0 * 100,
                      reason="t", strategy_name=strategy)


def test_lower_bound_floors_a_lucky_streak():
    """6 wins / 2 losses at 1:1 payoff: point WR 0.75 is 'profitable' (breakeven
    0.5), but the Wilson lower bound on only 8 trades sits below breakeven, so
    the conservative Kelly is zero — luck is held back."""
    ex = PaperTradingExecutor(initial_balance=1000.0)
    for _ in range(6):
        ex._trades.append(_exit("lucky-btc", 50.0))
    for _ in range(2):
        ex._trades.append(_exit("lucky-btc", -50.0))
    win_rate, payoff, n, wr_lo, _ = ex._live_edge_stats("lucky-btc")

    assert n == 8 and win_rate == pytest.approx(0.75) and payoff == pytest.approx(1.0)
    assert wr_lo < 0.5                                   # CI not yet clear of breakeven
    assert half_kelly_fraction(win_rate, payoff) > 0     # the POINT estimate would bet
    assert half_kelly_fraction(wr_lo, payoff) == 0.0     # the lower bound does NOT


def test_lower_bound_allows_a_proven_edge():
    """30 wins / 5 losses at 2:1 payoff: even the lower bound clears breakeven
    (0.333), so the conservative Kelly still sizes — genuine edge is not
    over-suppressed."""
    ex = PaperTradingExecutor(initial_balance=1000.0)
    for _ in range(30):
        ex._trades.append(_exit("proven-btc", 50.0))
    for _ in range(5):
        ex._trades.append(_exit("proven-btc", -25.0))
    win_rate, payoff, n, wr_lo, _ = ex._live_edge_stats("proven-btc")

    assert n == 35 and payoff == pytest.approx(2.0)
    assert wr_lo > 0.333                                 # CI clear of breakeven
    assert half_kelly_fraction(wr_lo, payoff) > 0        # still bets the proven edge


@pytest.mark.asyncio
async def test_entry_size_is_floored_for_lucky_streak():
    """Integration: a lucky-streak strategy enters at the observation floor, not
    a ramped size — the lower-bound gate reaches the live sizing path."""
    from src.strategies.base_strategy import Signal, SignalType, StrategyConfig, StrategyTier
    from src.strategies.registry import create_strategy

    ex = PaperTradingExecutor(initial_balance=10000.0)
    ex._mid_prices = {"BTC": 50000.0}
    ex._last_price_fetch = 9999999999.0
    for _ in range(6):
        ex._trades.append(_exit("lucky-entry-btc", 50.0))
    for _ in range(2):
        ex._trades.append(_exit("lucky-entry-btc", -50.0))

    config = StrategyConfig(name="lucky-entry-btc", symbol="BTC",
                            tier=StrategyTier.A, leverage=5, size_usd=1000.0)
    strategy = create_strategy("rsi", config)
    signal = Signal(signal_type=SignalType.LONG, symbol="BTC", size_usd=1000.0, reason="t")
    result = await ex.execute_signal(signal, strategy)

    assert result.success is True, f"Entry failed: {result.error}"
    pos = ex._positions["lucky-entry-btc:BTC"]
    # base 1000 * compound 1.0 * kelly floor 0.10 * leverage 1x (n<10) == ~100.
    # A point-WR ramp would have produced ~1000 (kelly_mult 1.0).
    assert pos.size_usd == pytest.approx(100.0, rel=0.02)
    assert pos.leverage == 1


@pytest.mark.asyncio
async def test_entry_ramps_leverage_for_proven_edge():
    """Integration: a strategy whose lower bound clears breakeven ramps leverage
    above 1 — F2 does not strand genuine edge at the floor."""
    from src.strategies.base_strategy import Signal, SignalType, StrategyConfig, StrategyTier
    from src.strategies.registry import create_strategy

    ex = PaperTradingExecutor(initial_balance=50000.0)
    ex._mid_prices = {"BTC": 50000.0}
    ex._last_price_fetch = 9999999999.0
    for _ in range(30):
        ex._trades.append(_exit("proven-entry-btc", 50.0))
    for _ in range(5):
        ex._trades.append(_exit("proven-entry-btc", -25.0))

    config = StrategyConfig(name="proven-entry-btc", symbol="BTC",
                            tier=StrategyTier.A, leverage=5, size_usd=1000.0)
    strategy = create_strategy("rsi", config)
    signal = Signal(signal_type=SignalType.LONG, symbol="BTC", size_usd=1000.0, reason="t")
    result = await ex.execute_signal(signal, strategy)

    assert result.success is True, f"Entry failed: {result.error}"
    assert ex._positions["proven-entry-btc:BTC"].leverage > 1
