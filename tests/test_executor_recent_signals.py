"""Tests for the F1 live-loop signals on the paper executor:
recent_realized_pnl (recent-window realized PnL) and open_position_count."""

from datetime import datetime, timezone

from src.execution.paper_executor import (
    PaperTradingExecutor, PaperTrade, PaperPosition, RECENT_PNL_WINDOW,
)


def _exit(strategy, pnl):
    return PaperTrade(id=0, symbol="BTC", side="long", action="exit", price=100.0,
                      size=1.0, size_usd=50.0, pnl=pnl, pnl_pct=pnl / 50.0 * 100,
                      reason="t", strategy_name=strategy)


def _pos(strategy, symbol="BTC"):
    return PaperPosition(symbol=symbol, side="long", size=1.0, entry_price=100.0,
                         entry_time=datetime.now(timezone.utc), strategy_name=strategy)


def test_recent_realized_pnl_uses_only_the_last_window():
    ex = PaperTradingExecutor(initial_balance=1000.0)
    # 5 old big winners, then WINDOW recent small losers
    for _ in range(5):
        ex._trades.append(_exit("x-btc", 100.0))     # stale +500 buffer
    for _ in range(RECENT_PNL_WINDOW):
        ex._trades.append(_exit("x-btc", -2.0))       # recent: -2 * WINDOW
    recent_pnl, n = ex.recent_realized_pnl("x-btc")
    assert n == RECENT_PNL_WINDOW
    # the stale +500 must NOT leak into the recent window
    assert recent_pnl == -2.0 * RECENT_PNL_WINDOW


def test_recent_realized_pnl_counts_all_when_fewer_than_window():
    ex = PaperTradingExecutor(initial_balance=1000.0)
    for pnl in (10.0, -3.0, 5.0):
        ex._trades.append(_exit("y-btc", pnl))
    recent_pnl, n = ex.recent_realized_pnl("y-btc")
    assert n == 3
    assert recent_pnl == 12.0


def test_recent_realized_pnl_ignores_other_strategies_and_entries():
    ex = PaperTradingExecutor(initial_balance=1000.0)
    ex._trades.append(_exit("other", 99.0))
    entry = _exit("z-btc", 7.0)
    entry.action = "entry"
    ex._trades.append(entry)
    recent_pnl, n = ex.recent_realized_pnl("z-btc")
    assert (recent_pnl, n) == (0.0, 0)


def test_open_position_count_per_strategy():
    ex = PaperTradingExecutor(initial_balance=1000.0)
    ex._positions["a-btc:BTC"] = _pos("a-btc")
    ex._positions["a-btc:ETH"] = _pos("a-btc", symbol="ETH")
    ex._positions["b-btc:BTC"] = _pos("b-btc")
    assert ex.open_position_count("a-btc") == 2
    assert ex.open_position_count("b-btc") == 1
    assert ex.open_position_count("none") == 0


def _strategy_attrs(**overrides):
    """A namespace carrying every StrategyInstanceOut-required attribute, so we
    can validate the from_attributes serialization seam the endpoint relies on."""
    from types import SimpleNamespace
    base = dict(
        id=1, name="x-btc", strategy_type="mean_reversion", tier="A", status="running",
        symbol="BTC", timeframe="1h", leverage=3, size_usd=50.0, target_pct=9.0,
        max_loss_pct=-8.0, lookback_days=14, interval_seconds=30, enabled=True,
        params={"z": 1.5}, total_trades=0, winning_trades=0, losing_trades=0,
        total_pnl=0.0, max_drawdown=0.0, iterations=0, errors=0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_strategy_out_serializes_live_loop_signals_from_attributes():
    # Locks the regression class that made blocker #1: the schema MUST declare
    # and round-trip the enriched live-loop fields, else FastAPI strips them.
    from src.api import schemas
    obj = _strategy_attrs(active_positions=2, recent_pnl=-12.5, recent_trades=14)
    out = schemas.StrategyInstanceOut.model_validate(obj)
    assert out.active_positions == 2
    assert out.recent_pnl == -12.5
    assert out.recent_trades == 14


def test_strategy_out_live_loop_signals_default_to_zero_when_absent():
    # The create / non-paper path never sets these -> must default, not error.
    from src.api import schemas
    out = schemas.StrategyInstanceOut.model_validate(_strategy_attrs())
    assert (out.active_positions, out.recent_pnl, out.recent_trades) == (0, 0.0, 0)
