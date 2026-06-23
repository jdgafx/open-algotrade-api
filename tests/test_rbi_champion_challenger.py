"""F1 — champion-challenger promotion on the incumbent's RECENT live track record.

The RBI optimizer promotes the best backtest candidate. Before F1 it did so
unconditionally, ignoring how the incumbent is actually doing with live capital.
F1 adds a live-performance gate around the (already-passed) backtest gate, keyed
on a RECENT window of realized trades (NOT lifetime cumulative PnL, which lags a
stale historical buffer and would freeze a decayed winner):

  - incumbent is a RECENT LIVE WINNER (>= MIN recent trades, positive recent PnL)
        -> RETAIN it; do not churn working params on backtest hope.
  - incumbent is a RECENT LIVE LOSER  (>= MIN recent trades, negative recent PnL)
        -> promote the challenger (try to fix it).
  - thin recent data (< MIN recent trades, e.g. fresh strategy or post-redeploy
    reset) -> defer to the backtest gate (unchanged behavior). SAFE degradation:
    never freezes on missing data.

These tests drive the public RBIPipeline.run_cycle interface with a fake
strategy + fake optimizer, asserting the promotion DECISION, not internals.
"""

import pytest
from unittest.mock import AsyncMock

from src.engine.rbi_pipeline import RBIPipeline, MIN_RECENT_TRADES_FOR_CHAMPION
from src.engine.optimizer import OptimizationEngine, OptimizationResult

_ENOUGH = MIN_RECENT_TRADES_FOR_CHAMPION + 2


def _strategy(active_positions=0, recent_pnl=0.0, recent_trades=0,
              total_pnl=0.0, total_trades=0, winning_trades=0):
    return {
        "id": 8, "strategy_type": "mean_reversion", "symbol": "BTC",
        "timeframe": "1h", "active_positions": active_positions,
        "params": {"z": 1.5},
        "recent_pnl": recent_pnl, "recent_trades": recent_trades,
        "total_pnl": total_pnl, "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": max(0, total_trades - winning_trades),
    }


def _cand(trades=30, pf=1.6, wr=45.0, dd=8.0, sharpe=1.3, dsr=0.99, cpcv=0.8):
    """A candidate that clears the backtest promotion gate."""
    return OptimizationResult(
        params={"z": 2.0}, in_sample_sharpe=1.2, out_sample_sharpe=sharpe,
        out_sample_profit_factor=pf, out_sample_win_rate=wr,
        out_sample_total_trades=trades, out_sample_max_drawdown=dd,
        composite_score=0.7, passed_walkforward=True,
        out_sample_dsr=dsr, cpcv_frac_positive=cpcv,
    )


def _pipe(strategy):
    opt = OptimizationEngine()
    opt.optimize = AsyncMock(return_value=[_cand()])  # always a gate-passing challenger
    return RBIPipeline(
        get_strategy_fn=AsyncMock(return_value=strategy),
        patch_strategy_fn=AsyncMock(), optimizer=opt,
    )


@pytest.mark.asyncio
async def test_recent_live_winner_is_retained_not_churned():
    strat = _strategy(recent_pnl=60.0, recent_trades=_ENOUGH)
    pipe = _pipe(strat)
    event = await pipe.run_cycle("mean_reversion", 8, "BTC")
    assert event.promoted is False
    assert event.reason == "retained_live_winner"
    pipe._patch_strategy_fn.assert_not_called()


@pytest.mark.asyncio
async def test_recent_live_loser_is_replaced_by_challenger():
    strat = _strategy(recent_pnl=-40.0, recent_trades=_ENOUGH)
    pipe = _pipe(strat)
    event = await pipe.run_cycle("mean_reversion", 8, "BTC")
    assert event.promoted is True
    assert event.reason == "promoted_over_live_loser"
    pipe._patch_strategy_fn.assert_awaited_once()


@pytest.mark.asyncio
async def test_decayed_winner_lifetime_positive_but_recent_negative_is_replaced():
    # The blocker-#2 case: huge stale lifetime profit, but RECENTLY bleeding.
    # Must be REPLACED (decided on recent window), never frozen on lifetime PnL.
    strat = _strategy(recent_pnl=-30.0, recent_trades=_ENOUGH,
                      total_pnl=500.0, total_trades=200, winning_trades=140)
    pipe = _pipe(strat)
    event = await pipe.run_cycle("mean_reversion", 8, "BTC")
    assert event.promoted is True
    assert event.reason == "promoted_over_live_loser"
    pipe._patch_strategy_fn.assert_awaited_once()


@pytest.mark.asyncio
async def test_thin_recent_data_defers_to_backtest_gate():
    # below MIN recent trades -> no trustworthy recent signal -> defer (promote).
    strat = _strategy(recent_pnl=15.0, recent_trades=MIN_RECENT_TRADES_FOR_CHAMPION - 1,
                      total_pnl=300.0, total_trades=80, winning_trades=55)
    pipe = _pipe(strat)
    event = await pipe.run_cycle("mean_reversion", 8, "BTC")
    assert event.promoted is True
    assert event.reason == "promotion_gate_passed"
    pipe._patch_strategy_fn.assert_awaited_once()


@pytest.mark.asyncio
async def test_post_reset_zero_recent_trades_defers_safely():
    # Railway redeploy wiped the trade window -> recent_trades 0 -> defer, do not freeze.
    strat = _strategy(recent_pnl=0.0, recent_trades=0, total_pnl=0.0, total_trades=0)
    pipe = _pipe(strat)
    event = await pipe.run_cycle("mean_reversion", 8, "BTC")
    assert event.promoted is True
    assert event.reason == "promotion_gate_passed"


@pytest.mark.asyncio
async def test_breakeven_recent_defers_to_backtest_gate():
    strat = _strategy(recent_pnl=0.0, recent_trades=_ENOUGH)
    pipe = _pipe(strat)
    event = await pipe.run_cycle("mean_reversion", 8, "BTC")
    assert event.promoted is True
    assert event.reason == "promotion_gate_passed"


@pytest.mark.asyncio
async def test_before_metrics_populated_with_recent_edge():
    strat = _strategy(recent_pnl=-40.0, recent_trades=_ENOUGH, total_pnl=10.0, total_trades=50)
    pipe = _pipe(strat)
    event = await pipe.run_cycle("mean_reversion", 8, "BTC")
    assert event.before_metrics.get("recent_pnl") == -40.0
    assert event.before_metrics.get("recent_trades") == _ENOUGH


@pytest.mark.asyncio
async def test_retained_winner_records_declined_challenger():
    strat = _strategy(recent_pnl=60.0, recent_trades=_ENOUGH)
    pipe = _pipe(strat)
    event = await pipe.run_cycle("mean_reversion", 8, "BTC")
    assert event.after_metrics.get("oos_sharpe") == 1.3
    assert event.after_params == {}
