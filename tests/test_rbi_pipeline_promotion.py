import pytest
from unittest.mock import AsyncMock
from src.engine.rbi_pipeline import RBIPipeline
from src.engine.optimizer import OptimizationEngine, OptimizationResult


def _strategy(active_positions=0):
    return {"id": 8, "strategy_type": "mean_reversion", "symbol": "BTC",
            "timeframe": "1h", "active_positions": active_positions, "params": {"z": 1.5}}


def _cand(trades, pf=1.6, wr=45.0, dd=8.0, sharpe=1.3, passed_wf=True):
    return OptimizationResult(
        params={"z": 2.0}, in_sample_sharpe=1.2, out_sample_sharpe=sharpe,
        out_sample_profit_factor=pf, out_sample_win_rate=wr,
        out_sample_total_trades=trades, out_sample_max_drawdown=dd,
        composite_score=0.7, passed_walkforward=passed_wf,
    )


@pytest.mark.asyncio
async def test_walkforward_pass_but_under_30_trades_is_not_promoted():
    opt = OptimizationEngine()
    opt.optimize = AsyncMock(return_value=[_cand(trades=22)])  # passes loose screen, fails promotion
    pipe = RBIPipeline(
        get_strategy_fn=AsyncMock(return_value=_strategy()),
        patch_strategy_fn=AsyncMock(), optimizer=opt,
    )
    event = await pipe.run_cycle("mean_reversion", 8, "BTC")
    assert event.promoted is False
    assert event.reason == "no_passing_candidates"
    pipe._patch_strategy_fn.assert_not_called()


@pytest.mark.asyncio
async def test_candidate_clearing_promotion_gate_is_promoted():
    opt = OptimizationEngine()
    opt.optimize = AsyncMock(return_value=[_cand(trades=30)])
    pipe = RBIPipeline(
        get_strategy_fn=AsyncMock(return_value=_strategy()),
        patch_strategy_fn=AsyncMock(), optimizer=opt,
    )
    event = await pipe.run_cycle("mean_reversion", 8, "BTC")
    assert event.promoted is True
    assert event.reason == "promotion_gate_passed"
    pipe._patch_strategy_fn.assert_awaited_once()
