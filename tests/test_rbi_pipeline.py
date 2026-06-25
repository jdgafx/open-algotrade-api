import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from src.engine.rbi_pipeline import RBIPipeline, PromotionEvent
from src.engine.optimizer import OptimizationEngine


def _mock_strategy(active_positions=0, params=None):
    s = {"id": 8, "strategy_type": "mean_reversion", "symbol": "BTC",
         "timeframe": "1h", "active_positions": active_positions,
         "params": params or {"zscore_entry": 1.5}}
    return s


@pytest.fixture
def pipeline():
    get_fn = AsyncMock(return_value=_mock_strategy())
    patch_fn = AsyncMock(return_value={})
    optimizer = OptimizationEngine()  # real engine so _passes_promotion_gate runs
    return RBIPipeline(get_strategy_fn=get_fn, patch_strategy_fn=patch_fn, optimizer=optimizer)


@pytest.mark.asyncio
async def test_stages_when_active_position(pipeline):
    """With active position, optimization runs but param application is staged."""
    from src.engine.optimizer import OptimizationResult
    passing = OptimizationResult(
        params={"zscore_entry": 2.0}, in_sample_sharpe=1.2, out_sample_sharpe=1.1,
        out_sample_profit_factor=1.5, out_sample_win_rate=48.0,
        out_sample_total_trades=30, out_sample_max_drawdown=8.0,
        composite_score=0.72, passed_walkforward=True,
        out_sample_dsr=0.99, cpcv_frac_positive=0.8,
    )
    pipeline._optimizer.optimize = AsyncMock(return_value=[passing])
    pipeline._get_strategy_fn = AsyncMock(return_value=_mock_strategy(active_positions=1))
    result = await pipeline.run_cycle("mean_reversion", 8, "BTC", "1h")
    assert result.promoted is True
    assert result.reason == "staged_active_position"
    assert result.after_params == {"zscore_entry": 2.0}
    pipeline._patch_strategy_fn.assert_not_called()


@pytest.mark.asyncio
async def test_skips_when_no_passing_candidates(pipeline):
    from src.engine.optimizer import OptimizationResult
    failing = OptimizationResult(
        params={}, in_sample_sharpe=0.5, out_sample_sharpe=0.3,
        out_sample_profit_factor=0.9, out_sample_win_rate=30.0,
        out_sample_total_trades=8, out_sample_max_drawdown=20.0,
        composite_score=0.2, passed_walkforward=False,
    )
    pipeline._optimizer.optimize = AsyncMock(return_value=[failing])
    result = await pipeline.run_cycle("mean_reversion", 8, "BTC", "1h")
    assert result.promoted is False
    assert result.reason == "no_passing_candidates"


@pytest.mark.asyncio
async def test_promotes_when_candidate_passes(pipeline):
    from src.engine.optimizer import OptimizationResult
    # meets promotion gate: 30 trades, sharpe>=1.0, pf>=1.5, wr>=40, dd<=10
    passing = OptimizationResult(
        params={"zscore_entry": 2.0}, in_sample_sharpe=1.2, out_sample_sharpe=1.1,
        out_sample_profit_factor=1.5, out_sample_win_rate=48.0,
        out_sample_total_trades=30, out_sample_max_drawdown=8.0,
        composite_score=0.72, passed_walkforward=True,
        out_sample_dsr=0.99, cpcv_frac_positive=0.8,  # clear the overfitting guards too
    )
    pipeline._optimizer.optimize = AsyncMock(return_value=[passing])
    result = await pipeline.run_cycle("mean_reversion", 8, "BTC", "1h")
    assert result.promoted is True
    assert result.after_params == {"zscore_entry": 2.0}
    pipeline._patch_strategy_fn.assert_called_once()


@pytest.mark.asyncio
async def test_rollback_returns_previous_params(pipeline):
    from src.engine.optimizer import OptimizationResult
    # meets promotion gate: 30 trades, sharpe>=1.0, pf>=1.5, wr>=40, dd<=10
    passing = OptimizationResult(
        params={"zscore_entry": 2.0}, in_sample_sharpe=1.2, out_sample_sharpe=1.1,
        out_sample_profit_factor=1.5, out_sample_win_rate=48.0,
        out_sample_total_trades=30, out_sample_max_drawdown=8.0,
        composite_score=0.72, passed_walkforward=True,
        out_sample_dsr=0.99, cpcv_frac_positive=0.8,  # clear the overfitting guards too
    )
    pipeline._optimizer.optimize = AsyncMock(return_value=[passing])
    await pipeline.run_cycle("mean_reversion", 8, "BTC", "1h")
    prev = pipeline.get_rollback_params("mean_reversion")
    assert prev == {"zscore_entry": 1.5}


def test_history_is_empty_initially(pipeline):
    assert pipeline.get_history() == []
