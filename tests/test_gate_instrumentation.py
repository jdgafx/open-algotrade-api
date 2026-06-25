"""U5 / T032 — explainable promotion rejects + gate-passability probe.

Two concerns:
  1. Instrumentation: a reject names WHICH criterion the best candidate failed,
     so a correct "no real edge" reject is distinguishable from a structurally
     unpassable gate (gate_failure_reason + the no_passing_candidates event's
     after_metrics).
  2. Passability probe: the gate, run through the REAL overfitting math (not
     hand-set DSR/CPCV fields), CAN promote a genuine edge — and correctly
     deflates that same edge when it is the luckiest of many wide-ranging
     trials. This proves the loop can still learn; the strictness has teeth,
     not a blanket block.
"""

import numpy as np
import pytest
from unittest.mock import AsyncMock

from src.engine.optimizer import OptimizationEngine, OptimizationResult
from src.engine.rbi_pipeline import RBIPipeline
from src.engine.rigor import DSR_MIN, deflated_sharpe_ratio


def _result(trades=30, pf=1.6, wr=45.0, dd=8.0, sharpe=1.3, dsr=0.99, cpcv=0.8,
            composite=0.7, params=None):
    return OptimizationResult(
        params=params or {"z": 2.0}, in_sample_sharpe=1.2, out_sample_sharpe=sharpe,
        out_sample_profit_factor=pf, out_sample_win_rate=wr,
        out_sample_total_trades=trades, out_sample_max_drawdown=dd,
        composite_score=composite, passed_walkforward=True,
        out_sample_dsr=dsr, cpcv_frac_positive=cpcv,
    )


# --- Instrumentation: gate_failure_reason names the first failing criterion ---

def test_passing_candidate_has_no_failure_reason():
    eng = OptimizationEngine()
    assert eng.gate_failure_reason(_result()) is None


def test_failure_reason_reports_trade_floor_first():
    eng = OptimizationEngine()
    reason = eng.gate_failure_reason(_result(trades=29))
    assert reason is not None and reason.startswith("oos_trades=29")
    assert ">=30" in reason


def test_failure_reason_reports_dsr_when_edge_bars_pass():
    eng = OptimizationEngine()
    # clears trades/PF/WR/DD/Sharpe but the overfitting guard catches it
    reason = eng.gate_failure_reason(_result(dsr=0.50))
    assert reason is not None and reason.startswith("dsr=0.500")
    assert ">=0.85" in reason


def test_failure_reason_reports_cpcv_last():
    eng = OptimizationEngine()
    # everything cheap passes incl. DSR; only CPCV stability fails (== 0.5, not > 0.5)
    reason = eng.gate_failure_reason(_result(cpcv=0.5))
    assert reason is not None and reason.startswith("cpcv_frac_positive=0.500")


def test_failure_reason_order_matches_gate():
    """A candidate failing multiple criteria reports the FIRST in gate order."""
    eng = OptimizationEngine()
    reason = eng.gate_failure_reason(_result(trades=10, pf=0.5, dsr=0.1, cpcv=0.0))
    assert reason.startswith("oos_trades=")  # trades checked before PF/DSR/CPCV


# --- Instrumentation: the run_cycle reject event carries the explanation ---

@pytest.mark.asyncio
async def test_reject_event_records_best_candidate_metrics_and_criterion():
    opt = OptimizationEngine()
    # two near-miss candidates; the higher-composite one is the one explained
    weak = _result(trades=8, dsr=0.2, cpcv=0.0, composite=0.2)
    best_miss = _result(trades=40, pf=1.6, wr=45.0, dd=8.0, sharpe=1.0,
                        dsr=0.40, cpcv=0.0, composite=0.65)
    opt.optimize = AsyncMock(return_value=[weak, best_miss])
    pipe = RBIPipeline(
        get_strategy_fn=AsyncMock(return_value={
            "id": 8, "strategy_type": "mean_reversion", "symbol": "BTC",
            "timeframe": "1h", "active_positions": 0, "params": {"z": 1.5}}),
        patch_strategy_fn=AsyncMock(), optimizer=opt,
    )
    event = await pipe.run_cycle("mean_reversion", 8, "BTC")

    assert event.promoted is False
    assert event.reason == "no_passing_candidates"          # back-compat preserved
    am = event.after_metrics
    assert am["failing_criterion"].startswith("dsr=0.400")  # best candidate's first failure
    assert am["oos_trades"] == 40                            # it IS the higher-composite miss
    assert am["composite_score"] == 0.65
    assert "dsr" in am and "cpcv_frac_positive" in am
    pipe._patch_strategy_fn.assert_not_called()


@pytest.mark.asyncio
async def test_reject_event_handles_empty_candidate_list():
    opt = OptimizationEngine()
    opt.optimize = AsyncMock(return_value=[])
    pipe = RBIPipeline(
        get_strategy_fn=AsyncMock(return_value={
            "id": 8, "strategy_type": "mean_reversion", "symbol": "BTC",
            "timeframe": "1h", "active_positions": 0, "params": {"z": 1.5}}),
        patch_strategy_fn=AsyncMock(), optimizer=opt,
    )
    event = await pipe.run_cycle("mean_reversion", 8, "BTC")
    assert event.reason == "no_passing_candidates"
    assert event.after_metrics["failing_criterion"] == "no_candidates"


# --- Passability probe: the REAL math can promote genuine edge, and deflates overfit ---

def _genuine_edge_returns():
    """A deterministic per-trade return series with a real positive edge:
    35 wins of +2%, 15 losses of -1% over 50 trades."""
    return np.concatenate([np.full(35, 0.02), np.full(15, -0.01)])


def _per_trade_sharpe(returns):
    return float(np.mean(returns) / np.std(returns, ddof=1))


def test_real_dsr_passes_for_genuine_edge_few_trials():
    """A real edge selected from a TIGHT trial distribution clears DSR_MIN —
    the gate is passable, not a blanket block (the core T032 worry)."""
    returns = _genuine_edge_returns()
    sr_selected = _per_trade_sharpe(returns)
    sr_trials = [0.10, 0.15, 0.12, 0.18, 0.14]          # few, low-variance tries
    dsr = deflated_sharpe_ratio(sr_selected, sr_trials, returns)["DSR"]
    assert dsr >= DSR_MIN


def test_real_dsr_deflates_same_edge_when_luckiest_of_many():
    """The SAME selected edge, if it were the best of 200 wide-ranging trials,
    is correctly deflated below DSR_MIN — the guard has teeth."""
    returns = _genuine_edge_returns()
    sr_selected = _per_trade_sharpe(returns)
    sr_trials = np.linspace(-0.5, 1.0, 200)             # many, high-variance tries
    dsr = deflated_sharpe_ratio(sr_selected, sr_trials, returns)["DSR"]
    assert dsr < DSR_MIN


def test_full_gate_promotes_a_genuine_edge_candidate():
    """End-to-end on the gate itself: a candidate carrying the REAL DSR computed
    above plus realistic edge metrics is promoted — proving the assembled gate
    (not just hand-set fields) admits genuine edge."""
    eng = OptimizationEngine()
    returns = _genuine_edge_returns()
    real_dsr = deflated_sharpe_ratio(
        _per_trade_sharpe(returns), [0.10, 0.15, 0.12, 0.18, 0.14], returns
    )["DSR"]
    candidate = _result(trades=35, pf=1.8, wr=46.0, dd=7.0, sharpe=0.9,
                        dsr=real_dsr, cpcv=0.83)
    assert eng._passes_promotion_gate(candidate) is True
    assert eng.gate_failure_reason(candidate) is None
