from types import SimpleNamespace
import pytest
from src.engine.optimizer import OptimizationEngine


def _result(trades, pf, wr, dd, sharpe=1.5):
    # Mirrors OptimizationResult (the object the promotion pipeline filters on).
    return SimpleNamespace(
        out_sample_total_trades=trades,
        out_sample_profit_factor=pf,
        out_sample_win_rate=wr,
        out_sample_max_drawdown=dd,
        out_sample_sharpe=sharpe,
    )


def test_promotion_gate_requires_30_trades():
    eng = OptimizationEngine()
    assert eng._passes_promotion_gate(_result(trades=29, pf=1.6, wr=45.0, dd=8.0)) is False
    assert eng._passes_promotion_gate(_result(trades=30, pf=1.6, wr=45.0, dd=8.0)) is True


def test_promotion_gate_rejects_thin_edge_even_with_many_trades():
    eng = OptimizationEngine()
    assert eng._passes_promotion_gate(_result(trades=40, pf=1.1, wr=33.0, dd=14.0)) is False


def test_promotion_gate_rejects_low_sharpe():
    eng = OptimizationEngine()
    assert eng._passes_promotion_gate(_result(trades=40, pf=1.6, wr=45.0, dd=8.0, sharpe=0.4)) is False


def test_promotion_gate_passes_at_all_thresholds():
    eng = OptimizationEngine()
    assert eng._passes_promotion_gate(
        _result(trades=30, pf=1.5, wr=40.0, dd=10.0, sharpe=1.0)
    ) is True
