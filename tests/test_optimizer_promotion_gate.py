from types import SimpleNamespace
import pytest
from src.engine.optimizer import OptimizationEngine
from src.engine.rigor import DSR_MIN


def _result(trades, pf, wr, dd, sharpe=1.5, dsr=0.99, cpcv=0.8):
    # Mirrors OptimizationResult (the object the promotion pipeline filters on).
    # dsr/cpcv default to passing so the original threshold tests still isolate
    # the criterion under test; the DSR/CPCV tests below drive them explicitly.
    return SimpleNamespace(
        out_sample_total_trades=trades,
        out_sample_profit_factor=pf,
        out_sample_win_rate=wr,
        out_sample_max_drawdown=dd,
        out_sample_sharpe=sharpe,
        out_sample_dsr=dsr,
        cpcv_frac_positive=cpcv,
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


# --- Overfitting guards (plan U2): DSR + CPCV are ADDITIVE to the screen above ---

def test_promotion_gate_rejects_low_dsr():
    """A candidate clearing every classic threshold but with a Deflated Sharpe
    below DSR_MIN (edge indistinguishable from the luckiest of N tries) is rejected."""
    eng = OptimizationEngine()
    strong = dict(trades=40, pf=1.6, wr=45.0, dd=8.0, sharpe=1.5)
    assert eng._passes_promotion_gate(_result(**strong, dsr=DSR_MIN - 0.01)) is False
    assert eng._passes_promotion_gate(_result(**strong, dsr=DSR_MIN)) is True


def test_promotion_gate_rejects_unstable_cpcv():
    """A candidate clearing every classic threshold + DSR but positive on only
    half-or-fewer combinatorial-purged-CV paths (no cross-subset stability) is rejected."""
    eng = OptimizationEngine()
    strong = dict(trades=40, pf=1.6, wr=45.0, dd=8.0, sharpe=1.5, dsr=0.99)
    assert eng._passes_promotion_gate(_result(**strong, cpcv=0.5)) is False   # not > 0.5
    assert eng._passes_promotion_gate(_result(**strong, cpcv=0.51)) is True


def test_default_result_without_guard_evidence_cannot_promote():
    """The OptimizationResult dataclass defaults (dsr=0.0, cpcv=0.0) must fail the
    gate — a candidate with no guard evidence is never promotable."""
    from src.engine.optimizer import OptimizationResult
    r = OptimizationResult(
        params={}, in_sample_sharpe=1.2, out_sample_sharpe=1.5,
        out_sample_profit_factor=1.6, out_sample_win_rate=45.0,
        out_sample_total_trades=40, out_sample_max_drawdown=8.0,
        composite_score=0.7, passed_walkforward=True,
    )
    assert r.out_sample_dsr == 0.0 and r.cpcv_frac_positive == 0.0
    assert eng_gate(r) is False


def eng_gate(r):
    return OptimizationEngine()._passes_promotion_gate(r)


def test_pre_cpcv_gate_includes_dsr_but_not_cpcv():
    """The cheap pre-CPCV screen (used to decide which candidates earn the
    expensive path eval) includes DSR but NOT the CPCV check."""
    eng = OptimizationEngine()
    # cpcv=0.0 (not yet computed) but everything else incl. DSR passes ->
    # pre-CPCV gate True, full gate False.
    r = _result(trades=40, pf=1.6, wr=45.0, dd=8.0, sharpe=1.5, dsr=0.99, cpcv=0.0)
    assert eng._passes_pre_cpcv_gate(r) is True
    assert eng._passes_promotion_gate(r) is False
