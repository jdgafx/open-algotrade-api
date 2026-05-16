"""Smoke tests for FastAPI app configuration and router registration."""
from types import SimpleNamespace

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Dynamic winner-set selection logic (mirrors compounder in main.py)
# ---------------------------------------------------------------------------

_STATIC_WINNERS = {'vwap-btc', 'rsi-btc', 'pivot-btc', 'turtle-btc'}
_WINNER_MIN_TRADES = 3


def _select_winners(running):
    """Extracted winner selection logic — keeps tests decoupled from the closure."""
    dynamic = {
        r.name for r in running
        if (r.total_pnl or 0) > 0 and (r.total_trades or 0) >= _WINNER_MIN_TRADES
    }
    return dynamic if dynamic else _STATIC_WINNERS


def _inst(name, total_pnl=0.0, total_trades=0):
    return SimpleNamespace(name=name, total_pnl=total_pnl, total_trades=total_trades)


def test_dynamic_winners_from_profitable_strategies():
    running = [
        _inst("vwap-btc", total_pnl=12.5, total_trades=5),
        _inst("flip-flop-btc", total_pnl=3.0, total_trades=4),
        _inst("turtle-btc", total_pnl=-1.0, total_trades=6),  # losing
        _inst("rsi-btc", total_pnl=0.0, total_trades=10),    # break-even
    ]
    winners = _select_winners(running)
    assert winners == {"vwap-btc", "flip-flop-btc"}


def test_dynamic_winners_requires_min_trades():
    running = [
        _inst("vwap-btc", total_pnl=50.0, total_trades=2),  # not enough trades
        _inst("rsi-btc", total_pnl=10.0, total_trades=3),   # exactly at threshold
    ]
    winners = _select_winners(running)
    assert winners == {"rsi-btc"}


def test_fallback_to_static_when_no_dynamic_winners():
    running = [
        _inst("vwap-btc", total_pnl=0.0, total_trades=0),
        _inst("flip-flop-btc", total_pnl=-5.0, total_trades=10),
    ]
    winners = _select_winners(running)
    assert winners == _STATIC_WINNERS


def test_none_pnl_and_trades_handled_safely():
    running = [_inst("vwap-btc", total_pnl=None, total_trades=None)]
    winners = _select_winners(running)
    assert winners == _STATIC_WINNERS


def test_app_has_rbi_optimize_routes():
    from src.api.main import app
    routes = [r.path for r in app.routes]
    assert any("/optimize/rbi" in p for p in routes), f"RBI optimize routes missing from {routes}"


def test_app_has_llm_gate_route():
    from src.api.main import app
    routes = [r.path for r in app.routes]
    assert any("/optimize/llm-gate" in p for p in routes), f"LLM gate route missing from {routes}"
