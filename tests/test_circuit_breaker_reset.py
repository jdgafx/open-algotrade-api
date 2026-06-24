"""U1 — Circuit-breaker reset must clear the FULL breaker state.

Regression coverage for the bug where reset cleared `consecutive_losses` but left
`max_drawdown`/`peak_pnl` intact, so a drawdown-tripped strategy re-tripped on its
first losing trade after a reset.
"""
from unittest.mock import MagicMock

import pytest

from src.strategies.base_strategy import StrategyConfig, StrategyTier
from src.strategies.registry import create_strategy


def _strategy(name="cb-test", **params):
    config = StrategyConfig(
        name=name, symbol="BTC", tier=StrategyTier.C, leverage=2, size_usd=1000.0,
        params=params,
    )
    return create_strategy("rsi", config)


def _trip_by_drawdown(strategy, limit=25.0):
    """Drive the strategy into a drawdown-tripped state: build a peak, then give
    back more than `limit` so max_drawdown >= limit trips the breaker."""
    strategy.config.params["max_strategy_drawdown"] = limit
    # Avoid the consecutive-loss breaker firing first.
    strategy.config.params["max_consecutive_losses"] = 999
    strategy.record_trade(+30.0)   # peak_pnl -> 30, total_pnl -> 30
    strategy.record_trade(-30.0)   # total_pnl -> 0, drawdown = 30 >= 25 -> trip
    return strategy


# ── base-level: the helper + the regression ────────────────────────────────

class TestResetCircuitBreakerHelper:
    def test_drawdown_trip_then_reset_then_loss_does_not_retrip(self):
        """THE regression: a drawdown-tripped strategy, reset, must not re-trip on
        the first subsequent loss."""
        s = _trip_by_drawdown(_strategy())
        assert s.state.circuit_breaker_triggered is True
        assert s.state.max_drawdown >= 25.0

        s.state.reset_circuit_breaker()

        assert s.state.circuit_breaker_triggered is False
        assert s.state.circuit_breaker_reason == ""
        assert s.state.consecutive_losses == 0
        assert s.state.max_drawdown == 0.0
        # peak_pnl is rebased to current total_pnl so drawdown starts at 0.
        assert s.state.peak_pnl == s.state.total_pnl

        # One more losing trade must NOT immediately re-trip the breaker.
        s.record_trade(-5.0)
        assert s.state.circuit_breaker_triggered is False

    def test_reset_preserves_realized_track_record(self):
        """Reset clears the halt, not the strategy's realized PnL / trade counters."""
        s = _trip_by_drawdown(_strategy())
        trades_before = s.state.total_trades
        pnl_before = s.state.total_pnl

        s.state.reset_circuit_breaker()

        assert s.state.total_trades == trades_before
        assert s.state.total_pnl == pnl_before

    def test_consecutive_loss_trip_then_reset(self):
        s = _strategy(max_consecutive_losses=3, max_strategy_drawdown=9999.0)
        for _ in range(3):
            s.record_trade(-1.0)
        assert s.state.circuit_breaker_triggered is True
        assert s.state.consecutive_losses >= 3

        s.state.reset_circuit_breaker()

        assert s.state.circuit_breaker_triggered is False
        assert s.state.consecutive_losses == 0
        # A single subsequent loss must not re-trip (counter genuinely reset).
        s.record_trade(-1.0)
        assert s.state.circuit_breaker_triggered is False

    def test_reset_on_untripped_strategy_is_safe(self):
        s = _strategy()
        s.state.reset_circuit_breaker()  # idempotent no-op
        assert s.state.circuit_breaker_triggered is False


# ── endpoint-level: routing + response ─────────────────────────────────────

class TestResetCircuitBreakerEndpoint:
    def test_reset_endpoint_clears_full_state(self, client):
        from src.engine.orchestrator import StrategyOrchestrator

        orch = StrategyOrchestrator(client=MagicMock(), executor=MagicMock())
        strat = _trip_by_drawdown(_strategy(name="ep-cb"))
        orch._strategies["ep-cb"] = strat
        client.app.state.orchestrator = orch

        resp = client.post("/strategies/ep-cb/circuit-breaker/reset")
        assert resp.status_code == 200
        data = resp.json()
        assert data["circuit_breaker_was_triggered"] is True
        assert data["circuit_breaker_reset"] is True

        # The live strategy state was fully reset, not just consecutive_losses.
        assert strat.state.circuit_breaker_triggered is False
        assert strat.state.max_drawdown == 0.0
        strat.record_trade(-5.0)
        assert strat.state.circuit_breaker_triggered is False

    def test_reset_endpoint_unknown_strategy_404(self, client):
        from src.engine.orchestrator import StrategyOrchestrator

        orch = StrategyOrchestrator(client=MagicMock(), executor=MagicMock())
        client.app.state.orchestrator = orch
        resp = client.post("/strategies/no-such/circuit-breaker/reset")
        assert resp.status_code == 404

    def test_reset_endpoint_no_orchestrator_503(self, client):
        client.app.state.orchestrator = None
        resp = client.post("/strategies/anything/circuit-breaker/reset")
        assert resp.status_code == 503
