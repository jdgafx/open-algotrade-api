"""U2 — POST /paper/reset must actually zero per-strategy anti-overtrading state.

The endpoint guarded its reset loop on `hasattr(orchestrator, "strategies")` and
iterated `orchestrator.strategies` — but the orchestrator attribute is `_strategies`,
so the guard was always False and the loop never ran. This characterizes that the
loop runs and zeroes the counters.
"""
from unittest.mock import MagicMock

from src.engine.orchestrator import StrategyOrchestrator
from src.strategies.base_strategy import StrategyConfig, StrategyTier
from src.strategies.registry import create_strategy


def _dirty_strategy(name="reset-me"):
    config = StrategyConfig(
        name=name, symbol="BTC", tier=StrategyTier.C, leverage=2, size_usd=1000.0,
    )
    s = create_strategy("rsi", config)
    # Simulate accumulated live state that /paper/reset is supposed to clear.
    s.state.total_trades = 7
    s.state.winning_trades = 3
    s.state.losing_trades = 4
    s.state.total_pnl = -12.5
    s.state.max_drawdown = 20.0
    s.state.peak_pnl = 7.5
    s.state.trades_this_hour = 5
    return s


def _wire(client):
    executor = MagicMock()
    executor.balance = 10000.0
    executor.initial_balance = 10000.0
    client.app.state.executor = executor
    client.app.state.paper_mode = True

    orch = StrategyOrchestrator(client=MagicMock(), executor=MagicMock())
    strat = _dirty_strategy()
    orch._strategies["reset-me"] = strat
    client.app.state.orchestrator = orch
    return executor, strat


class TestPaperResetStrategyState:
    def test_reset_zeroes_per_strategy_counters(self, client):
        """Proves the per-strategy reset loop actually executes (fails against the
        old `orchestrator.strategies` dead reference)."""
        executor, strat = _wire(client)

        resp = client.post("/paper/reset")
        assert resp.status_code == 200

        executor.reset.assert_called_once()
        assert strat.state.total_trades == 0
        assert strat.state.winning_trades == 0
        assert strat.state.losing_trades == 0
        assert strat.state.total_pnl == 0.0
        assert strat.state.max_drawdown == 0.0
        assert strat.state.peak_pnl == 0.0
        assert strat.state.trades_this_hour == 0

    def test_reset_without_orchestrator_still_resets_executor(self, client):
        executor = MagicMock()
        executor.balance = 10000.0
        executor.initial_balance = 10000.0
        client.app.state.executor = executor
        client.app.state.paper_mode = True
        client.app.state.orchestrator = None

        resp = client.post("/paper/reset")
        assert resp.status_code == 200
        executor.reset.assert_called_once()

    def test_reset_not_in_paper_mode_400(self, client):
        client.app.state.paper_mode = False
        client.app.state.executor = MagicMock()
        resp = client.post("/paper/reset")
        assert resp.status_code == 400
