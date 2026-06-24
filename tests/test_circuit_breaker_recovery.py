"""U4 — condition-aware circuit-breaker auto-recovery wired into the orchestrator.

Drives the orchestrator's `_try_shadow_recovery` directly (the loop branch calls it
each tick for a halted strategy) and asserts:
- a losing/below-bar shadow stream NEVER re-enables (the flip-flop-btc guarantee),
- a renewed-edge shadow stream re-enables exactly once and clears full breaker state,
- the toggle and min-sample gates behave,
- recovery never touches the real executor / balance.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from src.engine.orchestrator import StrategyOrchestrator
from src.engine.shadow_recovery import RECOVERY_MIN_TRADES
from src.strategies.base_strategy import (
    Signal, SignalType, StrategyConfig, StrategyState, StrategyTier,
)


class _ScriptedStrategy:
    """A halted strategy that emits a LONG when shadow-flat and a CLOSE_LONG when
    shadow-positioned. Win/loss is controlled by the exit price the test feeds via
    the data frame's close, not by the strategy."""

    def __init__(self, name="halted"):
        self.config = StrategyConfig(
            name=name, symbol="BTC", tier=StrategyTier.C, size_usd=1000.0,
        )
        self.state = StrategyState()
        self.state.circuit_breaker_triggered = True
        self.state.circuit_breaker_reason = "30 consecutive losses (test)"
        self.state.total_pnl = -50.0
        self.state.peak_pnl = 10.0
        self.state.max_drawdown = 60.0

    async def should_enter(self, data):
        return Signal(signal_type=SignalType.LONG, symbol="BTC", size_usd=1000.0)

    async def should_exit(self, data, position):
        return Signal(signal_type=SignalType.CLOSE_LONG, symbol="BTC", size_usd=1000.0)

    async def run_iteration(self, data, position=None):
        # Post-recovery / recovery-off ticks fall through to the real iteration path;
        # a no-op keeps the loop tests focused on the breaker branch behavior.
        return None


def _orch():
    client = MagicMock()
    executor = MagicMock()
    # AsyncMocks so any accidental real-executor call would be observable.
    executor.execute_signal = AsyncMock()
    executor.get_position = AsyncMock(return_value=None)
    return StrategyOrchestrator(client=client, executor=executor)


def _data(price):
    return pd.DataFrame({"close": [price]})


async def _drive_trades(orch, name, strat, n, *, win):
    """Drive n full shadow round-trips. Entry at 100; exit at 110 (win) or 95 (loss)."""
    exit_price = 110.0 if win else 95.0
    for _ in range(n):
        await orch._try_shadow_recovery(name, strat, _data(100.0))   # entry tick
        if strat.state.circuit_breaker_triggered:
            await orch._try_shadow_recovery(name, strat, _data(exit_price))  # exit tick
        else:
            break  # re-enabled mid-stream; stop driving


@pytest.mark.asyncio
async def test_losing_stream_never_reenables():
    """The flip-flop-btc guarantee: no edge => stays halted across many ticks."""
    orch = _orch()
    strat = _ScriptedStrategy()
    orch._strategies["halted"] = strat

    await _drive_trades(orch, "halted", strat, n=RECOVERY_MIN_TRADES + 10, win=False)

    assert strat.state.circuit_breaker_triggered is True


@pytest.mark.asyncio
async def test_renewed_edge_reenables_and_clears_state():
    orch = _orch()
    strat = _ScriptedStrategy()
    orch._strategies["halted"] = strat

    await _drive_trades(orch, "halted", strat, n=RECOVERY_MIN_TRADES + 5, win=True)

    assert strat.state.circuit_breaker_triggered is False
    assert strat.state.circuit_breaker_reason == ""
    assert strat.state.max_drawdown == 0.0
    assert strat.state.consecutive_losses == 0
    # Shadow state cleared on re-enable.
    assert orch._shadow.window_count("halted") == 0


@pytest.mark.asyncio
async def test_insufficient_sample_stays_halted():
    """Positive but too few shadow trades => no premature re-enable on a lucky streak."""
    orch = _orch()
    strat = _ScriptedStrategy()
    orch._strategies["halted"] = strat

    await _drive_trades(orch, "halted", strat, n=RECOVERY_MIN_TRADES - 1, win=True)

    assert strat.state.circuit_breaker_triggered is True


@pytest.mark.asyncio
async def test_recovery_never_touches_real_executor():
    orch = _orch()
    strat = _ScriptedStrategy()
    orch._strategies["halted"] = strat

    await _drive_trades(orch, "halted", strat, n=RECOVERY_MIN_TRADES + 5, win=True)

    orch.executor.execute_signal.assert_not_called()


@pytest.mark.asyncio
async def test_toggle_reflects_env(monkeypatch):
    from src.engine import shadow_recovery

    monkeypatch.setenv("CIRCUIT_BREAKER_AUTO_RECOVERY", "0")
    assert shadow_recovery.auto_recovery_enabled() is False
    monkeypatch.setenv("CIRCUIT_BREAKER_AUTO_RECOVERY", "1")
    assert shadow_recovery.auto_recovery_enabled() is True


# ── full-loop integration: the live-trip reachability fix (and its toggle guard) ──

def _alternating_ohlcv():
    """get_ohlcv stub: odd calls price 100 (shadow entry), even calls 110 (winning
    exit), so the shadow ledger accrues net wins over successive ticks."""
    counter = {"i": 0}

    def _fn(symbol, interval, lookback):
        counter["i"] += 1
        price = 100.0 if counter["i"] % 2 == 1 else 110.0
        return pd.DataFrame({
            "open": [price], "high": [price], "low": [price],
            "close": [price], "volume": [1.0],
        })

    return _fn


def _loop_orch(strat):
    orch = _orch()
    # Neutralize portfolio-level gates that would otherwise need a real balance feed.
    orch._check_daily_loss = lambda: False
    orch._check_weekly_drawdown = lambda: None
    orch._check_monthly_drawdown = lambda: False
    orch._check_regime_gate = lambda *a, **k: True
    orch.client.get_ohlcv = _alternating_ohlcv()
    orch.executor.get_position = AsyncMock(return_value=None)
    orch._strategies[strat.config.name] = strat
    return orch


@pytest.mark.asyncio
async def test_live_trip_keeps_looping_until_recovered(monkeypatch):
    """REGRESSION: a strategy that trips during a live session (not via redeploy
    rehydration) must stay in its loop and recover through shadow eval. Before the
    fix, the post-trade trip `break`d the loop task, making recovery unreachable for
    live trips."""
    monkeypatch.setenv("CIRCUIT_BREAKER_AUTO_RECOVERY", "1")
    strat = _ScriptedStrategy("halted")
    strat.config.interval_seconds = 0  # spin fast
    orch = _loop_orch(strat)

    task = asyncio.create_task(orch._run_strategy_loop("halted", strat))
    try:
        for _ in range(600):  # up to ~3s real time for the loop to accrue + recover
            if not strat.state.circuit_breaker_triggered:
                break
            await asyncio.sleep(0.005)
    finally:
        task.cancel()
        try:
            await task  # loop catches CancelledError and breaks cleanly
        except asyncio.CancelledError:
            pass

    # Recovered through the LIVE loop, not a redeploy.
    assert strat.state.circuit_breaker_triggered is False
    assert orch._shadow.window_count("halted") == 0  # cleared on re-enable


@pytest.mark.asyncio
async def test_toggle_off_loop_stops_and_skips_shadow(monkeypatch):
    """With auto-recovery OFF, a tripped strategy keeps the legacy behavior: the loop
    stops (task completes) and no shadow evaluation runs."""
    monkeypatch.setenv("CIRCUIT_BREAKER_AUTO_RECOVERY", "0")
    strat = _ScriptedStrategy("halted")
    strat.config.interval_seconds = 0
    orch = _loop_orch(strat)

    # Loop should exit on its own (break) rather than run forever.
    await asyncio.wait_for(orch._run_strategy_loop("halted", strat), timeout=2.0)

    assert strat.state.circuit_breaker_triggered is True   # never auto-recovered
    assert orch._shadow.window_count("halted") == 0        # shadow eval never ran


@pytest.mark.asyncio
async def test_get_recovery_status_surfaces_halted_only():
    orch = _orch()
    halted = _ScriptedStrategy("halted")
    healthy = _ScriptedStrategy("healthy")
    healthy.state.circuit_breaker_triggered = False
    orch._strategies = {"halted": halted, "healthy": healthy}

    await _drive_trades(orch, "halted", halted, n=3, win=False)

    status = orch.get_recovery_status()
    names = {row["strategy"] for row in status}
    assert names == {"halted"}              # healthy strategy not listed
    row = status[0]
    assert row["shadow_trades"] == 3
    assert row["recovery_min_trades"] == RECOVERY_MIN_TRADES
    assert row["clears_recovery_bar"] is False
