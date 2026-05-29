"""Test that the orchestrator's ruin guard wiring calls close_by_strategy."""
import asyncio
import pandas as pd
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.engine.orchestrator import StrategyOrchestrator
from src.strategies.base_strategy import StrategyConfig, StrategyTier


class _StopLoop(BaseException):
    """Sentinel: raised by the mocked asyncio.sleep to escape the while-True loop.

    Must subclass BaseException (not Exception) so it bypasses the broad
    `except Exception` handler inside _run_strategy_loop and propagates out.
    """


@pytest.mark.asyncio
async def test_ruin_guard_triggers_close_by_strategy():
    """When check_position_ruin returns (True, 0.4), close_by_strategy must be awaited."""

    # ── Minimal mocks ──
    non_empty_df = pd.DataFrame({
        "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [1000.0]
    })
    mock_client = MagicMock()
    mock_client.get_ohlcv = MagicMock(return_value=non_empty_df)

    mock_executor = MagicMock()
    mock_executor.get_position = AsyncMock(return_value={"symbol": "BTC", "side": "long"})
    mock_executor.check_position_ruin = AsyncMock(return_value=(True, 0.4))
    mock_executor.close_by_strategy = AsyncMock(return_value=[])
    # hasattr guard in loop: MagicMock auto-creates attributes, so this is always True
    assert hasattr(mock_executor, "check_position_ruin")

    orch = StrategyOrchestrator(client=mock_client, executor=mock_executor)

    # Minimal strategy mock with async on_error
    mock_strategy = MagicMock()
    config = StrategyConfig(
        name="test-btc", symbol="BTC", tier=StrategyTier.A,
        timeframe="1h", interval_seconds=1, lookback_days=1,
    )
    mock_strategy.config = config
    mock_strategy.run_iteration = AsyncMock(return_value=None)
    mock_strategy.on_error = AsyncMock()

    # Patch orchestrator methods to avoid daily-loss halt and regime gate noise
    orch._check_daily_loss = MagicMock(return_value=False)
    orch._check_regime_gate = MagicMock(return_value=True)

    async def _mock_sleep(seconds):
        # Escape the while-True loop as soon as close_by_strategy has been called
        if mock_executor.close_by_strategy.await_count >= 1:
            raise _StopLoop("escape while-True")

    # AsyncMock side_effect for to_thread: calls fn(*args) directly (synchronously)
    async def _to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with (
        patch("random.uniform", return_value=0.0),  # suppress startup jitter (jitter=0.0 < 0.5)
        patch("src.engine.orchestrator.asyncio.sleep", side_effect=_mock_sleep),
        patch("src.engine.orchestrator.asyncio.to_thread", side_effect=_to_thread),
    ):
        with pytest.raises(_StopLoop):
            await orch._run_strategy_loop("test-btc", mock_strategy)

    mock_executor.close_by_strategy.assert_awaited_once_with("test-btc")
    # run_iteration must NOT have been called (ruin guard pre-empts it)
    mock_strategy.run_iteration.assert_not_awaited()
