import pytest
from unittest.mock import AsyncMock, MagicMock
from src.services.risk_controller import RiskController
from src.services.risk_models import RiskConfig


def _rc():
    cfg = RiskConfig(trailing_drawdown_from_peak_pct=35.0, absolute_floor_usd=50.0)
    executor = MagicMock()
    executor.emergency_close_all = AsyncMock()
    return RiskController(config=cfg, client=MagicMock(), executor=executor), executor


@pytest.mark.asyncio
async def test_init_flags_default_false():
    rc, _ = _rc()
    assert rc._trailing_drawdown_halt is False
    assert rc._absolute_floor_halt is False
    assert rc.running_peak_equity == 0.0


@pytest.mark.asyncio
async def test_peak_updates_then_trailing_halt_fires():
    rc, executor = _rc()
    await rc._check_trailing_drawdown_from_peak(300.0)
    assert rc.running_peak_equity == 300.0
    assert rc._trailing_drawdown_halt is False
    await rc._check_trailing_drawdown_from_peak(192.0)  # 36% below peak
    assert rc._trailing_drawdown_halt is True
    executor.emergency_close_all.assert_awaited_once()
    assert rc.can_open_new_position() is False


@pytest.mark.asyncio
async def test_shallow_drawdown_does_not_halt():
    rc, executor = _rc()
    await rc._check_trailing_drawdown_from_peak(300.0)
    await rc._check_trailing_drawdown_from_peak(210.0)  # 30% < 35%
    assert rc._trailing_drawdown_halt is False
    executor.emergency_close_all.assert_not_called()
