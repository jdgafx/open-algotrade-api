import pytest
from unittest.mock import AsyncMock, MagicMock
from src.services.risk_controller import RiskController
from src.services.risk_models import RiskConfig


def _rc():
    cfg = RiskConfig(absolute_floor_usd=50.0)
    executor = MagicMock()
    executor.emergency_close_all = AsyncMock()
    return RiskController(config=cfg, client=MagicMock(), executor=executor), executor


@pytest.mark.asyncio
async def test_floor_halts_and_blocks():
    rc, executor = _rc()
    await rc._check_absolute_floor(49.0)
    assert rc._absolute_floor_halt is True
    executor.emergency_close_all.assert_awaited_once()
    assert rc.can_open_new_position() is False


@pytest.mark.asyncio
async def test_above_floor_no_halt():
    rc, executor = _rc()
    await rc._check_absolute_floor(51.0)
    assert rc._absolute_floor_halt is False
    executor.emergency_close_all.assert_not_called()


@pytest.mark.asyncio
async def test_exactly_at_floor_halts():
    rc, executor = _rc()
    await rc._check_absolute_floor(50.0)  # condition is <=, so at-floor must halt
    assert rc._absolute_floor_halt is True
    executor.emergency_close_all.assert_awaited_once()


@pytest.mark.asyncio
async def test_floor_disabled_at_zero():
    cfg = RiskConfig(absolute_floor_usd=0.0)
    executor = MagicMock()
    executor.emergency_close_all = AsyncMock()
    rc = RiskController(config=cfg, client=MagicMock(), executor=executor)
    await rc._check_absolute_floor(0.0)
    assert rc._absolute_floor_halt is False
    executor.emergency_close_all.assert_not_called()
