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
