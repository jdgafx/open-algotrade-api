import pytest
from unittest.mock import AsyncMock, MagicMock
from src.services.risk_controller import RiskController
from src.services.risk_models import RiskConfig


@pytest.mark.asyncio
async def test_check_cycle_triggers_account_level_halt():
    cfg = RiskConfig(trailing_drawdown_from_peak_pct=35.0, absolute_floor_usd=50.0)
    executor = MagicMock()
    executor.get_account_value = AsyncMock(return_value=120.0)
    executor.get_all_positions = AsyncMock(return_value=[])
    executor.emergency_close_all = AsyncMock()
    rc = RiskController(config=cfg, client=MagicMock(), executor=executor)
    rc.running_peak_equity = 200.0  # 120 is 40% below -> halt
    await rc._check_cycle()
    assert rc._trailing_drawdown_halt is True
    assert rc.can_open_new_position() is False
