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


@pytest.mark.asyncio
async def test_get_snapshot_exposes_running_peak():
    cfg = RiskConfig(trailing_drawdown_from_peak_pct=35.0, absolute_floor_usd=50.0)
    executor = MagicMock()
    executor.emergency_close_all = AsyncMock()
    rc = RiskController(config=cfg, client=MagicMock(), executor=executor)
    rc.running_peak_equity = 250.0
    snap = rc.get_snapshot()  # fallback path (no live cycle has run)
    assert snap.running_peak_equity == 250.0


@pytest.mark.asyncio
async def test_check_cycle_populates_peak_and_trailing_in_snapshot():
    cfg = RiskConfig(trailing_drawdown_from_peak_pct=35.0, absolute_floor_usd=50.0)
    executor = MagicMock()
    executor.get_account_value = AsyncMock(return_value=210.0)  # 30% below 300 peak
    executor.get_all_positions = AsyncMock(return_value=[])
    executor.emergency_close_all = AsyncMock()
    rc = RiskController(config=cfg, client=MagicMock(), executor=executor)
    rc.running_peak_equity = 300.0
    rc._start_of_day_equity = 300.0
    await rc._check_cycle()
    snap = rc.get_snapshot()
    assert snap.running_peak_equity == 300.0
    assert snap.trailing_drawdown_from_peak_pct == 30.0
