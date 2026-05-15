"""Schema-level regression tests."""
from src.api.schemas import StrategyInstanceUpdate


def test_strategy_instance_update_accepts_lookback_days():
    """PATCH /strategies/{name} must accept lookback_days.

    Regression for 2026-05-15: live PATCH returned HTTP 200 but did not
    update the field because the Pydantic schema did not declare it, so
    model_dump(exclude_unset=True) stripped it silently. Add to schema.
    """
    u = StrategyInstanceUpdate(lookback_days=14)
    assert u.lookback_days == 14
    dumped = u.model_dump(exclude_unset=True)
    assert "lookback_days" in dumped
    assert dumped["lookback_days"] == 14


def test_strategy_instance_update_other_fields_still_work():
    """Schema must continue to accept the other known fields."""
    u = StrategyInstanceUpdate(enabled=False, leverage=5, size_usd=200.0)
    d = u.model_dump(exclude_unset=True)
    assert d == {"enabled": False, "leverage": 5, "size_usd": 200.0}
