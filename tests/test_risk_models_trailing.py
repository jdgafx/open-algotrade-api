from src.services.risk_models import RiskConfig, RiskSnapshot, RiskStatus, RiskEventType


def test_riskconfig_trailing_and_floor_defaults():
    cfg = RiskConfig()
    assert cfg.trailing_drawdown_from_peak_pct == 35.0
    assert cfg.absolute_floor_usd == 50.0


def test_riskconfig_ruin_guard_buffer_default_and_bounds():
    import pytest
    from pydantic import ValidationError

    cfg = RiskConfig()
    assert cfg.ruin_guard_buffer_pct == 1.0
    # Bounds: ge=0.1, le=10.0
    assert RiskConfig(ruin_guard_buffer_pct=0.1).ruin_guard_buffer_pct == 0.1
    assert RiskConfig(ruin_guard_buffer_pct=10.0).ruin_guard_buffer_pct == 10.0
    with pytest.raises(ValidationError):
        RiskConfig(ruin_guard_buffer_pct=0.0)
    with pytest.raises(ValidationError):
        RiskConfig(ruin_guard_buffer_pct=10.1)


def test_risksnapshot_exposes_peak_and_trailing():
    snap = RiskSnapshot(status=RiskStatus.MONITORING, account_value=150.0,
                        running_peak_equity=200.0, trailing_drawdown_from_peak_pct=25.0)
    assert snap.running_peak_equity == 200.0
    assert snap.trailing_drawdown_from_peak_pct == 25.0


def test_new_event_types_exist():
    assert RiskEventType.TRAILING_DRAWDOWN_HALT.value == "TRAILING_DRAWDOWN_HALT"
    assert RiskEventType.ABSOLUTE_FLOOR_HALT.value == "ABSOLUTE_FLOOR_HALT"
