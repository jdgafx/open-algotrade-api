"""Unit tests for build_rbi_job_specs — the DB-derived RBI scheduler helper."""
from types import SimpleNamespace

from src.engine.rbi_schedule import build_rbi_job_specs


def _inst(stype, iid, sym="BTC", tf="1h"):
    return SimpleNamespace(strategy_type=stype, id=iid, symbol=sym, timeframe=tf, status="running", enabled=True)


def test_filters_unsupported_types():
    insts = [_inst("donchian", 1), _inst("sma_crossover", 2)]
    specs = build_rbi_job_specs(insts, supported_types={"sma_crossover"})
    assert len(specs) == 1
    assert specs[0]["strategy_type"] == "sma_crossover"
    assert specs[0]["strategy_id"] == 2


def test_cadence_by_timeframe():
    specs = build_rbi_job_specs([_inst("rsi", 3, tf="15m")], supported_types={"rsi"})
    assert specs[0]["hours"] == 4   # intraday re-optimizes more often


def test_covers_all_supported_instances():
    insts = [_inst("rsi", 1), _inst("macd", 2), _inst("adx", 3)]
    specs = build_rbi_job_specs(insts, supported_types={"rsi", "macd", "adx"})
    assert {s["strategy_id"] for s in specs} == {1, 2, 3}


def test_empty_instances_returns_empty():
    specs = build_rbi_job_specs([], supported_types={"rsi", "macd"})
    assert specs == []


def test_all_unsupported_returns_empty():
    insts = [_inst("flip_flop", 1), _inst("liquidation_dip", 2)]
    specs = build_rbi_job_specs(insts, supported_types={"rsi", "macd"})
    assert specs == []


def test_cadence_defaults_for_unknown_timeframe():
    specs = build_rbi_job_specs([_inst("rsi", 5, tf="3h")], supported_types={"rsi"})
    assert specs[0]["hours"] == 8   # default fallback


def test_4h_cadence():
    specs = build_rbi_job_specs([_inst("macd", 6, tf="4h")], supported_types={"macd"})
    assert specs[0]["hours"] == 12


def test_spec_fields_correct():
    inst = _inst("bollinger", 99, sym="ETH", tf="1h")
    specs = build_rbi_job_specs([inst], supported_types={"bollinger"})
    assert specs[0] == {
        "strategy_type": "bollinger",
        "strategy_id": 99,
        "symbol": "ETH",
        "timeframe": "1h",
        "hours": 8,
    }


def test_custom_hours_by_tf_override():
    custom = {"1h": 6}
    specs = build_rbi_job_specs([_inst("rsi", 7, tf="1h")], supported_types={"rsi"}, default_hours_by_tf=custom)
    assert specs[0]["hours"] == 6
