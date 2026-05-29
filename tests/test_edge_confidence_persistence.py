from src.api.models import StrategyInstance
from src.api.schemas import StrategyInstanceOut, StrategyPerformanceItem


def test_strategy_instance_has_edge_confidence_column():
    cols = StrategyInstance.__table__.columns
    assert "edge_confidence_score" in cols
    assert cols["edge_confidence_score"].default.arg == 0.0


def test_out_schema_exposes_edge_confidence():
    assert "edge_confidence_score" in StrategyInstanceOut.model_fields


def test_performance_schema_exposes_edge_confidence():
    assert "edge_confidence_score" in StrategyPerformanceItem.model_fields
