import json
import os
import pytest
from unittest.mock import MagicMock, patch
from src.engine.llm_gate import LLMGate, TradeContext, GateVerdict


def _ctx():
    return TradeContext(
        strategy="mean_reversion", signal="LONG", symbol="BTC",
        price=95000.0, regime="trending_up", signal_strength=0.7,
        recent_pnl=[1.5, -0.8, 2.1, -1.2, 0.5],
    )


def test_gate_off_always_proceeds():
    with patch.dict(os.environ, {"RBI_LLM_GATE": "off"}):
        gate = LLMGate()
    import asyncio
    verdict = asyncio.run(gate.evaluate(_ctx()))
    assert verdict.proceed is True
    assert verdict.reason == "gate_off"


def test_gate_off_for_weak_signal():
    with patch.dict(os.environ, {"RBI_LLM_GATE": "soft"}):
        gate = LLMGate()
    ctx = _ctx()
    ctx.signal_strength = 0.3  # below 0.5 threshold
    import asyncio
    verdict = asyncio.run(gate.evaluate(ctx))
    assert verdict.proceed is True
    assert verdict.reason == "gate_off"


def test_stats_empty():
    with patch.dict(os.environ, {"RBI_LLM_GATE": "off"}):
        gate = LLMGate()
    stats = gate.get_stats()
    assert stats["total_evaluations"] == 0
    assert stats["advised_against"] == 0


def test_soft_mode_always_returns_proceed_true():
    with patch.dict(os.environ, {"RBI_LLM_GATE": "soft"}):
        gate = LLMGate()

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps({"proceed": False, "confidence": 0.8, "reason": "counter trend"}))]

    with patch.object(gate, "_client") as mock_client:
        mock_client.messages.create.return_value = mock_response
        import asyncio
        verdict = asyncio.run(gate.evaluate(_ctx()))

    assert verdict.proceed is True
    assert "soft:" in verdict.reason


def test_llm_error_defaults_to_proceed():
    with patch.dict(os.environ, {"RBI_LLM_GATE": "soft"}):
        gate = LLMGate()
    with patch.object(gate, "_client") as mock_client:
        mock_client.messages.create.side_effect = Exception("API error")
        import asyncio
        verdict = asyncio.run(gate.evaluate(_ctx()))
    assert verdict.proceed is True
