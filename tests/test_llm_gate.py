import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
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
    ctx.signal_strength = 0.3
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
    """Even when LLM says don't proceed, soft mode passes it through."""
    with patch.dict(os.environ, {"RBI_LLM_GATE": "soft", "OPENROUTER_API_KEY": "sk-test"}):
        gate = LLMGate()

    fake_raw = {"proceed": False, "confidence": 0.8, "reason": "counter trend"}

    import asyncio
    with patch("src.engine.llm_gate._call_openrouter", new=AsyncMock(return_value=fake_raw)):
        verdict = asyncio.run(gate.evaluate(_ctx()))

    assert verdict.proceed is True
    assert "soft:" in verdict.reason


def test_llm_error_defaults_to_proceed():
    """Network errors must not block trade execution."""
    with patch.dict(os.environ, {"RBI_LLM_GATE": "soft", "OPENROUTER_API_KEY": "sk-test"}):
        gate = LLMGate()

    import asyncio
    with patch("src.engine.llm_gate._call_openrouter", new=AsyncMock(side_effect=Exception("API error"))):
        verdict = asyncio.run(gate.evaluate(_ctx()))

    assert verdict.proceed is True
    assert "error:" in verdict.reason


def test_memory_context_included_in_prompt():
    """Memory context from Supermemory is passed into the LLM prompt."""
    with patch.dict(os.environ, {"RBI_LLM_GATE": "soft", "OPENROUTER_API_KEY": "sk-test"}):
        gate = LLMGate()

    ctx = _ctx()
    ctx.memory_context = "[TRADE] 2026-05-15 | mean_reversion | LONG BTC | PnL: $12.50 (WIN)"

    captured_prompt = {}

    async def fake_call(prompt, model):
        captured_prompt["prompt"] = prompt
        return {"proceed": True, "confidence": 0.9, "reason": "aligned"}

    import asyncio
    # evaluate() reads OPENROUTER_API_KEY at call-time to choose the OpenRouter path
    # over the Anthropic fallback, so the key must be set around the call (not only
    # around construction) for the patched _call_openrouter to be exercised.
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test"}), \
            patch("src.engine.llm_gate._call_openrouter", new=fake_call):
        asyncio.run(gate.evaluate(ctx))

    assert "Past similar trades" in captured_prompt["prompt"]
    assert "WIN" in captured_prompt["prompt"]
