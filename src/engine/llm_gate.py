"""
LLM advisory gate for trade signals.

Uses OpenRouter (cheap inference) with DeepSeek as default model.
Falls back to Anthropic Haiku if OPENROUTER_API_KEY is absent.
Enriches prompts with recalled Supermemory trade history.
"""
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_MODEL = "deepseek/deepseek-chat-v3-0324"

_SYSTEM_PROMPT = (
    "You are a crypto trading signal evaluator for a Hyperliquid algorithmic trading system. "
    "Given a trade signal context and optional past trade history, evaluate whether to proceed. "
    "Respond ONLY with valid JSON: "
    '{"proceed": true|false, "confidence": 0.0-1.0, "reason": "one sentence"} '
    "Be conservative. Reject signals that go counter to the stated market regime."
)


@dataclass
class TradeContext:
    strategy: str
    signal: str           # "LONG" or "SHORT"
    symbol: str
    price: float
    regime: str
    signal_strength: float
    recent_pnl: list      # last 5 trade PnLs
    memory_context: str = ""  # recalled similar trades from Supermemory


@dataclass
class GateVerdict:
    proceed: bool
    confidence: float
    reason: str


@dataclass
class _EvalRecord:
    strategy: str
    signal: str
    proceed: bool
    confidence: float
    reason: str
    outcome: Optional[float] = None


async def _call_openrouter(prompt: str, model: str) -> dict:
    """Call OpenRouter API and return parsed JSON dict."""
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://api-production-1302.up.railway.app",
    }
    payload = {
        "model": model,
        "max_tokens": 120,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    async with httpx.AsyncClient(timeout=8.0) as client:
        r = await client.post(_OPENROUTER_URL, headers=headers, json=payload)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        return json.loads(content)


async def _call_anthropic_fallback(prompt: str) -> dict:
    """Anthropic Haiku fallback when OpenRouter key is absent."""
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(response.content[0].text)


class LLMGate:
    MODE_OFF = "off"
    MODE_SOFT = "soft"
    MODE_HARD = "hard"

    def __init__(self) -> None:
        self._mode = os.getenv("RBI_LLM_GATE", "soft")
        self._model = os.getenv("RBI_LLM_MODEL", _DEFAULT_MODEL)
        self._evals: list[_EvalRecord] = []

    async def evaluate(self, context: TradeContext) -> GateVerdict:
        if self._mode == self.MODE_OFF or context.signal_strength < 0.5:
            return GateVerdict(proceed=True, confidence=1.0, reason="gate_off")

        prompt = (
            f"Strategy: {context.strategy}, Signal: {context.signal}, "
            f"Symbol: {context.symbol}, Price: {context.price:.2f}, "
            f"Regime: {context.regime}, Signal strength: {context.signal_strength:.2f}, "
            f"Recent PnL (last 5): {context.recent_pnl}"
        )
        if context.memory_context:
            prompt += f"\n\nPast similar trades from memory:\n{context.memory_context}"

        try:
            if os.getenv("OPENROUTER_API_KEY"):
                raw = await _call_openrouter(prompt, self._model)
            else:
                raw = await _call_anthropic_fallback(prompt)
            llm_proceed = bool(raw.get("proceed", True))
            confidence = float(raw.get("confidence", 0.5))
            reason = str(raw.get("reason", ""))
        except Exception as e:
            logger.warning("LLMGate error: %s — defaulting proceed=True", e)
            llm_proceed, confidence, reason = True, 0.5, f"error:{e}"

        self._evals.append(_EvalRecord(
            strategy=context.strategy, signal=context.signal,
            proceed=llm_proceed, confidence=confidence, reason=reason,
        ))

        if self._mode == self.MODE_HARD and not llm_proceed:
            return GateVerdict(proceed=False, confidence=confidence, reason=reason)

        return GateVerdict(proceed=True, confidence=confidence, reason=f"soft:{reason}")

    def record_outcome(self, strategy: str, signal: str, pnl: float) -> None:
        for rec in reversed(self._evals):
            if rec.strategy == strategy and rec.signal == signal and rec.outcome is None:
                rec.outcome = pnl
                break

    def get_stats(self) -> dict:
        total = len(self._evals)
        advised_against = sum(1 for e in self._evals if not e.proceed)
        with_outcome = [e for e in self._evals if e.outcome is not None and not e.proceed]
        right = sum(1 for e in with_outcome if e.outcome < 0)
        wrong = len(with_outcome) - right
        return {
            "total_evaluations": total,
            "advised_against": advised_against,
            "advised_against_and_right": right,
            "advised_against_and_wrong": wrong,
            "accuracy": right / max(len(with_outcome), 1),
            "model": self._model,
            "mode": self._mode,
        }


llm_gate = LLMGate()
