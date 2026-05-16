"""
LLM advisory gate for trade signals.

Uses OpenRouter with Gemini 2.5 Flash as default (best financial reasoning, 1M ctx).
Falls back to Gemini 2.5 Flash Lite on timeout. Anthropic Haiku if no OpenRouter key.
Enriches prompts with recalled Supermemory trade history.
"""
import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_MODEL = "deepseek/deepseek-v4-flash:free"  # free, 1M ctx, DeepSeek V4
_FALLBACK_MODEL = "google/gemini-2.5-flash"  # $0.30/Mtok, fallback on timeout/error
_SUPERMEMORY_SEARCH_URL = "https://api.supermemory.ai/v3/search"
_SUPERMEMORY_ADD_URL = "https://api.supermemory.ai/v3/documents"

_SYSTEM_PROMPT = (
    "You are a crypto trading signal evaluator for a Hyperliquid algorithmic trading system. "
    "Given a trade signal context and optional past trade history, evaluate whether to proceed. "
    "Respond ONLY with valid JSON: "
    '{"proceed": true|false, "confidence": 0.0-1.0, "reason": "one sentence"} '
    "Be conservative. Validated rules (ordered by importance): "
    "1) Reject signals that go counter to the stated market regime — trend strategies fail in chop. "
    "2) ADX strategy signals are walk-forward validated (Sharpe 12.8, 60% WR) — favour them in TRENDING regime. "
    "3) If funding_bias=long_crowded and signal=LONG, reduce confidence (crowd already positioned). "
    "4) If funding_bias=short_crowded and signal=SHORT, reduce confidence. "
    "5) Contrarian trades against extreme funding bias have documented edge — slightly favour them. "
    "6) Reject if recent_pnl shows 3+ consecutive losses for this strategy (anti-tilt rule). "
    "7) Liquidation dip signals (strategy=liqdip*) in MEAN_REVERTING regime have high edge — favour. "
    "8) mean_reversion and consolidation_pop strategy types have poor track records — be skeptical. "
    "9) Low signal_strength < 0.55 is weak evidence — reduce confidence but don't auto-reject."
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
    memory_context: str = ""    # recalled similar trades from Supermemory
    funding_rate: Optional[float] = None   # annualized; positive = longs pay shorts
    funding_bias: str = "neutral"          # neutral | long_crowded | short_crowded


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


async def _recall_similar_trades(strategy: str, signal: str, symbol: str) -> str:
    """Search supermemory for past trades matching this strategy+signal+symbol combo."""
    api_key = os.getenv("SUPERMEMORY_API_KEY", "")
    if not api_key:
        return ""
    query = f"strategy={strategy} signal={signal} symbol={symbol} trade outcome"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.post(
                _SUPERMEMORY_SEARCH_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"q": query, "limit": 3},
            )
            if r.status_code != 200:
                return ""
            results = r.json().get("results", [])
            if not results:
                return ""
            snippets = [res.get("content", "")[:300] for res in results if res.get("content")]
            return "\n---\n".join(snippets[:3])
    except Exception as e:
        logger.debug("Supermemory recall skipped: %s", e)
        return ""


async def _save_trade_outcome(strategy: str, signal: str, symbol: str, proceed: bool, pnl: float) -> None:
    """Persist a completed gate evaluation + outcome to supermemory for future recall."""
    api_key = os.getenv("SUPERMEMORY_API_KEY", "")
    if not api_key:
        return
    outcome_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
    gate_str = "ALLOWED" if proceed else "BLOCKED"
    content = (
        f"LLM gate trade outcome: strategy={strategy} signal={signal} symbol={symbol} "
        f"gate={gate_str} pnl={outcome_str}. "
        f"{'Gate was correct (blocked a loser).' if not proceed and pnl < 0 else ''}"
        f"{'Gate was wrong (blocked a winner).' if not proceed and pnl > 0 else ''}"
        f"{'Trade allowed, profitable.' if proceed and pnl > 0 else ''}"
        f"{'Trade allowed, was a loss.' if proceed and pnl < 0 else ''}"
    )
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            await client.post(
                _SUPERMEMORY_ADD_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"content": content, "metadata": {"tags": ["open-algotrade-3", "llm-gate", strategy]}},
            )
    except Exception as e:
        logger.debug("Supermemory save skipped: %s", e)


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

        # Recall similar past trades if not already provided
        if not context.memory_context:
            context.memory_context = await _recall_similar_trades(
                context.strategy, context.signal, context.symbol
            )

        funding_str = ""
        if context.funding_rate is not None:
            funding_str = (
                f", Funding rate (annualized): {context.funding_rate:.2%}"
                f", Funding bias: {context.funding_bias}"
            )
        prompt = (
            f"Strategy: {context.strategy}, Signal: {context.signal}, "
            f"Symbol: {context.symbol}, Price: {context.price:.2f}, "
            f"Regime: {context.regime}, Signal strength: {context.signal_strength:.2f}, "
            f"Recent PnL (last 5): {context.recent_pnl}{funding_str}"
        )
        if context.memory_context:
            prompt += f"\n\nPast similar trades from memory:\n{context.memory_context}"

        try:
            if os.getenv("OPENROUTER_API_KEY"):
                try:
                    raw = await _call_openrouter(prompt, self._model)
                except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                    logger.warning("LLMGate primary model timeout/error (%s) — falling back to %s", e, _FALLBACK_MODEL)
                    raw = await _call_openrouter(prompt, _FALLBACK_MODEL)
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
        matched_proceed = True
        for rec in reversed(self._evals):
            if rec.strategy == strategy and rec.signal == signal and rec.outcome is None:
                rec.outcome = pnl
                matched_proceed = rec.proceed
                break
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(_save_trade_outcome(strategy, signal, "", matched_proceed, pnl))
        except Exception:
            pass

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
