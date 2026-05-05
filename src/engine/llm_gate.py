import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional
import anthropic

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a crypto trading signal evaluator for a Hyperliquid algorithmic trading system.
Given a trade signal context, evaluate whether to proceed with the trade.
Respond ONLY with valid JSON: {"proceed": true|false, "confidence": 0.0-1.0, "reason": "one sentence"}
Be conservative. Reject signals that go counter to the stated market regime."""


@dataclass
class TradeContext:
    strategy: str
    signal: str       # "LONG" or "SHORT"
    symbol: str
    price: float
    regime: str
    signal_strength: float
    recent_pnl: list  # last 5 trade PnLs


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
    outcome: Optional[float] = None   # set when trade closes (PnL)


class LLMGate:
    MODE_OFF = "off"
    MODE_SOFT = "soft"
    MODE_HARD = "hard"

    def __init__(self):
        self._mode = os.getenv("RBI_LLM_GATE", "soft")
        self._client = anthropic.Anthropic() if self._mode != self.MODE_OFF else None
        self._evals: list[_EvalRecord] = []

    async def evaluate(self, context: TradeContext) -> GateVerdict:
        if self._mode == self.MODE_OFF or context.signal_strength < 0.5:
            return GateVerdict(proceed=True, confidence=1.0, reason="gate_off")

        prompt = (
            f"Strategy: {context.strategy}, Signal: {context.signal}, "
            f"Symbol: {context.symbol}, Price: {context.price:.2f}, "
            f"Regime: {context.regime}, Signal strength: {context.signal_strength:.2f}, "
            f"Recent PnL (last 5 trades): {context.recent_pnl}"
        )

        try:
            response = self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=100,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = json.loads(response.content[0].text)
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

        # Soft mode: always proceed but surface the advisory
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
        }


llm_gate = LLMGate()
