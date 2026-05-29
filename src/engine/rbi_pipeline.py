import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from .optimizer import OptimizationEngine

logger = logging.getLogger(__name__)


@dataclass
class PromotionEvent:
    strategy_type: str
    strategy_id: int
    timestamp: str
    promoted: bool
    reason: str
    before_params: dict[str, Any]
    after_params: dict[str, Any]
    before_metrics: dict[str, Any]
    after_metrics: dict[str, Any]


class RBIPipeline:
    def __init__(
        self,
        get_strategy_fn: Callable,
        patch_strategy_fn: Callable,
        optimizer: Optional[OptimizationEngine] = None,
    ):
        self._get_strategy_fn = get_strategy_fn
        self._patch_strategy_fn = patch_strategy_fn
        self._optimizer = optimizer or OptimizationEngine(commission_pct=0.14)  # 2x the 0.07% research default — worst-case slippage so we never promote on rosy costs
        self._history: list[PromotionEvent] = []
        self._previous_params: dict[str, dict] = {}

    async def run_cycle(
        self,
        strategy_type: str,
        strategy_id: int,
        symbol: str,
        timeframe: str = "1h",
        lookback_days: int = 90,
        n_trials: int = 100,
    ) -> PromotionEvent:
        ts = datetime.now(timezone.utc).isoformat()

        current = await self._get_strategy_fn(strategy_id)
        if current.get("active_positions", 0) > 0:
            return PromotionEvent(
                strategy_type=strategy_type, strategy_id=strategy_id, timestamp=ts,
                promoted=False, reason="active_position",
                before_params=current.get("params", {}), after_params={},
                before_metrics={}, after_metrics={},
            )

        current_params = current.get("params", {})
        candidates = await self._optimizer.optimize(
            strategy_type=strategy_type, symbol=symbol,
            timeframe=timeframe, lookback_days=lookback_days, n_trials=n_trials,
        )

        passing = [c for c in candidates if self._optimizer._passes_promotion_gate(c)]
        if not passing:
            return PromotionEvent(
                strategy_type=strategy_type, strategy_id=strategy_id, timestamp=ts,
                promoted=False, reason="no_passing_candidates",
                before_params=current_params, after_params={},
                before_metrics={}, after_metrics={},
            )

        best = passing[0]
        self._previous_params[strategy_type] = current_params.copy()
        await self._patch_strategy_fn(strategy_id, {"params": best.params})

        event = PromotionEvent(
            strategy_type=strategy_type, strategy_id=strategy_id, timestamp=ts,
            promoted=True, reason="promotion_gate_passed",
            before_params=current_params, after_params=best.params,
            before_metrics={},
            after_metrics={
                "oos_sharpe": best.out_sample_sharpe,
                "oos_profit_factor": best.out_sample_profit_factor,
                "oos_win_rate": best.out_sample_win_rate,
                "oos_trades": best.out_sample_total_trades,
                "oos_max_drawdown": best.out_sample_max_drawdown,
                "composite_score": best.composite_score,
            },
        )
        self._history.append(event)
        logger.info(
            "RBI promoted %s (id=%d): OOS sharpe=%.2f PF=%.2f WR=%.1f%%",
            strategy_type, strategy_id, best.out_sample_sharpe,
            best.out_sample_profit_factor, best.out_sample_win_rate,
        )
        return event

    def get_rollback_params(self, strategy_type: str) -> Optional[dict]:
        return self._previous_params.get(strategy_type)

    def get_history(self) -> list[PromotionEvent]:
        return list(self._history)
