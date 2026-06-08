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


def _record_to_event(row: Any) -> PromotionEvent:
    """Convert a PromotionEventRecord ORM row to a PromotionEvent dataclass."""
    return PromotionEvent(
        strategy_type=row.strategy_type,
        strategy_id=row.strategy_id,
        timestamp=row.timestamp,
        promoted=row.promoted,
        reason=row.reason,
        before_params=row.before_params or {},
        after_params=row.after_params or {},
        before_metrics=row.before_metrics or {},
        after_metrics=row.after_metrics or {},
    )


class RBIPipeline:
    def __init__(
        self,
        get_strategy_fn: Callable,
        patch_strategy_fn: Callable,
        optimizer: Optional[OptimizationEngine] = None,
        db_session_factory: Optional[Callable] = None,
        strategy_type: Optional[str] = None,
    ):
        self._get_strategy_fn = get_strategy_fn
        self._patch_strategy_fn = patch_strategy_fn
        self._optimizer = optimizer or OptimizationEngine(commission_pct=0.14)  # 2x the 0.07% research default — worst-case slippage so we never promote on rosy costs
        self._db_session_factory = db_session_factory
        self._strategy_type = strategy_type
        self._history: list[PromotionEvent] = []
        self._previous_params: dict[str, dict] = {}
        self._run_count: int = 0

        # Rebuild in-memory cache from DB on construction if a session factory
        # is provided.  This survives Railway redeploys.
        if db_session_factory is not None:
            self._load_history_from_db()

    def _load_history_from_db(self) -> None:
        """Rebuild _history and _previous_params from persistent DB rows."""
        try:
            from src.api.models import PromotionEventRecord
            db = self._db_session_factory()
            try:
                q = db.query(PromotionEventRecord)
                if self._strategy_type:
                    q = q.filter(PromotionEventRecord.strategy_type == self._strategy_type)
                rows = q.order_by(PromotionEventRecord.timestamp.asc()).all()
                for row in rows:
                    event = _record_to_event(row)
                    self._history.append(event)
                    # Rebuild rollback params: last before_params per strategy_type
                    # for rows that actually promoted (same logic as live path).
                    if event.promoted:
                        self._previous_params[event.strategy_type] = event.before_params
                self._run_count = len(rows)
                logger.info(
                    "RBIPipeline: loaded %d events from DB (promoted: %d)",
                    len(rows),
                    sum(1 for e in self._history if e.promoted),
                )
            finally:
                db.close()
        except Exception as exc:
            logger.warning("RBIPipeline: could not load history from DB: %s", exc)

    def _persist_event(self, event: PromotionEvent) -> None:
        """Write a PromotionEvent to the DB.  No-op if no session factory."""
        if self._db_session_factory is None:
            return
        try:
            from src.api.models import PromotionEventRecord
            db = self._db_session_factory()
            try:
                record = PromotionEventRecord(
                    strategy_type=event.strategy_type,
                    strategy_id=event.strategy_id,
                    timestamp=event.timestamp,
                    promoted=event.promoted,
                    reason=event.reason,
                    before_params=event.before_params,
                    after_params=event.after_params,
                    before_metrics=event.before_metrics,
                    after_metrics=event.after_metrics,
                )
                db.add(record)
                db.commit()
            finally:
                db.close()
        except Exception as exc:
            logger.error("RBIPipeline: failed to persist event to DB: %s", exc)

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
            event = PromotionEvent(
                strategy_type=strategy_type, strategy_id=strategy_id, timestamp=ts,
                promoted=False, reason="active_position",
                before_params=current.get("params", {}), after_params={},
                before_metrics={}, after_metrics={},
            )
            self._history.append(event)
            self._run_count += 1
            self._persist_event(event)
            return event

        current_params = current.get("params", {})
        candidates = await self._optimizer.optimize(
            strategy_type=strategy_type, symbol=symbol,
            timeframe=timeframe, lookback_days=lookback_days, n_trials=n_trials,
        )

        passing = [c for c in candidates if self._optimizer._passes_promotion_gate(c)]
        if not passing:
            event = PromotionEvent(
                strategy_type=strategy_type, strategy_id=strategy_id, timestamp=ts,
                promoted=False, reason="no_passing_candidates",
                before_params=current_params, after_params={},
                before_metrics={}, after_metrics={},
            )
            self._history.append(event)
            self._run_count += 1
            self._persist_event(event)
            return event

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
        self._run_count += 1
        self._persist_event(event)
        logger.info(
            "RBI promoted %s (id=%d): OOS sharpe=%.2f PF=%.2f WR=%.1f%%",
            strategy_type, strategy_id, best.out_sample_sharpe,
            best.out_sample_profit_factor, best.out_sample_win_rate,
        )
        return event

    @property
    def run_count(self) -> int:
        return self._run_count

    def get_rollback_params(self, strategy_type: str) -> Optional[dict]:
        return self._previous_params.get(strategy_type)

    def get_history(self) -> list[PromotionEvent]:
        return list(self._history)
