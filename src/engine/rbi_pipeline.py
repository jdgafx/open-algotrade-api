import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from .optimizer import OptimizationEngine

logger = logging.getLogger(__name__)

# F1 — champion-challenger. Minimum RECENT realized trades before the
# incumbent's live track record is trusted enough to protect a winner / condemn
# a loser. Below this we have no trustworthy RECENT signal and defer to the
# backtest gate (safe degradation — never freeze on missing/stale data).
# The decision is on a RECENT window (recent_pnl/recent_trades), NOT lifetime
# cumulative PnL: a decayed winner with a big stale historical buffer but
# negative recent PnL must be replaced, not frozen.
MIN_RECENT_TRADES_FOR_CHAMPION = 10


def _incumbent_live_edge(current: dict) -> dict:
    """Extract the incumbent strategy's RECENT realized live performance from the
    strategy record returned by get_strategy_fn (GET /strategies/{name}).

    Decision keys: recent_pnl / recent_trades (sum + count over the last N closed
    trades for the strategy). Lifetime total_pnl/total_trades are kept for
    observability only — they are NOT what the promotion decision keys off."""
    recent_trades = int(current.get("recent_trades", 0) or 0)
    recent_pnl = float(current.get("recent_pnl", 0.0) or 0.0)
    total_trades = int(current.get("total_trades", 0) or 0)
    total_pnl = float(current.get("total_pnl", 0.0) or 0.0)
    return {
        "recent_pnl": recent_pnl,
        "recent_trades": recent_trades,
        "live_pnl": total_pnl,        # lifetime — recorded, not decided on
        "live_trades": total_trades,  # lifetime — recorded, not decided on
    }


def _rejected_candidate_metrics(best: Any, failing_criterion: str) -> dict:
    """The best (highest-composite) rejected candidate's OOS metrics plus the
    first gate criterion it failed (U5 / T032). Persisted into a
    `no_passing_candidates` event's after_metrics so a reject is auditable:
    "the search found no real edge" (a near-miss with sensible metrics) is
    distinguishable from "the gate is structurally unpassable" (e.g. a candidate
    that clears every edge bar but can never reach the OOS-trade floor)."""
    return {
        "oos_sharpe": best.out_sample_sharpe,
        "oos_profit_factor": best.out_sample_profit_factor,
        "oos_win_rate": best.out_sample_win_rate,
        "oos_trades": best.out_sample_total_trades,
        "oos_max_drawdown": best.out_sample_max_drawdown,
        "dsr": best.out_sample_dsr,
        "cpcv_frac_positive": best.cpcv_frac_positive,
        "composite_score": best.composite_score,
        "failing_criterion": failing_criterion,
    }


def _challenger_metrics(best: Any) -> dict:
    """The backtest OOS metrics of a candidate, recorded on the event so a
    declined or accepted challenger is auditable."""
    return {
        "oos_sharpe": best.out_sample_sharpe,
        "oos_profit_factor": best.out_sample_profit_factor,
        "oos_win_rate": best.out_sample_win_rate,
        "oos_trades": best.out_sample_total_trades,
        "oos_max_drawdown": best.out_sample_max_drawdown,
        "composite_score": best.composite_score,
    }


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
        self._consecutive_no_pass: dict[str, int] = {}
        self._adapted_spaces: dict[str, dict] = {}

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


    async def _adapt_param_space(self, strategy_type: str, rejection_history: list[dict]) -> dict:
        """Call LLM to narrow Optuna search space after repeated gate failures.

        Returns a narrowed param space dict (same format as PARAM_SPACES values),
        or {} on any error (safe degradation — optimizer uses original space).
        """
        import os, json
        from .param_spaces import PARAM_SPACES

        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            return {}

        current_bounds = PARAM_SPACES.get(strategy_type, {})
        if not current_bounds:
            return {}

        # Format rejection history as readable context
        rejection_lines = []
        for i, r in enumerate(rejection_history[-3:], 1):
            reason = r.get("failing_criterion", "unknown")
            sharpe = r.get("oos_sharpe", 0)
            trades = r.get("oos_trades", 0)
            dsr = r.get("dsr", 0)
            rejection_lines.append(
                f"Run {i}: failing={reason}, oos_sharpe={sharpe:.3f}, oos_trades={trades}, dsr={dsr:.3f}"
            )

        bounds_str = json.dumps({
            k: list(v[1:]) for k, v in current_bounds.items()
        }, indent=2)

        prompt = (
            f"Strategy type: {strategy_type}\n"
            f"Last {len(rejection_lines)} optimizer runs all failed the promotion gate:\n"
            + "\n".join(rejection_lines) +
            f"\n\nCurrent Optuna search bounds:\n{bounds_str}\n\n"
            "Return JSON with narrowed bounds for the next search to focus on regions "
            "more likely to pass (higher OOS trades, better DSR). Only include params "
            "where narrowing is justified. Stay within the original bounds. "
            'Format: {"param_name": [low, high], ...}. Return only the JSON object.'
        )

        try:
            import httpx
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": "meta-llama/llama-3.3-70b-instruct:free",
                "max_tokens": 300,
                "messages": [
                    {"role": "system", "content": "You are an Optuna hyperparameter search advisor. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
            }
            async with httpx.AsyncClient(timeout=8.0) as client:
                r = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers, json=payload,
                )
                r.raise_for_status()
                content_str = r.json()["choices"][0]["message"]["content"]
                raw = json.loads(content_str)

            # Validate: must be a dict of param -> [low, high], within original bounds
            if not isinstance(raw, dict):
                return {}
            adapted = {}
            for param, bounds in raw.items():
                if param not in current_bounds:
                    continue
                orig = current_bounds[param]
                kind = orig[0]
                try:
                    lo, hi = float(bounds[0]), float(bounds[1])
                    orig_lo, orig_hi = float(orig[1]), float(orig[2])
                    if lo < orig_lo: lo = orig_lo
                    if hi > orig_hi: hi = orig_hi
                    if lo >= hi:
                        continue
                    adapted[param] = (kind, lo, hi) if kind in ("float", "int") else orig
                except (TypeError, IndexError, ValueError):
                    continue

            if adapted:
                logger.info("RBI: LLM adapted param space for %s: %d params narrowed", strategy_type, len(adapted))
            return adapted
        except Exception as e:
            logger.debug("RBI: LLM param adaptation failed for %s: %s", strategy_type, e)
            return {}
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
        has_active_position = current.get("active_positions", 0) > 0
        current_params = current.get("params", {})
        candidates = await self._optimizer.optimize(
            strategy_type=strategy_type, symbol=symbol,
            timeframe=timeframe, lookback_days=lookback_days, n_trials=n_trials,
            param_space_override=self._adapted_spaces.get(strategy_type) or None,
        )

        passing = [c for c in candidates if self._optimizer._passes_promotion_gate(c)]
        if not passing:
            # Explainable reject (U5 / T032): record the best candidate's OOS
            # metrics + the first gate criterion it failed, so a correct reject
            # (the search genuinely found no real edge) is distinguishable from a
            # structurally-unpassable gate. reason stays "no_passing_candidates"
            # for back-compat; the explanation lives in after_metrics.
            if candidates:
                best_rejected = max(candidates, key=lambda c: c.composite_score)
                failing = self._optimizer.gate_failure_reason(best_rejected) or "unknown"
                after_metrics = _rejected_candidate_metrics(best_rejected, failing)
            else:
                after_metrics = {"failing_criterion": "no_candidates"}
            # Track consecutive no_pass for LLM param adaptation (C6)
            self._consecutive_no_pass[strategy_type] = self._consecutive_no_pass.get(strategy_type, 0) + 1
            consecutive = self._consecutive_no_pass[strategy_type]
            if consecutive == 3:
                adapted = await self._adapt_param_space(strategy_type, [
                    e.after_metrics for e in self._history
                    if e.strategy_type == strategy_type and e.reason == "no_passing_candidates"
                ][-3:])
                if adapted:
                    self._adapted_spaces[strategy_type] = adapted
                # Surface the adaptation outcome: persist it on the event (visible via
                # /optimize/rbi/history, survives redeploy) AND log at WARNING so a
                # silently-failing LLM call (no key / API error → {}) is never invisible.
                after_metrics["param_space_adapted"] = len(adapted)
                logger.warning(
                    "RBI param-space adaptation triggered for %s after 3 consecutive "
                    "no-pass: %d params narrowed (next run uses tightened bounds)",
                    strategy_type, len(adapted),
                )
            elif consecutive >= 9:
                # Stop adapting — no real edge in current regime
                self._consecutive_no_pass[strategy_type] = 0
                self._adapted_spaces.pop(strategy_type, None)
            event = PromotionEvent(
                strategy_type=strategy_type, strategy_id=strategy_id, timestamp=ts,
                promoted=False, reason="no_passing_candidates",
                before_params=current_params, after_params={},
                before_metrics={}, after_metrics=after_metrics,
            )
            self._history.append(event)
            self._run_count += 1
            self._persist_event(event)
            logger.info(
                "RBI no promotion for %s (id=%d): best candidate %s",
                strategy_type, strategy_id,
                after_metrics.get("failing_criterion", "no_candidates"),
            )
            return event

        best = passing[0]
        live = _incumbent_live_edge(current)

        # F1 champion-challenger: a passing backtest candidate is not enough on
        # its own. Decide on the incumbent's RECENT realized window: protect a
        # RECENT winner from being churned on backtest hope; replace a RECENT
        # loser; otherwise (thin/absent recent data — fresh strategy or
        # post-redeploy reset) defer to the backtest gate that already passed.
        proven_record = live["recent_trades"] >= MIN_RECENT_TRADES_FOR_CHAMPION
        if proven_record and live["recent_pnl"] > 0:
            event = PromotionEvent(
                strategy_type=strategy_type, strategy_id=strategy_id, timestamp=ts,
                promoted=False, reason="retained_live_winner",
                before_params=current_params, after_params={},
                before_metrics=live, after_metrics=_challenger_metrics(best),
            )
            self._history.append(event)
            self._run_count += 1
            self._persist_event(event)
            logger.info(
                "RBI retained live winner %s (id=%d): recent_pnl=%.2f over %d recent "
                "trades; declined challenger OOS sharpe=%.2f",
                strategy_type, strategy_id, live["recent_pnl"], live["recent_trades"],
                best.out_sample_sharpe,
            )
            return event

        # Reset adaptation counter on any promotion
        self._consecutive_no_pass[strategy_type] = 0
        self._adapted_spaces.pop(strategy_type, None)
        self._previous_params[strategy_type] = current_params.copy()
        if not has_active_position:
            await self._patch_strategy_fn(strategy_id, {"params": best.params})
            reason = ("promoted_over_live_loser"
                      if proven_record and live["recent_pnl"] < 0
                      else "promotion_gate_passed")
        else:
            reason = "staged_active_position"

        event = PromotionEvent(
            strategy_type=strategy_type, strategy_id=strategy_id, timestamp=ts,
            promoted=True, reason=reason,
            before_params=current_params, after_params=best.params,
            before_metrics=live, after_metrics=_challenger_metrics(best),
        )
        self._history.append(event)
        self._run_count += 1
        self._persist_event(event)
        logger.info(
            "RBI promoted %s (id=%d) [%s]: OOS sharpe=%.2f PF=%.2f WR=%.1f%% "
            "(incumbent recent_pnl=%.2f over %d recent trades)",
            strategy_type, strategy_id, reason, best.out_sample_sharpe,
            best.out_sample_profit_factor, best.out_sample_win_rate,
            live["recent_pnl"], live["recent_trades"],
        )
        return event

    @property
    def run_count(self) -> int:
        return self._run_count

    def get_rollback_params(self, strategy_type: str) -> Optional[dict]:
        return self._previous_params.get(strategy_type)

    def get_history(self) -> list[PromotionEvent]:
        return list(self._history)


def build_rbi_job_specs(
    instances: list[Any],
    supported_types: set[str],
    default_hours_by_tf: dict[str, int] | None = None,
) -> list[dict]:
    """Build RBI scheduler job specs from live StrategyInstance rows."""
    hours_by_tf = default_hours_by_tf or {"5m": 4, "15m": 4, "1h": 8, "4h": 12, "1d": 24}
    specs: list[dict] = []
    for inst in instances:
        if inst.strategy_type not in supported_types:
            continue
        specs.append(
            {
                "strategy_type": inst.strategy_type,
                "strategy_id": inst.id,
                "symbol": inst.symbol,
                "timeframe": inst.timeframe,
                "hours": hours_by_tf.get(inst.timeframe, 8),
            }
        )
    return specs
