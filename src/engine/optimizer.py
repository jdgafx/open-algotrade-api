import asyncio
import logging
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import optuna
import pandas as pd

from .backtester import Backtester
from .data_cache import candle_cache
from .param_spaces import suggest_params
from .rigor import (
    DSR_MIN,
    backtest_fitness,
    cpcv_stability,
    deflated_sharpe_ratio,
    purged_combinatorial_kfold,
)

# The backtester reports an ANNUALIZED Sharpe (mean/std * sqrt(252), backtester.py).
# DSR must be fed per-period (per-trade) Sharpes consistent with the per-trade
# returns series, so trial/selected Sharpes are de-annualized by this factor.
_ANNUALIZATION = math.sqrt(252)

# T032 part B: at the documented default (lookback_days=90, WALKFORWARD_SPLIT=0.6)
# the OOS window is only ~36 calendar days. At 1h bars that's plenty of room for
# PROMOTION_MIN_OOS_TRADES=30 to be reachable by a real edge (live evidence: rsi,
# flip_flop, nadaraya_watson all clear 47-57 OOS trades at 1h/90d -- they get
# correctly rejected on profit_factor instead, not starved on sample size). But
# the SAME 36-day window is too short for a 4h/1d strategy to ever accumulate 30
# trades, REGARDLESS of edge quality -- a structural artifact of a fixed calendar
# window not scaling with bar granularity, not an honest "no edge" verdict.
# Fix: scale the EFFECTIVE lookback by cadence (relative to the 1h baseline the
# 90-day default was sized for) so slower timeframes get proportionally more
# calendar time to reach the SAME trade-count floor. The floor itself (30) never
# moves -- this only widens the window it's measured over. Capped at the
# documented API lookback ceiling (rbi_optimize.py Field le=365) so this never
# silently demands more history than the exchange is likely to have; if the
# exchange doesn't have it, `_get_data` raises -- an honest failure, not a
# loosened gate.
_TIMEFRAME_HOURS = {"5m": 5 / 60, "15m": 15 / 60, "1h": 1.0, "4h": 4.0, "1d": 24.0}
_LOOKBACK_BASELINE_TIMEFRAME = "1h"
_MAX_LOOKBACK_DAYS = 365

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class OptimizationResult:
    params: dict[str, Any]
    in_sample_sharpe: float
    out_sample_sharpe: float
    out_sample_profit_factor: float
    out_sample_win_rate: float
    out_sample_total_trades: int
    out_sample_max_drawdown: float
    composite_score: float
    passed_walkforward: bool
    # Overfitting guards (plan U2). Defaults keep older constructions valid;
    # a real optimize() run populates both. The gate treats the defaults as a
    # fail (DSR 0.0 < DSR_MIN; cpcv 0.0 not > CPCV_MIN_FRAC_POSITIVE), so a
    # candidate with no guard evidence can never be promoted.
    out_sample_dsr: float = 0.0
    cpcv_frac_positive: float = 0.0
    # T032 part B: the actual (cadence-scaled) lookback window this candidate
    # was evaluated over -- observability so a reject's after_metrics can show
    # whether the OOS-trade floor was already given a widened window before
    # failing (a structural-gate signal) vs. failed at the unscaled default.
    lookback_days_used: int = 0


class OptimizationEngine:
    WALKFORWARD_SPLIT = 0.6
    MIN_OOS_PROFIT_FACTOR = 1.2
    MIN_OOS_WIN_RATE = 35.0
    MIN_OOS_TRADES = 5
    MAX_OOS_DRAWDOWN = 15.0

    # --- Promotion-to-live gate (strict; distinct from the research screen above) ---
    PROMOTION_MIN_OOS_TRADES = 30
    PROMOTION_MIN_PROFIT_FACTOR = 1.5
    PROMOTION_MIN_WIN_RATE = 40.0      # percent
    PROMOTION_MAX_DRAWDOWN = 10.0      # percent; tighter than research MAX_OOS_DRAWDOWN=15.0
    # Lowered from 1.0 → 0.5: post-lookahead-audit OOS Sharpe is far lower than
    # the inflated 12.8 seen before the sort guard was added.  1.0 was effectively
    # blocking all real candidates; 0.5 is still a meaningful positive edge floor.
    PROMOTION_MIN_SHARPE = 0.5
    # Overfitting guards (plan U2) — ADDITIVE to the checks above, never a relaxation.
    PROMOTION_MIN_DSR = DSR_MIN              # Deflated Sharpe >= this (0.95)
    CPCV_MIN_FRAC_POSITIVE = 0.5             # > half the combinatorial-purged CV paths net-positive

    def __init__(self, initial_capital: float = 10000.0, commission_pct: float = 0.07):
        self._initial_capital = initial_capital
        self._commission_pct = commission_pct

    @staticmethod
    def _frequency_scaled_lookback(timeframe: str, lookback_days: int) -> int:
        """Scale the calendar lookback so the OOS window's calendar span grows
        with bar interval (T032 part B) -- see module comment above
        `_TIMEFRAME_HOURS`. Timeframes at or faster than the 1h baseline are
        returned unchanged (already trade-rich at the documented default)."""
        tf_hours = _TIMEFRAME_HOURS.get(timeframe, _TIMEFRAME_HOURS[_LOOKBACK_BASELINE_TIMEFRAME])
        base_hours = _TIMEFRAME_HOURS[_LOOKBACK_BASELINE_TIMEFRAME]
        if tf_hours <= base_hours:
            return lookback_days
        scaled = int(round(lookback_days * (tf_hours / base_hours)))
        return min(scaled, _MAX_LOOKBACK_DAYS)

    async def optimize(
        self,
        strategy_type: str,
        symbol: str,
        timeframe: str = "1h",
        lookback_days: int = 90,
        n_trials: int = 100,
        param_space_override: dict | None = None,
    ) -> list[OptimizationResult]:
        lookback_days = self._frequency_scaled_lookback(timeframe, lookback_days)
        data = await self._get_data(symbol, timeframe, lookback_days)
        if data is None or len(data) < 50:
            raise ValueError(
                f"Insufficient data for {symbol}/{timeframe}/{lookback_days}d: "
                f"got {len(data) if data is not None else 0} bars"
            )

        # Guard: ensure data is chronologically ordered (oldest first) before
        # the walk-forward split.  ccxt returns ascending order by default, but
        # the cache could theoretically return data in a different order if it
        # was stored after a downstream re-sort.  Sort explicitly so the split
        # invariant holds regardless of the caller.
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.sort_index(ascending=True)
            logger.debug(
                "WF split guard: DatetimeIndex range %s → %s (%d bars)",
                data.index[0], data.index[-1], len(data),
            )
        else:
            logger.warning(
                "WF split guard: data has non-DatetimeIndex (%s); cannot verify "
                "chronological order.  Assuming ascending (as fetched).",
                type(data.index).__name__,
            )

        split_idx = int(len(data) * self.WALKFORWARD_SPLIT)
        in_sample = data.iloc[:split_idx].reset_index(drop=True)
        out_sample = data.iloc[split_idx:].reset_index(drop=True)

        # Invariant: both slices must be non-empty and temporally disjoint.
        # Use ValueError, not assert — assert is silenced by python -O (PYTHONOPTIMIZE=1).
        if len(in_sample) == 0 or len(out_sample) == 0:
            raise ValueError(
                f"Walk-forward split produced empty slice: in={len(in_sample)} "
                f"out={len(out_sample)} (total={len(data)}, split={self.WALKFORWARD_SPLIT})"
            )
        if isinstance(data.index, pd.DatetimeIndex):
            # After sort_index ascending, the last IS bar must precede the first OOS bar.
            is_last = data.index[split_idx - 1]
            oos_first = data.index[split_idx]
            if is_last >= oos_first:
                raise ValueError(
                    f"Temporal overlap: IS ends at {is_last}, OOS starts at {oos_first}. "
                    "Walk-forward split is invalid — data may not be sorted ascending."
                )
            logger.debug("WF split: IS ends %s | OOS starts %s", is_last, oos_first)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        )

        for _ in range(n_trials):
            await asyncio.sleep(0)  # yield to event loop between trials so concurrent runs interleave
            trial = study.ask()
            try:
                params = suggest_params(trial, strategy_type, space_override=param_space_override)
                bt_result = await self._run_backtest(strategy_type, symbol, timeframe, in_sample, params)
                value = bt_result.sharpe_ratio if bt_result.total_trades >= 5 else float("-inf")
                study.tell(trial, value)
            except Exception as e:
                logger.debug("Trial failed for %s: %s", strategy_type, e)
                study.tell(trial, float("-inf"))

        results: list[OptimizationResult] = []
        trials_sorted = sorted(
            [t for t in study.trials if t.value is not None and t.value != float("-inf")],
            key=lambda t: t.value,
            reverse=True,
        )

        # The full distribution of tried (in-sample) Sharpes, de-annualized to
        # per-trade units, is what DSR deflates the selected edge against — the
        # more configs we searched, the higher the bar a real edge must clear.
        sr_trials = [t.value / _ANNUALIZATION for t in trials_sorted]

        for trial in trials_sorted[:10]:
            params = trial.params
            try:
                oos = await self._run_backtest(strategy_type, symbol, timeframe, out_sample, params)
                # U3 (R4): rank candidates by the NET-of-cost, risk-adjusted
                # fitness (Sortino − DD − turnover, Wilson-shrunk) — NOT a gross
                # Sharpe/PF composite that rewards high-turnover fee-bleeders that
                # lose their edge to commission. trade['pnl'] is already
                # commission-net (Backtester deducts entry+exit commission per
                # trade; the at-most-one forced-close trade is the lone gross
                # record — negligible). Full fees+slippage+funding is a bounded
                # follow-up extension (plan U3 seam map).
                net_trades = [{"pnl_net": t["pnl"]} for t in oos.trades]
                oos_span_days = max(lookback_days * (1.0 - self.WALKFORWARD_SPLIT), 1.0)
                composite = backtest_fitness(net_trades, oos_span_days)["score"]
                # DSR (cheap — no extra backtest): per-trade Sharpe of the OOS
                # trades, deflated by the tried-Sharpe distribution.
                oos_returns = [t["pnl_pct"] / 100.0 for t in oos.trades]
                sr_selected = self._per_trade_sharpe(oos_returns)
                dsr = deflated_sharpe_ratio(
                    sr_selected, sr_trials or [sr_selected], oos_returns
                )["DSR"]
                results.append(OptimizationResult(
                    params=params,
                    in_sample_sharpe=trial.value,
                    out_sample_sharpe=oos.sharpe_ratio,
                    out_sample_profit_factor=oos.profit_factor,
                    out_sample_win_rate=oos.win_rate,
                    out_sample_total_trades=oos.total_trades,
                    out_sample_max_drawdown=oos.max_drawdown_pct,
                    composite_score=composite,
                    passed_walkforward=self._passes_walkforward(oos),
                    out_sample_dsr=dsr,
                    lookback_days_used=lookback_days,
                ))
            except Exception as e:
                logger.warning("OOS validation failed for %s params %s: %s", strategy_type, params, e)

        results.sort(key=lambda r: r.composite_score, reverse=True)

        # CPCV stability is the expensive guard (~15 backtests per candidate), so
        # only candidates already clearing every cheap check INCLUDING DSR earn it
        # — usually 0-1 candidates, keeping the added cost bounded.
        for r in results:
            if self._passes_pre_cpcv_gate(r):
                r.cpcv_frac_positive = await self._cpcv_frac_positive(
                    strategy_type, symbol, timeframe, data, r.params
                )

        return results

    @staticmethod
    def _per_trade_sharpe(returns: list[float]) -> float:
        """Per-trade Sharpe (unannualized): mean/stdev of the per-trade return
        series. Matches the units DSR expects (and the meta-repo reference)."""
        if len(returns) < 2:
            return 0.0
        std = statistics.stdev(returns)
        return statistics.mean(returns) / std if std > 0 else 0.0

    async def _cpcv_frac_positive(
        self, strategy_type, symbol, timeframe, data, params
    ) -> float:
        """Fraction of combinatorial-purged-CV paths on which `params` is
        net-positive. The selected params are re-backtested on each of the C(6,2)=15
        purged+embargoed test folds; a fold's score is its mean per-trade return
        (sign is what drives stability — a per-fold promotion-trade floor would
        zero out short folds and is intentionally not applied here)."""
        n = len(data)
        scores: list[float] = []
        for _train_idx, test_idx in purged_combinatorial_kfold(n):
            if len(test_idx) < 2:
                continue
            fold = data.iloc[test_idx].reset_index(drop=True)
            try:
                bt = await self._run_backtest(strategy_type, symbol, timeframe, fold, params)
            except Exception as e:
                logger.debug("CPCV fold backtest failed for %s: %s", strategy_type, e)
                continue
            if bt.total_trades < 1 or not bt.trades:
                continue
            scores.append(statistics.mean(t["pnl_pct"] / 100.0 for t in bt.trades))
        if not scores:
            return 0.0
        return cpcv_stability(scores)["frac_positive"]

    def _passes_walkforward(self, oos_result) -> bool:
        return (
            oos_result.profit_factor >= self.MIN_OOS_PROFIT_FACTOR
            and oos_result.win_rate >= self.MIN_OOS_WIN_RATE
            and oos_result.total_trades >= self.MIN_OOS_TRADES
            and oos_result.max_drawdown_pct <= self.MAX_OOS_DRAWDOWN
        )

    def _passes_pre_cpcv_gate(self, result: "OptimizationResult") -> bool:
        """The cheap promotion checks (no extra backtest): the original 5-criterion
        screen PLUS the Deflated-Sharpe guard. Used to decide which candidates are
        worth the expensive CPCV path eval — and is the first half of the full gate.
        """
        return (
            result.out_sample_total_trades >= self.PROMOTION_MIN_OOS_TRADES
            and result.out_sample_profit_factor >= self.PROMOTION_MIN_PROFIT_FACTOR
            and result.out_sample_win_rate >= self.PROMOTION_MIN_WIN_RATE
            and result.out_sample_max_drawdown <= self.PROMOTION_MAX_DRAWDOWN
            and result.out_sample_sharpe >= self.PROMOTION_MIN_SHARPE
            and result.out_sample_dsr >= self.PROMOTION_MIN_DSR
        )

    def _passes_promotion_gate(self, result: "OptimizationResult") -> bool:
        """Strict gate a strategy must clear before it earns LIVE leverage.

        Distinct from _passes_walkforward (permissive research screen). Promotion
        requires a statistically meaningful sample so Kelly sizing is not estimated
        off noise, AND overfitting guards — Deflated Sharpe (is the edge real or the
        luckiest of N tries?) plus combinatorial-purged-CV stability (does it hold
        across data subsets?) — so a backtest-lucky config never reaches live
        capital. ADDITIVE to the original screen; never relaxes PF/WR/Sharpe/trades/DD.
        """
        return (
            self._passes_pre_cpcv_gate(result)
            and result.cpcv_frac_positive > self.CPCV_MIN_FRAC_POSITIVE
        )

    def gate_failure_reason(self, result: "OptimizationResult") -> Optional[str]:
        """The first promotion criterion `result` fails, as a human-readable
        string, or None if it would be promoted (U5 / T032).

        Order matches _passes_promotion_gate exactly — the six cheap checks
        (_passes_pre_cpcv_gate) then CPCV — so the reported criterion is the
        same one the gate tripped on. This lets an explainable reject
        distinguish a correct "the search found no real edge" from a
        structurally-unpassable gate (e.g. an OOS-trade floor no live-cadence
        strategy can ever clear). It NEVER changes the gate decision; it only
        explains it.
        """
        checks = [
            ("oos_trades", result.out_sample_total_trades, self.PROMOTION_MIN_OOS_TRADES, ">="),
            ("profit_factor", result.out_sample_profit_factor, self.PROMOTION_MIN_PROFIT_FACTOR, ">="),
            ("win_rate", result.out_sample_win_rate, self.PROMOTION_MIN_WIN_RATE, ">="),
            ("max_drawdown", result.out_sample_max_drawdown, self.PROMOTION_MAX_DRAWDOWN, "<="),
            ("oos_sharpe", result.out_sample_sharpe, self.PROMOTION_MIN_SHARPE, ">="),
            ("dsr", result.out_sample_dsr, self.PROMOTION_MIN_DSR, ">="),
            ("cpcv_frac_positive", result.cpcv_frac_positive, self.CPCV_MIN_FRAC_POSITIVE, ">"),
        ]
        for name, actual, threshold, op in checks:
            if op == ">=":
                passed = actual >= threshold
            elif op == "<=":
                passed = actual <= threshold
            else:  # ">"
                passed = actual > threshold
            if not passed:
                return f"{name}={actual:.3f} fails {op}{threshold}"
        return None

    async def _run_backtest(self, strategy_type, symbol, timeframe, data, params):
        # ponytail: bt.run is async-wrapped but pure CPU when `data` is passed in
        # (no real awaits — strategy.should_enter/exit just crunch indicators). On
        # the main loop it never yields, so the boot-time RBI burst (trials + OOS +
        # CPCV, all routed here) starves /health and /strategies for 15-20 min.
        # Run it in a worker thread: the GIL round-robins every ~5ms, keeping the
        # event loop responsive. One offload point covers all three callers.
        return await asyncio.to_thread(
            self._run_backtest_sync, strategy_type, symbol, timeframe, data, params
        )

    def _run_backtest_sync(self, strategy_type, symbol, timeframe, data, params):
        bt = Backtester(initial_capital=self._initial_capital, commission_pct=self._commission_pct)
        return asyncio.run(bt.run(
            strategy_type=strategy_type,
            symbol=symbol,
            timeframe=timeframe,
            params=params,
            data=data,
        ))

    async def _get_data(self, symbol: str, timeframe: str, lookback_days: int) -> Optional[pd.DataFrame]:
        cached = candle_cache.get(symbol, timeframe, lookback_days)
        if cached is not None:
            return cached
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=lookback_days)
        bt = Backtester(initial_capital=10000.0, commission_pct=0.07)
        data = await bt._fetch_data(symbol, timeframe, start_dt.isoformat(), end_dt.isoformat())
        if data is not None:
            candle_cache.set(symbol, timeframe, lookback_days, data)
        return data
