import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import optuna
import pandas as pd

from .backtester import Backtester
from .data_cache import candle_cache
from .param_spaces import suggest_params

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
    PROMOTION_MIN_SHARPE = 1.0

    def __init__(self, initial_capital: float = 10000.0, commission_pct: float = 0.07):
        self._initial_capital = initial_capital
        self._commission_pct = commission_pct

    async def optimize(
        self,
        strategy_type: str,
        symbol: str,
        timeframe: str = "1h",
        lookback_days: int = 90,
        n_trials: int = 100,
    ) -> list[OptimizationResult]:
        data = await self._get_data(symbol, timeframe, lookback_days)
        if data is None or len(data) < 50:
            raise ValueError(
                f"Insufficient data for {symbol}/{timeframe}/{lookback_days}d: "
                f"got {len(data) if data is not None else 0} bars"
            )

        split_idx = int(len(data) * self.WALKFORWARD_SPLIT)
        in_sample = data.iloc[:split_idx].reset_index(drop=True)
        out_sample = data.iloc[split_idx:].reset_index(drop=True)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        )

        for _ in range(n_trials):
            trial = study.ask()
            try:
                params = suggest_params(trial, strategy_type)
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

        for trial in trials_sorted[:10]:
            params = trial.params
            try:
                oos = await self._run_backtest(strategy_type, symbol, timeframe, out_sample, params)
                pf_capped = min(oos.profit_factor, 5.0)
                composite = 0.7 * oos.sharpe_ratio + 0.3 * (pf_capped / 5.0)
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
                ))
            except Exception as e:
                logger.warning("OOS validation failed for %s params %s: %s", strategy_type, params, e)

        results.sort(key=lambda r: r.composite_score, reverse=True)
        return results

    def _passes_walkforward(self, oos_result) -> bool:
        return (
            oos_result.profit_factor >= self.MIN_OOS_PROFIT_FACTOR
            and oos_result.win_rate >= self.MIN_OOS_WIN_RATE
            and oos_result.total_trades >= self.MIN_OOS_TRADES
            and oos_result.max_drawdown_pct <= self.MAX_OOS_DRAWDOWN
        )

    def _passes_promotion_gate(self, result: "OptimizationResult") -> bool:
        """Strict gate a strategy must clear before it earns LIVE leverage.

        Distinct from _passes_walkforward (permissive research screen). Promotion
        requires a statistically meaningful sample so Kelly sizing is not estimated
        off noise. Operates on an OptimizationResult (out_sample_* fields).
        """
        return (
            result.out_sample_total_trades >= self.PROMOTION_MIN_OOS_TRADES
            and result.out_sample_profit_factor >= self.PROMOTION_MIN_PROFIT_FACTOR
            and result.out_sample_win_rate >= self.PROMOTION_MIN_WIN_RATE
            and result.out_sample_max_drawdown <= self.PROMOTION_MAX_DRAWDOWN
            and result.out_sample_sharpe >= self.PROMOTION_MIN_SHARPE
        )

    async def _run_backtest(self, strategy_type, symbol, timeframe, data, params):
        bt = Backtester(initial_capital=self._initial_capital, commission_pct=self._commission_pct)
        return await bt.run(
            strategy_type=strategy_type,
            symbol=symbol,
            timeframe=timeframe,
            params=params,
            data=data,
        )

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
