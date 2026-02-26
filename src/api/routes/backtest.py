"""
Backtesting API Routes — Run backtests, view results, compare strategies.

Uses the backtesting engine with 2x commission (slippage buffer),
multi-symbol/multi-timeframe testing, and full metrics suite.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.api.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/backtest", tags=["backtest"])


# ── Schemas ──────────────────────────────────────

class BacktestConfig(BaseModel):
    strategy_type: str
    symbol: str = "BTC"
    timeframe: str = "1h"
    lookback_days: int = Field(default=90, ge=7, le=365)
    initial_capital: float = Field(default=10000.0, ge=100)
    commission_pct: float = Field(default=0.07, description="Commission % (default 0.07 = 2x HL taker fee)")
    leverage: int = Field(default=1, ge=1, le=50)
    params: Dict[str, Any] = {}

class BacktestMetrics(BaseModel):
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    avg_drawdown_pct: float
    profit_factor: float
    expectancy: float
    win_rate: float
    total_trades: int
    exposure_time_pct: float
    avg_trade_pct: float

class EquityCurvePoint(BaseModel):
    timestamp: str
    equity: float

class BacktestTrade(BaseModel):
    entry_time: str
    exit_time: str
    side: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    pnl_usd: float

class BacktestResult(BaseModel):
    id: int
    config: BacktestConfig
    metrics: BacktestMetrics
    equity_curve: List[EquityCurvePoint]
    trades: List[BacktestTrade]
    status: str = "completed"
    error: Optional[str] = None
    created_at: Optional[str] = None

class BacktestSummary(BaseModel):
    id: int
    strategy_type: str
    symbol: str
    timeframe: str
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    status: str
    created_at: Optional[str] = None

class MultiBacktestConfig(BaseModel):
    strategy_type: str
    symbols: List[str] = ["BTC", "ETH", "SOL"]
    timeframes: List[str] = ["1h", "4h", "1d"]
    lookback_days: int = 90
    initial_capital: float = 10000.0
    params: Dict[str, Any] = {}


# ── In-memory result store (replace with DB in production) ──

_backtest_results: Dict[int, BacktestResult] = {}
_next_id = 1


# ── Endpoints ────────────────────────────────────

@router.post("/run", response_model=BacktestResult)
async def run_backtest(config: BacktestConfig, background_tasks: BackgroundTasks):
    """
    Run a backtest for a strategy with given configuration.

    Always uses 2x commission to account for slippage (MoonDev principle).
    Returns full metrics including Sharpe, Sortino, Calmar, profit factor, etc.
    """
    global _next_id

    from src.strategies.registry import get_strategy_class
    if not get_strategy_class(config.strategy_type):
        raise HTTPException(status_code=400, detail=f"Unknown strategy type: {config.strategy_type}")

    result_id = _next_id
    _next_id += 1

    try:
        from src.engine.backtester import Backtester
        bt = Backtester()
        raw = await bt.run(
            strategy_type=config.strategy_type,
            symbol=config.symbol,
            timeframe=config.timeframe,
            lookback_days=config.lookback_days,
            initial_capital=config.initial_capital,
            commission_pct=config.commission_pct,
            leverage=config.leverage,
            params=config.params,
        )

        result = BacktestResult(
            id=result_id,
            config=config,
            metrics=BacktestMetrics(**raw.get("metrics", {})),
            equity_curve=[EquityCurvePoint(**p) for p in raw.get("equity_curve", [])],
            trades=[BacktestTrade(**t) for t in raw.get("trades", [])],
            status="completed",
        )
    except Exception as e:
        logger.error("Backtest failed: %s", e)
        result = BacktestResult(
            id=result_id,
            config=config,
            metrics=BacktestMetrics(
                total_return_pct=0, sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
                max_drawdown_pct=0, avg_drawdown_pct=0, profit_factor=0, expectancy=0,
                win_rate=0, total_trades=0, exposure_time_pct=0, avg_trade_pct=0,
            ),
            equity_curve=[],
            trades=[],
            status="failed",
            error=str(e),
        )

    _backtest_results[result_id] = result
    return result


@router.get("/results/{result_id}", response_model=BacktestResult)
def get_backtest_result(result_id: int):
    """Get a stored backtest result."""
    result = _backtest_results.get(result_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Backtest result {result_id} not found")
    return result


@router.get("/history", response_model=List[BacktestSummary])
def get_backtest_history(limit: int = Query(50, le=200)):
    """Get list of past backtests."""
    summaries = []
    for rid, result in sorted(_backtest_results.items(), reverse=True)[:limit]:
        summaries.append(BacktestSummary(
            id=rid,
            strategy_type=result.config.strategy_type,
            symbol=result.config.symbol,
            timeframe=result.config.timeframe,
            total_return_pct=result.metrics.total_return_pct,
            sharpe_ratio=result.metrics.sharpe_ratio,
            max_drawdown_pct=result.metrics.max_drawdown_pct,
            win_rate=result.metrics.win_rate,
            total_trades=result.metrics.total_trades,
            status=result.status,
            created_at=result.created_at,
        ))
    return summaries


@router.post("/multi", response_model=List[BacktestSummary])
async def run_multi_backtest(config: MultiBacktestConfig, background_tasks: BackgroundTasks):
    """
    Run a strategy across multiple symbols and timeframes.

    MoonDev's approach: test across BTC, ETH, SOL + multiple timeframes
    to check robustness. Uses 2x commission for all.
    """
    global _next_id

    from src.strategies.registry import get_strategy_class
    if not get_strategy_class(config.strategy_type):
        raise HTTPException(status_code=400, detail=f"Unknown strategy type: {config.strategy_type}")

    results = []
    for symbol in config.symbols:
        for timeframe in config.timeframes:
            result_id = _next_id
            _next_id += 1

            try:
                from src.engine.backtester import Backtester
                bt = Backtester()
                raw = await bt.run(
                    strategy_type=config.strategy_type,
                    symbol=symbol,
                    timeframe=timeframe,
                    lookback_days=config.lookback_days,
                    initial_capital=config.initial_capital,
                    commission_pct=0.07,
                    leverage=1,
                    params=config.params,
                )

                metrics = raw.get("metrics", {})
                result = BacktestResult(
                    id=result_id,
                    config=BacktestConfig(
                        strategy_type=config.strategy_type,
                        symbol=symbol,
                        timeframe=timeframe,
                        lookback_days=config.lookback_days,
                        initial_capital=config.initial_capital,
                        params=config.params,
                    ),
                    metrics=BacktestMetrics(**metrics),
                    equity_curve=[EquityCurvePoint(**p) for p in raw.get("equity_curve", [])],
                    trades=[BacktestTrade(**t) for t in raw.get("trades", [])],
                    status="completed",
                )
                _backtest_results[result_id] = result

                results.append(BacktestSummary(
                    id=result_id,
                    strategy_type=config.strategy_type,
                    symbol=symbol,
                    timeframe=timeframe,
                    total_return_pct=metrics.get("total_return_pct", 0),
                    sharpe_ratio=metrics.get("sharpe_ratio", 0),
                    max_drawdown_pct=metrics.get("max_drawdown_pct", 0),
                    win_rate=metrics.get("win_rate", 0),
                    total_trades=metrics.get("total_trades", 0),
                    status="completed",
                ))
            except Exception as e:
                logger.error("Multi-backtest failed for %s/%s: %s", symbol, timeframe, e)
                results.append(BacktestSummary(
                    id=result_id,
                    strategy_type=config.strategy_type,
                    symbol=symbol,
                    timeframe=timeframe,
                    total_return_pct=0,
                    sharpe_ratio=0,
                    max_drawdown_pct=0,
                    win_rate=0,
                    total_trades=0,
                    status="failed",
                ))

    return results
