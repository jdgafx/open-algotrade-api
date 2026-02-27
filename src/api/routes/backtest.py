"""
Backtesting API Routes -- Run backtests, view results, compare strategies.

Uses the backtesting engine with 2x commission (slippage buffer),
multi-symbol/multi-timeframe testing, and full metrics suite.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/backtest", tags=["backtest"])


# -- Schemas --------------------------------------------------

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


# -- In-memory result store (replace with DB in production) --

_backtest_results: Dict[int, BacktestResult] = {}
_next_id = 1


# -- Helpers --------------------------------------------------

def _lookback_to_dates(lookback_days: int) -> tuple[str, str]:
    """Convert lookback_days to ISO-format start_date and end_date strings."""
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=lookback_days)
    return start_dt.isoformat(), end_dt.isoformat()


def _engine_result_to_schema(
    engine_result,
    result_id: int,
    config: BacktestConfig,
) -> BacktestResult:
    """
    Map the engine's BacktestResult dataclass to the route's BacktestResult
    Pydantic schema, computing any metrics the engine does not provide.
    """
    # Compute sortino ratio from trades (downside deviation only)
    trade_returns = [t["pnl_pct"] / 100.0 for t in engine_result.trades if t.get("pnl_pct") is not None]
    sortino = 0.0
    if len(trade_returns) > 1:
        import math
        import statistics
        mean_ret = statistics.mean(trade_returns)
        downside = [r for r in trade_returns if r < 0]
        if downside:
            downside_std = statistics.stdev(downside) if len(downside) > 1 else abs(downside[0])
            if downside_std > 0:
                sortino = (mean_ret / downside_std) * math.sqrt(252)

    # Calmar ratio = annualized return / max drawdown
    calmar = 0.0
    if engine_result.max_drawdown_pct > 0 and engine_result.total_pnl_pct != 0:
        calmar = round(engine_result.total_pnl_pct / engine_result.max_drawdown_pct, 3)

    # Expectancy = avg_win * win_rate - avg_loss * loss_rate
    win_rate_frac = engine_result.win_rate / 100.0
    loss_rate_frac = 1.0 - win_rate_frac
    expectancy = (engine_result.avg_win * win_rate_frac) + (engine_result.avg_loss * loss_rate_frac)

    # Exposure time: fraction of bars where a position was open
    exposure_time_pct = 0.0
    if engine_result.total_bars > 0 and engine_result.trades:
        bars_in_trade = sum(
            (t.get("exit_idx", 0) or 0) - (t.get("entry_idx", 0) or 0)
            for t in engine_result.trades
        )
        exposure_time_pct = round(bars_in_trade / engine_result.total_bars * 100, 2)

    # Avg trade pct
    avg_trade_pct = 0.0
    if engine_result.total_trades > 0:
        avg_trade_pct = round(
            sum(t.get("pnl_pct", 0) for t in engine_result.trades) / engine_result.total_trades,
            4,
        )

    # Avg drawdown from equity curve
    avg_drawdown_pct = 0.0
    if engine_result.equity_curve:
        dd_values = [p.get("drawdown_pct", 0) for p in engine_result.equity_curve]
        if dd_values:
            avg_drawdown_pct = round(sum(dd_values) / len(dd_values), 2)

    metrics = BacktestMetrics(
        total_return_pct=engine_result.total_pnl_pct,
        sharpe_ratio=engine_result.sharpe_ratio,
        sortino_ratio=round(sortino, 3),
        calmar_ratio=calmar,
        max_drawdown_pct=engine_result.max_drawdown_pct,
        avg_drawdown_pct=avg_drawdown_pct,
        profit_factor=engine_result.profit_factor,
        expectancy=round(expectancy, 2),
        win_rate=engine_result.win_rate,
        total_trades=engine_result.total_trades,
        exposure_time_pct=exposure_time_pct,
        avg_trade_pct=avg_trade_pct,
    )

    # Map equity curve: engine uses {"bar": int, "equity": float, "drawdown_pct": float}
    equity_curve = [
        EquityCurvePoint(
            timestamp=str(p.get("bar", 0)),
            equity=p.get("equity", 0.0),
        )
        for p in engine_result.equity_curve
    ]

    # Map trades: engine uses {"entry_idx", "entry_price", "entry_time", "side",
    #   "size_usd", "exit_idx", "exit_price", "exit_time", "exit_reason", "pnl", "pnl_pct"}
    trades = [
        BacktestTrade(
            entry_time=t.get("entry_time", ""),
            exit_time=t.get("exit_time", ""),
            side=t.get("side", ""),
            entry_price=t.get("entry_price", 0.0),
            exit_price=t.get("exit_price", 0.0),
            pnl_pct=t.get("pnl_pct", 0.0),
            pnl_usd=t.get("pnl", 0.0),
        )
        for t in engine_result.trades
    ]

    return BacktestResult(
        id=result_id,
        config=config,
        metrics=metrics,
        equity_curve=equity_curve,
        trades=trades,
        status="completed",
        created_at=engine_result.created_at,
    )


def _make_empty_result(result_id: int, config: BacktestConfig, error: str) -> BacktestResult:
    """Create a failed/empty BacktestResult for error cases."""
    return BacktestResult(
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
        error=error,
    )


# -- Endpoints ------------------------------------------------

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

        # Pass initial_capital and commission_pct to the constructor
        bt = Backtester(
            initial_capital=config.initial_capital,
            commission_pct=config.commission_pct,
        )

        # Convert lookback_days to start_date / end_date
        start_date, end_date = _lookback_to_dates(config.lookback_days)

        # Merge leverage into strategy params so the strategy can use it.
        # Also multiply effective size by leverage for the backtester's position sizing.
        params = dict(config.params)
        params["leverage"] = config.leverage

        engine_result = await bt.run(
            strategy_type=config.strategy_type,
            symbol=config.symbol,
            timeframe=config.timeframe,
            start_date=start_date,
            end_date=end_date,
            params=params,
        )

        result = _engine_result_to_schema(engine_result, result_id, config)

    except Exception as e:
        logger.error("Backtest failed: %s", e)
        result = _make_empty_result(result_id, config, str(e))

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

                # Multi-backtest uses 2x HL taker fee (0.07%) and no leverage
                bt = Backtester(
                    initial_capital=config.initial_capital,
                    commission_pct=0.07,
                )

                start_date, end_date = _lookback_to_dates(config.lookback_days)

                engine_result = await bt.run(
                    strategy_type=config.strategy_type,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    params=config.params,
                )

                # Build a BacktestConfig for this specific symbol/timeframe combo
                combo_config = BacktestConfig(
                    strategy_type=config.strategy_type,
                    symbol=symbol,
                    timeframe=timeframe,
                    lookback_days=config.lookback_days,
                    initial_capital=config.initial_capital,
                    params=config.params,
                )

                bt_result = _engine_result_to_schema(engine_result, result_id, combo_config)
                _backtest_results[result_id] = bt_result

                results.append(BacktestSummary(
                    id=result_id,
                    strategy_type=config.strategy_type,
                    symbol=symbol,
                    timeframe=timeframe,
                    total_return_pct=bt_result.metrics.total_return_pct,
                    sharpe_ratio=bt_result.metrics.sharpe_ratio,
                    max_drawdown_pct=bt_result.metrics.max_drawdown_pct,
                    win_rate=bt_result.metrics.win_rate,
                    total_trades=bt_result.metrics.total_trades,
                    status="completed",
                    created_at=bt_result.created_at,
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
