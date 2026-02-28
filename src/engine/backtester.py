"""
Backtesting Engine — Run strategies against historical data.

Accepts a strategy_type, symbol, timeframe, date range, and params.
Loads historical candles, runs strategy's should_enter/should_exit,
tracks simulated trades, and returns performance metrics.
"""

import asyncio
import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from src.strategies.base_strategy import Signal, SignalType, StrategyConfig, StrategyTier
from src.strategies.registry import create_strategy, get_strategy_class

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    entry_idx: int
    entry_price: float
    entry_time: str
    side: str  # "long" or "short"
    size_usd: float
    exit_idx: Optional[int] = None
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    exit_reason: Optional[str] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class BacktestResult:
    id: str
    strategy_type: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    params: Dict[str, Any]
    total_bars: int = 0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    trades: List[Dict[str, Any]] = field(default_factory=list)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = ""
    duration_ms: int = 0


class Backtester:
    """Run a strategy against historical candle data."""

    def __init__(self, initial_capital: float = 10000.0, commission_pct: float = 0.05):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct

    async def run(
        self,
        strategy_type: str,
        symbol: str,
        timeframe: str = "1h",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """
        Run a backtest.

        If `data` is provided, use it directly. Otherwise, fetch historical data.
        Data must have columns: open, high, low, close, volume (lowercase).
        """
        start_ts = datetime.now(timezone.utc)
        bt_id = str(uuid.uuid4())[:8]

        # Create strategy instance
        config = StrategyConfig(
            name=f"backtest-{strategy_type}-{bt_id}",
            symbol=symbol,
            tier=StrategyTier.D,
            timeframe=timeframe,
            params=params or {},
        )
        strategy = create_strategy(strategy_type, config)

        # Get data
        if data is None:
            data = await self._fetch_data(symbol, timeframe, start_date, end_date)

        if data is None or len(data) < 10:
            raise ValueError(f"Insufficient data: got {len(data) if data is not None else 0} bars, need at least 10")

        # Normalize column names to lowercase
        data.columns = [c.lower() for c in data.columns]
        required = {"open", "high", "low", "close"}
        if not required.issubset(set(data.columns)):
            raise ValueError(f"Data must have columns: {required}. Got: {list(data.columns)}")

        if "volume" not in data.columns:
            data["volume"] = 0

        # Run simulation
        trades: List[BacktestTrade] = []
        equity = self.initial_capital
        peak_equity = equity
        max_dd = 0.0
        equity_curve = []
        current_trade: Optional[BacktestTrade] = None
        returns: List[float] = []

        for i in range(1, len(data)):
            window = data.iloc[:i + 1].copy()
            bar = data.iloc[i]

            if current_trade is None:
                # Check for entry
                signal = await strategy.should_enter(window)
                if signal and signal.signal_type in (SignalType.LONG, SignalType.SHORT):
                    entry_price = bar["close"]
                    commission = entry_price * (self.commission_pct / 100)
                    current_trade = BacktestTrade(
                        entry_idx=i,
                        entry_price=entry_price,
                        entry_time=str(data.index[i]) if hasattr(data.index[i], 'isoformat') else str(i),
                        side="long" if signal.signal_type == SignalType.LONG else "short",
                        size_usd=config.size_usd,
                    )
                    equity -= commission * (config.size_usd / entry_price)
            else:
                # Calculate current position PnL
                current_price = bar["close"]
                if current_trade.side == "long":
                    pnl_pct = (current_price - current_trade.entry_price) / current_trade.entry_price * 100
                else:
                    pnl_pct = (current_trade.entry_price - current_price) / current_trade.entry_price * 100

                position = {
                    "size": current_trade.size_usd / current_trade.entry_price,
                    "is_long": current_trade.side == "long",
                    "entry_price": current_trade.entry_price,
                    "pnl_perc": pnl_pct,
                }

                signal = await strategy.should_exit(window, position)
                if signal and signal.signal_type != SignalType.NONE:
                    exit_price = bar["close"]
                    commission = exit_price * (self.commission_pct / 100)

                    if current_trade.side == "long":
                        trade_pnl = (exit_price - current_trade.entry_price) / current_trade.entry_price
                    else:
                        trade_pnl = (current_trade.entry_price - exit_price) / current_trade.entry_price

                    trade_pnl_usd = trade_pnl * current_trade.size_usd
                    trade_pnl_usd -= commission * (current_trade.size_usd / exit_price) * 2  # entry + exit

                    current_trade.exit_idx = i
                    current_trade.exit_price = exit_price
                    current_trade.exit_time = str(data.index[i]) if hasattr(data.index[i], 'isoformat') else str(i)
                    current_trade.exit_reason = signal.reason
                    current_trade.pnl = round(trade_pnl_usd, 4)
                    current_trade.pnl_pct = round(trade_pnl * 100, 4)

                    equity += trade_pnl_usd
                    returns.append(trade_pnl)
                    strategy.record_trade(trade_pnl_usd)

                    trades.append(current_trade)
                    current_trade = None

            # Track equity curve
            if peak_equity > 0:
                dd = (peak_equity - equity) / peak_equity * 100
                if dd > max_dd:
                    max_dd = dd
            if equity > peak_equity:
                peak_equity = equity

            if i % max(1, len(data) // 200) == 0:
                equity_curve.append({
                    "bar": i,
                    "equity": round(equity, 2),
                    "drawdown_pct": round((peak_equity - equity) / peak_equity * 100, 2) if peak_equity > 0 else 0,
                })

        # Close any open trade at last bar
        if current_trade is not None:
            last_price = data.iloc[-1]["close"]
            if current_trade.side == "long":
                trade_pnl = (last_price - current_trade.entry_price) / current_trade.entry_price
            else:
                trade_pnl = (current_trade.entry_price - last_price) / current_trade.entry_price

            trade_pnl_usd = trade_pnl * current_trade.size_usd
            current_trade.exit_idx = len(data) - 1
            current_trade.exit_price = last_price
            current_trade.exit_time = str(data.index[-1]) if hasattr(data.index[-1], 'isoformat') else str(len(data) - 1)
            current_trade.exit_reason = "backtest_end"
            current_trade.pnl = round(trade_pnl_usd, 4)
            current_trade.pnl_pct = round(trade_pnl * 100, 4)
            equity += trade_pnl_usd
            returns.append(trade_pnl)
            trades.append(current_trade)

        # Calculate metrics
        total_trades = len(trades)
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]
        win_rate = (len(winning) / total_trades * 100) if total_trades > 0 else 0
        total_pnl = sum(t.pnl for t in trades)
        total_pnl_pct = (equity - self.initial_capital) / self.initial_capital * 100

        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

        avg_trade = total_pnl / total_trades if total_trades > 0 else 0
        avg_win = gross_profit / len(winning) if winning else 0
        avg_loss = -gross_loss / len(losing) if losing else 0
        best_trade = max((t.pnl for t in trades), default=0)
        worst_trade = min((t.pnl for t in trades), default=0)

        # Sharpe ratio (annualized, assuming returns per trade)
        sharpe = 0.0
        if returns and len(returns) > 1:
            import statistics
            mean_ret = statistics.mean(returns)
            std_ret = statistics.stdev(returns)
            if std_ret > 0:
                sharpe = (mean_ret / std_ret) * math.sqrt(252)  # Annualize

        duration_ms = int((datetime.now(timezone.utc) - start_ts).total_seconds() * 1000)

        return BacktestResult(
            id=bt_id,
            strategy_type=strategy_type,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date or str(data.index[0]),
            end_date=end_date or str(data.index[-1]),
            params=params or {},
            total_bars=len(data),
            total_trades=total_trades,
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=round(win_rate, 2),
            total_pnl=round(total_pnl, 2),
            total_pnl_pct=round(total_pnl_pct, 2),
            max_drawdown=round(max_dd, 2),
            max_drawdown_pct=round(max_dd, 2),
            sharpe_ratio=round(sharpe, 3),
            profit_factor=round(profit_factor, 3) if profit_factor != float('inf') else 999.0,
            avg_trade_pnl=round(avg_trade, 2),
            best_trade=round(best_trade, 2),
            worst_trade=round(worst_trade, 2),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            trades=[
                {
                    "entry_idx": t.entry_idx,
                    "entry_price": t.entry_price,
                    "entry_time": t.entry_time,
                    "side": t.side,
                    "size_usd": t.size_usd,
                    "exit_idx": t.exit_idx,
                    "exit_price": t.exit_price,
                    "exit_time": t.exit_time,
                    "exit_reason": t.exit_reason,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                }
                for t in trades
            ],
            equity_curve=equity_curve,
            created_at=datetime.now(timezone.utc).isoformat(),
            duration_ms=duration_ms,
        )

    async def _fetch_data(
        self, symbol: str, timeframe: str, start_date: Optional[str], end_date: Optional[str]
    ) -> pd.DataFrame:
        """Fetch historical candle data via ccxt or HyperliquidClient."""
        try:
            import ccxt

            exchange = ccxt.hyperliquid({"enableRateLimit": True})
            ohlcv_symbol = f"{symbol}/USDC:USDC"

            since = None
            if start_date:
                since = int(datetime.fromisoformat(start_date.replace("Z", "+00:00")).timestamp() * 1000)

            limit = 1000
            all_candles = []
            fetch_since = since

            # Paginated fetch
            for _ in range(10):  # Max 10 pages = 10k candles
                candles = exchange.fetch_ohlcv(ohlcv_symbol, timeframe=timeframe, since=fetch_since, limit=limit)
                if not candles:
                    break
                all_candles.extend(candles)
                fetch_since = candles[-1][0] + 1
                if len(candles) < limit:
                    break

            if not all_candles:
                raise ValueError(f"No data returned for {symbol}")

            df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)

            if end_date:
                end_dt = pd.Timestamp(end_date, tz="UTC")
                df = df[df.index <= end_dt]

            return df

        except ImportError:
            logger.warning("ccxt not installed, trying HyperliquidClient")
            try:
                from src.lib.nice_funcs import HyperliquidClient
                client = HyperliquidClient()
                data = await asyncio.to_thread(client.get_ohlcv, symbol, timeframe, 1000)
                if isinstance(data, pd.DataFrame):
                    data.columns = [c.lower() for c in data.columns]
                    return data
            except Exception as e:
                logger.error("Failed to fetch data via HyperliquidClient: %s", e)

            raise ValueError(f"Could not fetch historical data for {symbol}. Install ccxt: pip install ccxt")
