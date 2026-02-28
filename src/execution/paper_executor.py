"""
PaperTradingExecutor — Simulated execution against live Hyperliquid prices.

Drop-in replacement for HyperliquidVaultExecutor. Uses the public HL Info API
(no wallet/account/deposit needed) to get real-time prices, then simulates
order fills, position tracking, and PnL locally.

Usage:
    # In main.py lifespan, swap executor based on PAPER_TRADING env var:
    if os.getenv("PAPER_TRADING", "true").lower() == "true":
        executor = PaperTradingExecutor(base_url=constants.MAINNET_API_URL)
    else:
        executor = HyperliquidVaultExecutor(client=client)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from src.strategies.base_strategy import (
    BaseStrategy,
    Signal,
    SignalType,
)

logger = logging.getLogger(__name__)


@dataclass
class PaperPosition:
    symbol: str
    side: str  # "long" or "short"
    size: float  # asset units (signed: positive=long, negative=short)
    entry_price: float
    entry_time: datetime
    leverage: int = 1
    size_usd: float = 0.0
    strategy_name: str = ""
    unrealized_pnl: float = 0.0


@dataclass
class PaperTrade:
    id: int
    symbol: str
    side: str
    action: str  # "entry" or "exit"
    price: float
    size: float
    size_usd: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    reason: str = ""
    strategy_name: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PaperOrderResult:
    """Mimics OrderResult from nice_funcs for compatibility."""
    success: bool
    oid: Optional[int] = None
    status: str = ""
    raw: Optional[Dict] = None
    error: Optional[str] = None


@dataclass
class ExecutionResult:
    """Same interface as hl_executor.ExecutionResult."""
    success: bool
    order_result: Optional[PaperOrderResult] = None
    realized_pnl: float = 0.0
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PaperTradingExecutor:
    """
    Simulated executor using live Hyperliquid prices.

    No wallet, no account, no deposit needed. Uses the public Info API
    for real-time mid prices and simulates order fills with configurable slippage.
    """

    def __init__(
        self,
        base_url: str = "https://api.hyperliquid.xyz",
        default_slippage: float = 0.0005,  # Reduced: assume maker orders (MoonDev: "maker only")
        max_position_usd: float = 5000.0,  # Reduced: cap per-position exposure
        initial_balance: float = 10000.0,
        commission_pct: float = 0.02,  # HL maker fee (was 0.035 taker - MoonDev: "no taker orders")
    ):
        self.base_url = base_url
        self.default_slippage = default_slippage
        self.max_position_usd = max_position_usd
        self.commission_pct = commission_pct / 100  # convert bps to decimal

        # Paper account state
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.peak_balance = initial_balance
        self._positions: Dict[str, PaperPosition] = {}  # symbol -> position
        self._trades: List[PaperTrade] = []
        self._trade_counter = 0
        self._execution_history: List[ExecutionResult] = []

        # Price cache (refreshed on each call)
        self._mid_prices: Dict[str, float] = {}
        self._last_price_fetch: float = 0

        # For compatibility with orchestrator
        self.vault_address = "paper-trading"

        logger.info(
            "PaperTradingExecutor initialized | balance=$%.2f | slippage=%.2f%% | commission=%.4f%%",
            initial_balance,
            default_slippage * 100,
            commission_pct,
        )

    def reset(self) -> None:
        """Reset paper trading to initial state: restore balance, clear positions and trades."""
        logger.info(
            "[PAPER] Resetting | balance=$%.2f -> $%.2f | closing %d positions | clearing %d trades",
            self.balance,
            self.initial_balance,
            len(self._positions),
            len(self._trades),
        )
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self._positions.clear()
        self._trades.clear()
        self._trade_counter = 0
        self._execution_history.clear()

    def _fetch_mid_price(self, symbol: str) -> float:
        """Fetch current mid price from HL public API."""
        now = time.time()
        # Cache prices for 2 seconds to avoid hammering the API
        if now - self._last_price_fetch > 2:
            try:
                resp = requests.post(
                    f"{self.base_url}/info",
                    headers={"Content-Type": "application/json"},
                    json={"type": "allMids"},
                    timeout=5,
                )
                resp.raise_for_status()
                self._mid_prices = {k: float(v) for k, v in resp.json().items()}
                self._last_price_fetch = now
            except Exception as e:
                logger.warning("Failed to fetch mid prices: %s", e)

        price = self._mid_prices.get(symbol)
        if price is None:
            raise ValueError(f"No price data for {symbol}")
        return price

    def _get_fill_price(self, symbol: str, is_buy: bool) -> float:
        """Get simulated fill price with slippage."""
        mid = self._fetch_mid_price(symbol)
        if is_buy:
            return mid * (1 + self.default_slippage)
        else:
            return mid * (1 - self.default_slippage)

    async def execute_signal(
        self, signal: Signal, strategy: BaseStrategy
    ) -> ExecutionResult:
        """Execute a strategy signal in paper mode."""
        if signal.signal_type == SignalType.NONE:
            return ExecutionResult(success=True)

        if signal.signal_type in (SignalType.LONG, SignalType.SHORT):
            return await self._execute_entry(signal, strategy)
        elif signal.signal_type in (
            SignalType.CLOSE_LONG,
            SignalType.CLOSE_SHORT,
            SignalType.CLOSE_ALL,
        ):
            return await self._execute_exit(signal, strategy)
        else:
            return ExecutionResult(
                success=False, error=f"Unknown signal type: {signal.signal_type}"
            )

    async def _execute_entry(
        self, signal: Signal, strategy: BaseStrategy
    ) -> ExecutionResult:
        """Simulate an entry order."""
        config = strategy.config
        symbol = signal.symbol or config.symbol
        is_buy = signal.signal_type == SignalType.LONG

        try:
            fill_price = await asyncio.to_thread(
                self._get_fill_price, symbol, is_buy
            )

            size_usd = signal.size_usd or config.size_usd
            size_usd = min(size_usd, self.max_position_usd)

            # Check balance
            margin_required = size_usd / config.leverage if config.leverage > 1 else size_usd
            if margin_required > self.balance:
                return ExecutionResult(
                    success=False,
                    error=f"Insufficient balance: need ${margin_required:.2f}, have ${self.balance:.2f}",
                )

            asset_size = size_usd / fill_price
            commission = size_usd * self.commission_pct
            signed_size = asset_size if is_buy else -asset_size

            # Record position
            pos_key = symbol
            if pos_key in self._positions:
                # Already in a position — could average in, but for simplicity skip
                return ExecutionResult(
                    success=False,
                    error=f"Already in position for {symbol}",
                )

            self._positions[pos_key] = PaperPosition(
                symbol=symbol,
                side="long" if is_buy else "short",
                size=signed_size,
                entry_price=fill_price,
                entry_time=datetime.now(timezone.utc),
                leverage=config.leverage,
                size_usd=size_usd,
                strategy_name=config.name,
            )

            self.balance -= commission
            self._trade_counter += 1

            trade = PaperTrade(
                id=self._trade_counter,
                symbol=symbol,
                side="long" if is_buy else "short",
                action="entry",
                price=fill_price,
                size=asset_size,
                size_usd=size_usd,
                reason=signal.reason,
                strategy_name=config.name,
            )
            self._trades.append(trade)

            order_result = PaperOrderResult(
                success=True,
                oid=self._trade_counter,
                status="filled",
            )
            result = ExecutionResult(success=True, order_result=order_result)
            self._execution_history.append(result)

            logger.info(
                "[PAPER] Entry | %s | %s %s %.6f @ $%.2f ($%.0f) | commission=$%.2f | balance=$%.2f",
                config.name,
                "LONG" if is_buy else "SHORT",
                symbol,
                asset_size,
                fill_price,
                size_usd,
                commission,
                self.balance,
            )
            return result

        except Exception as e:
            logger.error("[PAPER] Entry error | %s | %s | %s", config.name, symbol, e)
            result = ExecutionResult(success=False, error=str(e))
            self._execution_history.append(result)
            return result

    async def _execute_exit(
        self, signal: Signal, strategy: BaseStrategy
    ) -> ExecutionResult:
        """Simulate an exit order."""
        config = strategy.config
        symbol = signal.symbol or config.symbol

        try:
            pos = self._positions.get(symbol)
            if not pos:
                return ExecutionResult(success=True, error="No position to close")

            is_buy = pos.size < 0  # closing a short = buy
            fill_price = await asyncio.to_thread(
                self._get_fill_price, symbol, is_buy
            )

            abs_size = abs(pos.size)
            exit_usd = abs_size * fill_price
            commission = exit_usd * self.commission_pct

            # Calculate PnL
            if pos.side == "long":
                pnl = (fill_price - pos.entry_price) * abs_size
            else:
                pnl = (pos.entry_price - fill_price) * abs_size

            # Apply leverage to PnL
            pnl_on_margin = pnl * pos.leverage if pos.leverage > 1 else pnl
            pnl_pct = (pnl / (pos.entry_price * abs_size)) * 100

            self.balance += pnl_on_margin - commission

            # Track peak for drawdown
            if self.balance > self.peak_balance:
                self.peak_balance = self.balance

            # Record trade
            strategy.record_trade(pnl_on_margin)

            self._trade_counter += 1
            trade = PaperTrade(
                id=self._trade_counter,
                symbol=symbol,
                side=pos.side,
                action="exit",
                price=fill_price,
                size=abs_size,
                size_usd=exit_usd,
                pnl=pnl_on_margin,
                pnl_pct=pnl_pct,
                reason=signal.reason,
                strategy_name=config.name,
            )
            self._trades.append(trade)

            # Remove position
            del self._positions[symbol]

            order_result = PaperOrderResult(
                success=True,
                oid=self._trade_counter,
                status="filled",
            )
            result = ExecutionResult(
                success=True,
                order_result=order_result,
                realized_pnl=pnl_on_margin,
            )
            self._execution_history.append(result)

            logger.info(
                "[PAPER] Exit | %s | %s %s %.6f @ $%.2f | pnl=$%.2f (%.2f%%) | balance=$%.2f",
                config.name,
                pos.side,
                symbol,
                abs_size,
                fill_price,
                pnl_on_margin,
                pnl_pct,
                self.balance,
            )
            return result

        except Exception as e:
            logger.error("[PAPER] Exit error | %s | %s | %s", config.name, symbol, e)
            result = ExecutionResult(success=False, error=str(e))
            self._execution_history.append(result)
            return result

    async def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current paper position for a symbol."""
        pos = self._positions.get(symbol)
        if not pos:
            return None

        try:
            mid = await asyncio.to_thread(self._fetch_mid_price, symbol)
            abs_size = abs(pos.size)

            if pos.side == "long":
                unrealized_pnl = (mid - pos.entry_price) * abs_size
            else:
                unrealized_pnl = (pos.entry_price - mid) * abs_size

            pnl_pct = (unrealized_pnl / (pos.entry_price * abs_size)) * 100

            return {
                "symbol": symbol,
                "size": pos.size,
                "entry_px": pos.entry_price,
                "pnl_perc": pnl_pct,
                "unrealized_pnl": unrealized_pnl,
                "is_long": pos.side == "long",
                "side": pos.side,
            }
        except Exception:
            return {
                "symbol": symbol,
                "size": pos.size,
                "entry_px": pos.entry_price,
                "pnl_perc": 0,
                "unrealized_pnl": 0,
                "is_long": pos.side == "long",
                "side": pos.side,
            }

    async def get_all_positions(self) -> List[Dict[str, Any]]:
        """Get all open paper positions."""
        positions = []
        for symbol in self._positions:
            pos = await self.get_position(symbol)
            if pos:
                positions.append(pos)
        return positions

    async def get_account_value(self) -> float:
        """Get total paper account value (balance + unrealized PnL)."""
        total_unrealized = 0
        for symbol, pos in self._positions.items():
            try:
                mid = self._fetch_mid_price(symbol)
                abs_size = abs(pos.size)
                if pos.side == "long":
                    total_unrealized += (mid - pos.entry_price) * abs_size
                else:
                    total_unrealized += (pos.entry_price - mid) * abs_size
            except Exception:
                pass
        return self.balance + total_unrealized

    async def emergency_close_all(self) -> List[ExecutionResult]:
        """Close all paper positions."""
        logger.critical("[PAPER] EMERGENCY CLOSE ALL")
        results = []
        for symbol in list(self._positions.keys()):
            pos = self._positions[symbol]
            is_buy = pos.size < 0
            try:
                fill_price = self._get_fill_price(symbol, is_buy)
                abs_size = abs(pos.size)

                if pos.side == "long":
                    pnl = (fill_price - pos.entry_price) * abs_size
                else:
                    pnl = (pos.entry_price - fill_price) * abs_size

                self.balance += pnl - (abs_size * fill_price * self.commission_pct)
                del self._positions[symbol]

                results.append(ExecutionResult(success=True, realized_pnl=pnl))
                logger.info("[PAPER] Emergency close | %s | pnl=$%.2f", symbol, pnl)
            except Exception as e:
                results.append(ExecutionResult(success=False, error=str(e)))
        return results

    async def check_pnl_guard(
        self,
        symbol: str,
        target_pct: float = 5.0,
        max_loss_pct: float = -10.0,
    ) -> Optional[ExecutionResult]:
        """Check PnL guard and close if triggered."""
        pos_data = await self.get_position(symbol)
        if not pos_data:
            return None

        pnl_pct = pos_data.get("pnl_perc", 0)
        if pnl_pct >= target_pct or pnl_pct <= max_loss_pct:
            logger.info(
                "[PAPER] PnL guard triggered | %s | pnl=%.2f%% | target=%.1f%% | max_loss=%.1f%%",
                symbol, pnl_pct, target_pct, max_loss_pct,
            )
            # Simulate close
            pos = self._positions.get(symbol)
            if pos:
                is_buy = pos.size < 0
                fill_price = self._get_fill_price(symbol, is_buy)
                abs_size = abs(pos.size)

                if pos.side == "long":
                    pnl = (fill_price - pos.entry_price) * abs_size
                else:
                    pnl = (pos.entry_price - fill_price) * abs_size

                self.balance += pnl - (abs_size * fill_price * self.commission_pct)
                del self._positions[symbol]

                return ExecutionResult(success=True, realized_pnl=pnl)
        return None

    def get_active_positions(self) -> Dict[str, Dict]:
        """Get active paper positions as dict."""
        return {
            k: {
                "strategy": v.strategy_name,
                "symbol": v.symbol,
                "side": v.side,
                "size": v.size,
                "entry_time": v.entry_time.isoformat(),
                "entry_price": v.entry_price,
            }
            for k, v in self._positions.items()
        }

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get paper trading execution statistics."""
        if not self._execution_history:
            return {
                "mode": "paper",
                "total_executions": 0,
                "success_rate": 0,
                "total_pnl": 0,
                "balance": self.balance,
                "initial_balance": self.initial_balance,
            }

        successes = sum(1 for r in self._execution_history if r.success)
        total_pnl = sum(r.realized_pnl for r in self._execution_history)

        drawdown = 0
        if self.peak_balance > 0:
            drawdown = ((self.peak_balance - self.balance) / self.peak_balance) * 100

        return {
            "mode": "paper",
            "total_executions": len(self._execution_history),
            "successful": successes,
            "failed": len(self._execution_history) - successes,
            "success_rate": round(successes / len(self._execution_history) * 100, 1),
            "total_realized_pnl": round(total_pnl, 2),
            "active_positions": len(self._positions),
            "balance": round(self.balance, 2),
            "initial_balance": self.initial_balance,
            "total_return_pct": round(
                ((self.balance - self.initial_balance) / self.initial_balance) * 100, 2
            ),
            "peak_balance": round(self.peak_balance, 2),
            "max_drawdown_pct": round(drawdown, 2),
            "total_trades": len(self._trades),
            "vault_address": "paper-trading",
        }

    def get_trade_history(self) -> List[Dict]:
        """Get all paper trades as dicts."""
        return [
            {
                "id": t.id,
                "symbol": t.symbol,
                "side": t.side,
                "action": t.action,
                "price": t.price,
                "size": t.size,
                "size_usd": t.size_usd,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "reason": t.reason,
                "strategy": t.strategy_name,
                "timestamp": t.timestamp.isoformat(),
            }
            for t in self._trades
        ]

    def get_equity_curve(self) -> List[Dict]:
        """Build equity curve from trade history."""
        curve = [{"timestamp": self._trades[0].timestamp.isoformat() if self._trades else datetime.now(timezone.utc).isoformat(), "equity": self.initial_balance}]
        running = self.initial_balance

        for t in self._trades:
            if t.action == "exit":
                running += t.pnl
                curve.append({
                    "timestamp": t.timestamp.isoformat(),
                    "equity": round(running, 2),
                })

        return curve
