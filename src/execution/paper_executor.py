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
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
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
        commission_pct: float = 0.0002,  # HL maker fee: 0.02% expressed as decimal fraction
    ):
        self.base_url = base_url
        self.default_slippage = default_slippage
        self.max_position_usd = max_position_usd
        self.commission_pct = commission_pct  # already a decimal fraction (0.0002 = 0.02%)

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

    # ── Persistence ──

    def _state_path(self) -> Path:
        data_dir = Path(os.getenv("DATA_DIR", "/data"))
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir / "paper_state.json"

    def save_state(self) -> None:
        """Persist paper trading state to disk so it survives redeploys."""
        try:
            state = {
                "balance": self.balance,
                "initial_balance": self.initial_balance,
                "peak_balance": self.peak_balance,
                "trade_counter": self._trade_counter,
                "positions": {
                    k: {
                        "symbol": p.symbol, "side": p.side, "size": p.size,
                        "entry_price": p.entry_price,
                        "entry_time": p.entry_time.isoformat(),
                        "leverage": p.leverage, "size_usd": p.size_usd,
                        "strategy_name": p.strategy_name,
                        "unrealized_pnl": p.unrealized_pnl,
                    }
                    for k, p in self._positions.items()
                },
                "trades": [
                    {
                        "id": t.id, "symbol": t.symbol, "side": t.side,
                        "action": t.action, "price": t.price, "size": t.size,
                        "size_usd": t.size_usd, "pnl": t.pnl,
                        "pnl_pct": t.pnl_pct, "reason": t.reason,
                        "strategy_name": t.strategy_name,
                        "timestamp": t.timestamp.isoformat(),
                    }
                    for t in self._trades[-500:]  # keep last 500 trades
                ],
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            self._state_path().write_text(json.dumps(state, indent=2))
            logger.info("[PAPER] State saved | %d trades | balance=$%.2f", len(self._trades), self.balance)
        except Exception as e:
            logger.warning("[PAPER] Failed to save state: %s", e)

    def load_state(self) -> bool:
        """Restore paper trading state from disk. Returns True if state was loaded."""
        path = self._state_path()
        if not path.exists():
            logger.info("[PAPER] No saved state found at %s", path)
            return False
        try:
            state = json.loads(path.read_text())
            self.balance = state.get("balance", self.initial_balance)
            self.peak_balance = state.get("peak_balance", self.balance)
            self._trade_counter = state.get("trade_counter", 0)

            self._positions.clear()
            for k, p in state.get("positions", {}).items():
                self._positions[k] = PaperPosition(
                    symbol=p["symbol"], side=p["side"], size=p["size"],
                    entry_price=p["entry_price"],
                    entry_time=datetime.fromisoformat(p["entry_time"]),
                    leverage=p.get("leverage", 1),
                    size_usd=p.get("size_usd", 0),
                    strategy_name=p.get("strategy_name", ""),
                    unrealized_pnl=p.get("unrealized_pnl", 0),
                )

            self._trades.clear()
            for t in state.get("trades", []):
                self._trades.append(PaperTrade(
                    id=t["id"], symbol=t["symbol"], side=t["side"],
                    action=t["action"], price=t["price"], size=t["size"],
                    size_usd=t["size_usd"], pnl=t.get("pnl", 0),
                    pnl_pct=t.get("pnl_pct", 0), reason=t.get("reason", ""),
                    strategy_name=t.get("strategy_name", ""),
                    timestamp=datetime.fromisoformat(t["timestamp"]),
                ))

            logger.info(
                "[PAPER] State restored | %d trades | %d positions | balance=$%.2f (saved %s)",
                len(self._trades), len(self._positions), self.balance,
                state.get("saved_at", "unknown"),
            )
            return True
        except Exception as e:
            logger.warning("[PAPER] Failed to load state: %s", e)
            return False

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
            # Compound sizing: grow proportionally on profit, cap at 3×; never shrink below base
            if self.initial_balance > 0:
                compound_mult = min(max(self.balance / self.initial_balance, 1.0), 3.0)
                size_usd = size_usd * compound_mult
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

            # Record position — keyed by strategy:symbol for isolation
            pos_key = f"{config.name}:{symbol}"
            if pos_key in self._positions:
                return ExecutionResult(
                    success=False,
                    error=f"Already in position for {config.name}:{symbol}",
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
            self.save_state()
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
            pos_key = f"{config.name}:{symbol}"
            pos = self._positions.get(pos_key)
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
            del self._positions[pos_key]

            # Fire-and-forget: persist trade outcome to Supermemory for self-tuning
            try:
                import asyncio as _aio
                from src.services.trade_memory import store_trade as _store
                _aio.create_task(_store(
                    strategy=config.name,
                    symbol=symbol,
                    signal=pos.side.upper(),
                    pnl=pnl_on_margin,
                    regime="unknown",
                    params=config.params,
                    price=fill_price,
                ))
            except Exception:
                pass

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

            try:
                self.save_state()
            except Exception as e:
                logger.warning(f"Failed to save state after exit: {e}")

            return result

        except Exception as e:
            logger.error("[PAPER] Exit error | %s | %s | %s", config.name, symbol, e)
            result = ExecutionResult(success=False, error=str(e))
            self._execution_history.append(result)
            return result

    async def get_position(self, symbol: str, strategy_name: str = "") -> Optional[Dict[str, Any]]:
        """Get current paper position for a strategy:symbol pair."""
        pos_key = f"{strategy_name}:{symbol}" if strategy_name else symbol
        pos = self._positions.get(pos_key)
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
        for pos_key, pos in self._positions.items():
            try:
                mid = await asyncio.to_thread(self._fetch_mid_price, pos.symbol)
                abs_size = abs(pos.size)
                if pos.side == "long":
                    unrealized_pnl = (mid - pos.entry_price) * abs_size
                else:
                    unrealized_pnl = (pos.entry_price - mid) * abs_size
                pnl_pct = (unrealized_pnl / (pos.entry_price * abs_size)) * 100 if abs_size > 0 else 0
                positions.append({
                    "symbol": pos.symbol,
                    "strategy_name": pos.strategy_name,
                    "size_usd": pos.size_usd,
                    "size": pos.size,
                    "entry_px": pos.entry_price,
                    "pnl_perc": pnl_pct,
                    "unrealized_pnl": unrealized_pnl,
                    "is_long": pos.side == "long",
                    "side": pos.side,
                })
            except Exception:
                positions.append({
                    "symbol": pos.symbol,
                    "strategy_name": pos.strategy_name,
                    "size_usd": pos.size_usd,
                    "size": pos.size,
                    "entry_px": pos.entry_price,
                    "pnl_perc": 0,
                    "unrealized_pnl": 0,
                    "is_long": pos.side == "long",
                    "side": pos.side,
                })
        return positions

    async def get_account_value(self) -> float:
        """Get total paper account value (balance + unrealized PnL)."""
        total_unrealized = 0
        for pos_key, pos in self._positions.items():
            try:
                mid = await asyncio.to_thread(self._fetch_mid_price, pos.symbol)
                abs_size = abs(pos.size)
                if pos.side == "long":
                    total_unrealized += (mid - pos.entry_price) * abs_size
                else:
                    total_unrealized += (pos.entry_price - mid) * abs_size
            except Exception:
                pass
        return self.balance + total_unrealized

    async def close_by_symbol(self, symbol: str) -> List[ExecutionResult]:
        """Close all paper positions for a given symbol — called by risk controller."""
        results = []
        keys_to_close = [k for k, p in self._positions.items() if p.symbol == symbol]
        for pos_key in keys_to_close:
            pos = self._positions[pos_key]
            is_buy = pos.size < 0
            try:
                fill_price = await asyncio.to_thread(self._get_fill_price, pos.symbol, is_buy)
                abs_size = abs(pos.size)
                if pos.side == "long":
                    pnl = (fill_price - pos.entry_price) * abs_size
                else:
                    pnl = (pos.entry_price - fill_price) * abs_size
                pnl_on_margin = pnl * pos.leverage if pos.leverage > 1 else pnl
                commission = abs_size * fill_price * self.commission_pct
                self.balance += pnl_on_margin - commission
                if self.balance > self.peak_balance:
                    self.peak_balance = self.balance
                pnl_pct = (pnl / (pos.entry_price * abs_size)) * 100 if abs_size > 0 else 0
                self._trade_counter += 1
                trade = PaperTrade(
                    id=self._trade_counter, symbol=symbol, side=pos.side,
                    action="exit", price=fill_price, size=abs_size,
                    size_usd=abs_size * fill_price, pnl=pnl_on_margin,
                    pnl_pct=pnl_pct, reason="risk_stop",
                    strategy_name=pos.strategy_name,
                )
                self._trades.append(trade)
                del self._positions[pos_key]
                results.append(ExecutionResult(success=True, realized_pnl=pnl_on_margin))
                logger.info("[PAPER] Risk close | %s | pnl=$%.2f (%.2f%%)", symbol, pnl_on_margin, pnl_pct)
            except Exception as e:
                logger.error("[PAPER] close_by_symbol error | %s | %s", symbol, e)
                results.append(ExecutionResult(success=False, error=str(e)))
        self.save_state()
        return results

    async def close_by_strategy(self, strategy_name: str) -> List[ExecutionResult]:
        """Close every open paper position whose strategy_name matches.

        Used when a strategy is being disabled — without this, the orchestrator
        stops calling should_exit on the disabled strategy and its positions
        drift indefinitely. Mirrors close_by_symbol; emits exit trades with
        reason='strategy_disabled' so they're attributable in the trade log.
        """
        results: List[ExecutionResult] = []
        keys_to_close = [
            k for k, p in self._positions.items() if p.strategy_name == strategy_name
        ]
        for pos_key in keys_to_close:
            pos = self._positions[pos_key]
            is_buy = pos.size < 0
            try:
                fill_price = await asyncio.to_thread(self._get_fill_price, pos.symbol, is_buy)
                abs_size = abs(pos.size)
                if pos.side == "long":
                    pnl = (fill_price - pos.entry_price) * abs_size
                else:
                    pnl = (pos.entry_price - fill_price) * abs_size
                pnl_on_margin = pnl * pos.leverage if pos.leverage > 1 else pnl
                commission = abs_size * fill_price * self.commission_pct
                self.balance += pnl_on_margin - commission
                if self.balance > self.peak_balance:
                    self.peak_balance = self.balance
                pnl_pct = (pnl / (pos.entry_price * abs_size)) * 100 if abs_size > 0 else 0
                self._trade_counter += 1
                trade = PaperTrade(
                    id=self._trade_counter, symbol=pos.symbol, side=pos.side,
                    action="exit", price=fill_price, size=abs_size,
                    size_usd=abs_size * fill_price, pnl=pnl_on_margin,
                    pnl_pct=pnl_pct, reason="strategy_disabled",
                    strategy_name=pos.strategy_name,
                )
                self._trades.append(trade)
                del self._positions[pos_key]
                results.append(ExecutionResult(success=True, realized_pnl=pnl_on_margin))
                logger.info(
                    "[PAPER] Orphan close | %s | %s | pnl=$%.2f (%.2f%%)",
                    strategy_name, pos.symbol, pnl_on_margin, pnl_pct,
                )
            except Exception as e:
                logger.error("[PAPER] close_by_strategy error | %s | %s | %s", strategy_name, pos.symbol, e)
                results.append(ExecutionResult(success=False, error=str(e)))
        if results:
            self.save_state()
        return results

    async def emergency_close_all(self) -> List[ExecutionResult]:
        """Close all paper positions."""
        logger.critical("[PAPER] EMERGENCY CLOSE ALL")
        results = []
        for pos_key in list(self._positions.keys()):
            pos = self._positions[pos_key]
            is_buy = pos.size < 0
            try:
                fill_price = await asyncio.to_thread(self._get_fill_price, pos.symbol, is_buy)
                abs_size = abs(pos.size)

                if pos.side == "long":
                    pnl = (fill_price - pos.entry_price) * abs_size
                else:
                    pnl = (pos.entry_price - fill_price) * abs_size

                pnl_on_margin = pnl * pos.leverage if pos.leverage > 1 else pnl
                commission = abs_size * fill_price * self.commission_pct
                pnl_pct = (pnl / (pos.entry_price * abs_size)) * 100 if abs_size > 0 else 0
                self.balance += pnl_on_margin - commission
                if self.balance > self.peak_balance:
                    self.peak_balance = self.balance

                self._trade_counter += 1
                trade = PaperTrade(
                    id=self._trade_counter, symbol=pos.symbol, side=pos.side,
                    action="exit", price=fill_price, size=abs_size,
                    size_usd=abs_size * fill_price, pnl=pnl_on_margin,
                    pnl_pct=pnl_pct, reason="emergency_close",
                    strategy_name=pos.strategy_name,
                )
                self._trades.append(trade)
                del self._positions[pos_key]

                results.append(ExecutionResult(success=True, realized_pnl=pnl_on_margin))
                logger.info("[PAPER] Emergency close | %s | pnl=$%.2f (%.2f%%)", pos.symbol, pnl_on_margin, pnl_pct)
            except Exception as e:
                results.append(ExecutionResult(success=False, error=str(e)))
        return results

    async def check_pnl_guard(
        self,
        symbol: str,
        target_pct: float = 5.0,
        max_loss_pct: float = -10.0,
        strategy_name: str = "",
    ) -> Optional[ExecutionResult]:
        """Check PnL guard and close if triggered."""
        pos_data = await self.get_position(symbol, strategy_name=strategy_name)
        if not pos_data:
            return None

        pnl_pct = pos_data.get("pnl_perc", 0)
        if pnl_pct >= target_pct or pnl_pct <= max_loss_pct:
            logger.info(
                "[PAPER] PnL guard triggered | %s | pnl=%.2f%% | target=%.1f%% | max_loss=%.1f%%",
                symbol, pnl_pct, target_pct, max_loss_pct,
            )
            pos_key = f"{strategy_name}:{symbol}" if strategy_name else symbol
            pos = self._positions.get(pos_key)
            if pos:
                is_buy = pos.size < 0
                fill_price = await asyncio.to_thread(self._get_fill_price, pos.symbol, is_buy)
                abs_size = abs(pos.size)

                if pos.side == "long":
                    pnl = (fill_price - pos.entry_price) * abs_size
                else:
                    pnl = (pos.entry_price - fill_price) * abs_size

                self.balance += pnl - (abs_size * fill_price * self.commission_pct)
                del self._positions[pos_key]

                return ExecutionResult(success=True, realized_pnl=pnl)
        return None

    def get_active_positions(self) -> Dict[str, Dict]:
        """Get active paper positions as dict keyed by symbol."""
        return {
            v.symbol: {
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
