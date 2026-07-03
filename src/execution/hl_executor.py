"""
HyperliquidVaultExecutor — Bridges BaseStrategy signals to Hyperliquid vault execution.

Translates Signal objects from strategies into actual HyperliquidClient API calls.
Handles leverage, sizing, vault routing, PnL tracking, and position lifecycle.

Supports limit order entry (default) with automatic market order fallback.
Limit entries pay maker fees (0.01%) vs taker fees (0.035%) — a 3.5x savings.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.lib.nice_funcs import HyperliquidClient, OrderResult
from src.strategies.base_strategy import (
    BaseStrategy,
    Signal,
    SignalType,
    StrategyConfig,
)

logger = logging.getLogger(__name__)

# Fee tracking constants
MAKER_FEE_BPS = 1.0   # 0.01%
TAKER_FEE_BPS = 3.5   # 0.035%


@dataclass
class ExecutionResult:
    success: bool
    order_result: Optional[OrderResult] = None
    realized_pnl: float = 0.0
    error: Optional[str] = None
    fill_type: str = ""  # "limit", "market", or "limit_fallback_market"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class HyperliquidVaultExecutor:
    """
    Executes strategy signals on Hyperliquid via vault/subaccount.

    One executor per vault. Multiple strategies can share an executor
    (the orchestrator manages this).
    """

    def __init__(
        self,
        client: HyperliquidClient,
        vault_address: Optional[str] = None,
        default_slippage: float = 0.01,
        max_position_usd: float = 10000.0,
        use_limit_orders: bool = True,
        limit_order_timeout: float = 30.0,
    ):
        self.client = client
        self.vault_address = vault_address or client.vault_address
        self.default_slippage = default_slippage
        self.max_position_usd = max_position_usd
        self.use_limit_orders = use_limit_orders
        self.limit_order_timeout = limit_order_timeout

        self._active_positions: Dict[str, Dict] = {}
        self._execution_history: List[ExecutionResult] = []
        self._fee_savings_usd: float = 0.0

        logger.info(
            "HyperliquidVaultExecutor initialized | vault=%s | max_pos=$%.0f | limit_orders=%s | timeout=%.0fs",
            self.vault_address or "master",
            self.max_position_usd,
            self.use_limit_orders,
            self.limit_order_timeout,
        )

    async def execute_signal(
        self, signal: Signal, strategy: BaseStrategy
    ) -> ExecutionResult:
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
        config = strategy.config
        symbol = signal.symbol or config.symbol
        is_buy = signal.signal_type == SignalType.LONG

        try:
            await asyncio.to_thread(
                self.client.update_leverage,
                symbol,
                config.leverage,
                is_cross=False,
                vault_address=self.vault_address,
            )

            size_usd = signal.size_usd or config.size_usd
            size_usd = min(size_usd, self.max_position_usd)

            asset_size = await asyncio.to_thread(
                self.client.usd_to_asset_size, symbol, size_usd
            )

            if asset_size <= 0:
                return ExecutionResult(
                    success=False, error=f"Size too small: {asset_size}"
                )

            # Try limit order first (maker fee 0.01%) with market fallback (taker 0.035%)
            if self.use_limit_orders:
                result = await self._execute_limit_entry(
                    symbol, is_buy, asset_size, size_usd, config.name
                )
            else:
                result = await self._execute_market_entry(
                    symbol, is_buy, asset_size, size_usd, config.name
                )

            if result.success:
                self._active_positions[f"{config.name}:{symbol}"] = {
                    "strategy": config.name,
                    "symbol": symbol,
                    "side": "long" if is_buy else "short",
                    "size": asset_size,
                    "entry_time": datetime.now(timezone.utc).isoformat(),
                    "signal_reason": signal.reason,
                    "fill_type": result.fill_type,
                }
                # Track fee savings when limit orders fill
                if result.fill_type == "limit":
                    saved = size_usd * (TAKER_FEE_BPS - MAKER_FEE_BPS) / 10000
                    self._fee_savings_usd += saved
                    logger.info(
                        "Entry executed (LIMIT) | %s | %s %s %.6f ($%.0f) | oid=%s | saved $%.4f fees",
                        config.name,
                        "LONG" if is_buy else "SHORT",
                        symbol,
                        asset_size,
                        size_usd,
                        result.order_result.oid if result.order_result else "?",
                        saved,
                    )
                else:
                    logger.info(
                        "Entry executed (MARKET%s) | %s | %s %s %.6f ($%.0f) | oid=%s",
                        " fallback" if result.fill_type == "limit_fallback_market" else "",
                        config.name,
                        "LONG" if is_buy else "SHORT",
                        symbol,
                        asset_size,
                        size_usd,
                        result.order_result.oid if result.order_result else "?",
                    )
            else:
                logger.warning(
                    "Entry failed | %s | %s %s | error=%s",
                    config.name,
                    symbol,
                    signal.signal_type.value,
                    result.error,
                )

            self._execution_history.append(result)
            return result

        except Exception as e:
            logger.error("Entry exception | %s | %s | %s", config.name, symbol, e)
            result = ExecutionResult(success=False, error=str(e))
            self._execution_history.append(result)
            return result

    async def _execute_limit_entry(
        self,
        symbol: str,
        is_buy: bool,
        asset_size: float,
        size_usd: float,
        strategy_name: str,
    ) -> ExecutionResult:
        """
        Attempt limit order entry with two tries before falling back to market.

        1st attempt: place at best bid (buy) or best ask (sell) for queue priority.
        2nd attempt: if unfilled, cancel and retry at mid-price for better fill chance.
        Fallback: if still unfilled, cancel and use market order.
        """
        ask, bid, _ = await asyncio.to_thread(self.client.ask_bid, symbol)
        mid = (ask + bid) / 2

        # Attempt 1: passive limit at bid (buy) or ask (sell)
        limit_px = bid if is_buy else ask

        logger.info(
            "Limit entry attempt 1 | %s | %s %.6f @ %.6f (bid=%.6f ask=%.6f)",
            strategy_name,
            "BUY" if is_buy else "SELL",
            asset_size,
            limit_px,
            bid,
            ask,
        )

        order_result = await asyncio.to_thread(
            self.client.limit_order,
            symbol,
            is_buy,
            asset_size,
            limit_px,
            False,       # reduce_only
            "Gtc",       # good-til-cancel
            self.vault_address,
        )

        # Immediately filled
        if order_result.success and order_result.status == "filled":
            return ExecutionResult(
                success=True, order_result=order_result, fill_type="limit"
            )

        # Resting — wait for fill
        if order_result.success and order_result.oid is not None:
            filled = await self._wait_for_fill(
                symbol, order_result.oid, self.limit_order_timeout / 2
            )
            if filled:
                return ExecutionResult(
                    success=True, order_result=order_result, fill_type="limit"
                )

            # Cancel unfilled order before attempt 2
            await asyncio.to_thread(
                self.client.cancel_order, symbol, order_result.oid, self.vault_address
            )
            logger.info(
                "Limit entry attempt 1 timed out | %s | cancelling oid=%s",
                strategy_name,
                order_result.oid,
            )
        elif not order_result.success:
            # Limit order itself failed (e.g. insufficient margin) — go straight to market
            logger.warning(
                "Limit entry attempt 1 failed | %s | error=%s | falling back to market",
                strategy_name,
                order_result.error,
            )
            return await self._execute_market_entry(
                symbol, is_buy, asset_size, size_usd, strategy_name,
                fill_type="limit_fallback_market",
            )

        # Attempt 2: retry at mid-price (more aggressive, still maker)
        ask2, bid2, _ = await asyncio.to_thread(self.client.ask_bid, symbol)
        mid2 = (ask2 + bid2) / 2
        limit_px2 = mid2

        logger.info(
            "Limit entry attempt 2 | %s | %s %.6f @ %.6f (mid)",
            strategy_name,
            "BUY" if is_buy else "SELL",
            asset_size,
            limit_px2,
        )

        order_result2 = await asyncio.to_thread(
            self.client.limit_order,
            symbol,
            is_buy,
            asset_size,
            limit_px2,
            False,
            "Gtc",
            self.vault_address,
        )

        if order_result2.success and order_result2.status == "filled":
            return ExecutionResult(
                success=True, order_result=order_result2, fill_type="limit"
            )

        if order_result2.success and order_result2.oid is not None:
            filled = await self._wait_for_fill(
                symbol, order_result2.oid, self.limit_order_timeout / 2
            )
            if filled:
                return ExecutionResult(
                    success=True, order_result=order_result2, fill_type="limit"
                )

            # Cancel and fall back to market
            await asyncio.to_thread(
                self.client.cancel_order, symbol, order_result2.oid, self.vault_address
            )
            logger.info(
                "Limit entry attempt 2 timed out | %s | cancelling oid=%s | falling back to market",
                strategy_name,
                order_result2.oid,
            )

        # Fallback: market order
        return await self._execute_market_entry(
            symbol, is_buy, asset_size, size_usd, strategy_name,
            fill_type="limit_fallback_market",
        )

    async def _wait_for_fill(
        self, symbol: str, oid: int, timeout: float
    ) -> bool:
        """
        Poll open orders to detect when a limit order fills.

        Returns True if the order is no longer in the open orders list
        (meaning it was filled), False if timeout is reached.
        """
        poll_interval = min(2.0, timeout / 5)
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            await asyncio.sleep(poll_interval)

            open_orders = await asyncio.to_thread(self.client.get_open_orders)
            still_open = any(
                o.get("oid") == oid and o.get("coin") == symbol
                for o in open_orders
            )

            if not still_open:
                logger.debug("Order filled | %s | oid=%s", symbol, oid)
                return True

        return False

    async def _execute_market_entry(
        self,
        symbol: str,
        is_buy: bool,
        asset_size: float,
        size_usd: float,
        strategy_name: str,
        fill_type: str = "market",
    ) -> ExecutionResult:
        """Execute a market order entry (taker fees apply)."""
        order_result = await asyncio.to_thread(
            self.client.market_order,
            symbol,
            is_buy,
            asset_size,
            self.default_slippage,
            False,
            self.vault_address,
        )

        result = ExecutionResult(
            success=order_result.success,
            order_result=order_result,
            fill_type=fill_type,
        )

        if not order_result.success:
            result.error = order_result.error

        return result

    async def _execute_exit(
        self, signal: Signal, strategy: BaseStrategy
    ) -> ExecutionResult:
        config = strategy.config
        symbol = signal.symbol or config.symbol

        try:
            position_data = await self.get_position(symbol)

            if not position_data or position_data.get("size", 0) == 0:
                return ExecutionResult(success=True, error="No position to close")

            abs_size = abs(position_data["size"])
            is_buy = position_data["size"] < 0

            order_result = await asyncio.to_thread(
                self.client.market_order,
                symbol,
                is_buy,
                abs_size,
                self.default_slippage * 2,
                True,
                self.vault_address,
            )

            realized_pnl = position_data.get("unrealized_pnl", 0.0)

            result = ExecutionResult(
                success=order_result.success,
                order_result=order_result,
                realized_pnl=realized_pnl,
            )

            if order_result.success:
                strategy.record_trade(realized_pnl)
                pos_key = f"{config.name}:{symbol}"
                self._active_positions.pop(pos_key, None)
                logger.info(
                    "Exit executed | %s | %s %.6f | pnl=$%.2f | reason=%s",
                    config.name,
                    symbol,
                    abs_size,
                    realized_pnl,
                    signal.reason,
                )
            else:
                result.error = order_result.error

            self._execution_history.append(result)
            return result

        except Exception as e:
            logger.error("Exit exception | %s | %s | %s", config.name, symbol, e)
            result = ExecutionResult(success=False, error=str(e))
            self._execution_history.append(result)
            return result

    async def get_position(self, symbol: str, strategy_name: str = "") -> Optional[Dict[str, Any]]:
        try:
            (
                positions,
                in_pos,
                size,
                pos_sym,
                entry_px,
                pnl_perc,
                is_long,
            ) = await asyncio.to_thread(self.client.get_position, symbol)

            if not in_pos:
                return None

            return {
                "symbol": symbol,
                "size": size,
                "entry_px": entry_px,
                "pnl_perc": pnl_perc,
                "unrealized_pnl": entry_px * abs(size) * (pnl_perc / 100)
                if entry_px
                else 0,
                "is_long": is_long,
                "side": "long" if is_long else "short",
            }

        except Exception as e:
            logger.error("get_position failed | %s | %s", symbol, e)
            return None

    async def get_all_positions(self) -> List[Dict[str, Any]]:
        try:
            return await asyncio.to_thread(self.client.get_all_positions)
        except Exception as e:
            logger.error("get_all_positions failed | %s", e)
            return []

    async def get_account_value(self) -> float:
        try:
            return await asyncio.to_thread(self.client.get_account_value)
        except Exception as e:
            logger.error("get_account_value failed | %s", e)
            return 0.0

    async def emergency_close_all(self) -> List[ExecutionResult]:
        logger.critical("EMERGENCY CLOSE ALL triggered")
        results = []

        positions = await self.get_all_positions()
        for pos in positions:
            symbol = pos["coin"]
            try:
                closed = await asyncio.to_thread(
                    self.client.kill_switch, symbol, 5, self.vault_address
                )
                results.append(ExecutionResult(success=closed))
                logger.info("Emergency close | %s | success=%s", symbol, closed)
            except Exception as e:
                results.append(ExecutionResult(success=False, error=str(e)))
                logger.error("Emergency close failed | %s | %s", symbol, e)

        self._active_positions.clear()
        return results

    async def check_pnl_guard(
        self,
        symbol: str,
        target_pct: float = 5.0,
        max_loss_pct: float = -10.0,
    ) -> Optional[ExecutionResult]:
        try:
            pnl_closed, in_pos, size, is_long = await asyncio.to_thread(
                self.client.pnl_close,
                symbol,
                target_pct,
                max_loss_pct,
                self.vault_address,
            )

            if pnl_closed:
                logger.info("PnL guard triggered | %s | closed position", symbol)
                return ExecutionResult(success=True)

            return None

        except Exception as e:
            logger.error("PnL guard failed | %s | %s", symbol, e)
            return None

    async def close_by_strategy(self, strategy_name: str) -> List[ExecutionResult]:
        """Close every live position attributed to strategy_name.

        Called by the orchestrator ruin-guard and hard-stop paths. Uses
        _active_positions as the authoritative list of what this executor
        opened for this strategy (keyed strategy:symbol), then issues a
        market reduce-only order for each one.

        Market orders are intentional here — this is a forced safety close,
        not an entry, so taker fees are acceptable over fill uncertainty.

        Returns the list of ExecutionResults (one per closed position).
        No-ops silently if there are no matching positions.
        """
        results: List[ExecutionResult] = []

        # Collect symbols tracked under this strategy in our local registry.
        prefix = f"{strategy_name}:"
        pos_keys = [k for k in list(self._active_positions.keys()) if k.startswith(prefix)]

        if not pos_keys:
            logger.info(
                "close_by_strategy | %s | no tracked positions — no-op", strategy_name
            )
            return results

        for pos_key in pos_keys:
            pos_info = self._active_positions.get(pos_key)
            if not pos_info:
                continue
            symbol = pos_info["symbol"]
            side = pos_info.get("side", "long")

            try:
                # Fetch live position size from the exchange — don't trust local registry
                # for the actual open size (it may be stale after a restart or partial fill).
                position_data = await self.get_position(symbol)

                if not position_data or position_data.get("size", 0) == 0:
                    logger.info(
                        "close_by_strategy | %s | %s | no live position, removing from registry",
                        strategy_name, symbol,
                    )
                    self._active_positions.pop(pos_key, None)
                    continue

                abs_size = abs(position_data["size"])
                # To close a long we sell (is_buy=False); to close a short we buy (is_buy=True)
                is_buy = position_data["size"] < 0

                logger.critical(
                    "[RUIN-GUARD/HARD-STOP] close_by_strategy | %s | closing %s %.6f (%s) | reason=force_close",
                    strategy_name, symbol, abs_size, "short" if is_buy else "long",
                )

                order_result = await asyncio.to_thread(
                    self.client.market_order,
                    symbol,
                    is_buy,
                    abs_size,
                    self.default_slippage * 2,  # wider slippage for immediate fill
                    True,                        # reduce_only
                    self.vault_address,
                )

                realized_pnl = position_data.get("unrealized_pnl", 0.0)

                result = ExecutionResult(
                    success=order_result.success,
                    order_result=order_result,
                    realized_pnl=realized_pnl,
                )

                if order_result.success:
                    self._active_positions.pop(pos_key, None)
                    logger.info(
                        "close_by_strategy | %s | %s closed | pnl=$%.2f | oid=%s",
                        strategy_name, symbol, realized_pnl,
                        order_result.oid if order_result else "?",
                    )
                else:
                    result.error = order_result.error
                    logger.error(
                        "close_by_strategy | %s | %s FAILED | error=%s",
                        strategy_name, symbol, order_result.error,
                    )

                self._execution_history.append(result)
                results.append(result)

            except Exception as e:
                logger.error(
                    "close_by_strategy exception | %s | %s | %s", strategy_name, symbol, e
                )
                results.append(ExecutionResult(success=False, error=str(e)))

        return results

    def get_active_positions(self) -> Dict[str, Dict]:
        return dict(self._active_positions)

    def get_execution_stats(self, live_strategies=None) -> Dict[str, Any]:
        # live_strategies: interface parity with PaperExecutor; no durable
        # trade ledger in live mode, so the sunk/live split is paper-only
        if not self._execution_history:
            return {"total_executions": 0, "success_rate": 0, "total_pnl": 0}

        successes = sum(1 for r in self._execution_history if r.success)
        total_pnl = sum(r.realized_pnl for r in self._execution_history)

        limit_fills = sum(
            1 for r in self._execution_history if r.fill_type == "limit"
        )
        market_fills = sum(
            1 for r in self._execution_history if r.fill_type == "market"
        )
        fallback_fills = sum(
            1 for r in self._execution_history
            if r.fill_type == "limit_fallback_market"
        )

        return {
            "total_executions": len(self._execution_history),
            "successful": successes,
            "failed": len(self._execution_history) - successes,
            "success_rate": round(successes / len(self._execution_history) * 100, 1),
            "total_realized_pnl": round(total_pnl, 2),
            "active_positions": len(self._active_positions),
            "vault_address": self.vault_address,
            "limit_fills": limit_fills,
            "market_fills": market_fills,
            "limit_fallback_fills": fallback_fills,
            "fee_savings_usd": round(self._fee_savings_usd, 4),
        }
