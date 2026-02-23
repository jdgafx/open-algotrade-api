"""
HyperliquidVaultExecutor — Bridges BaseStrategy signals to Hyperliquid vault execution.

Translates Signal objects from strategies into actual HyperliquidClient API calls.
Handles leverage, sizing, vault routing, PnL tracking, and position lifecycle.
"""

import asyncio
import logging
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


@dataclass
class ExecutionResult:
    success: bool
    order_result: Optional[OrderResult] = None
    realized_pnl: float = 0.0
    error: Optional[str] = None
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
    ):
        self.client = client
        self.vault_address = vault_address or client.vault_address
        self.default_slippage = default_slippage
        self.max_position_usd = max_position_usd

        self._active_positions: Dict[str, Dict] = {}
        self._execution_history: List[ExecutionResult] = []

        logger.info(
            "HyperliquidVaultExecutor initialized | vault=%s | max_pos=$%.0f",
            self.vault_address or "master",
            self.max_position_usd,
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
                success=order_result.success, order_result=order_result
            )

            if order_result.success:
                self._active_positions[f"{config.name}:{symbol}"] = {
                    "strategy": config.name,
                    "symbol": symbol,
                    "side": "long" if is_buy else "short",
                    "size": asset_size,
                    "entry_time": datetime.now(timezone.utc).isoformat(),
                    "signal_reason": signal.reason,
                }
                logger.info(
                    "Entry executed | %s | %s %s %.6f ($%.0f) | oid=%s",
                    config.name,
                    "LONG" if is_buy else "SHORT",
                    symbol,
                    asset_size,
                    size_usd,
                    order_result.oid,
                )
            else:
                result.error = order_result.error
                logger.warning(
                    "Entry failed | %s | %s %s | error=%s",
                    config.name,
                    symbol,
                    signal.signal_type.value,
                    order_result.error,
                )

            self._execution_history.append(result)
            return result

        except Exception as e:
            logger.error("Entry exception | %s | %s | %s", config.name, symbol, e)
            result = ExecutionResult(success=False, error=str(e))
            self._execution_history.append(result)
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

    async def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
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

    def get_active_positions(self) -> Dict[str, Dict]:
        return dict(self._active_positions)

    def get_execution_stats(self) -> Dict[str, Any]:
        if not self._execution_history:
            return {"total_executions": 0, "success_rate": 0, "total_pnl": 0}

        successes = sum(1 for r in self._execution_history if r.success)
        total_pnl = sum(r.realized_pnl for r in self._execution_history)

        return {
            "total_executions": len(self._execution_history),
            "successful": successes,
            "failed": len(self._execution_history) - successes,
            "success_rate": round(successes / len(self._execution_history) * 100, 1),
            "total_realized_pnl": round(total_pnl, 2),
            "active_positions": len(self._active_positions),
            "vault_address": self.vault_address,
        }
