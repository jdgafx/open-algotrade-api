"""
Strategy Orchestrator — Runs multiple strategies concurrently via the executor.

Lifecycle:
1. Load strategy configs from DB (StrategyInstance table)
2. Instantiate each enabled strategy via the registry
3. Run each on its own asyncio loop interval
4. Route signals through HyperliquidVaultExecutor
5. Track PnL, errors, and health per strategy
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.execution.hl_executor import HyperliquidVaultExecutor
from src.lib.nice_funcs import HyperliquidClient
from src.strategies.base_strategy import BaseStrategy, StrategyConfig, StrategyTier
from src.strategies.registry import create_strategy, get_strategy_class

logger = logging.getLogger(__name__)


class StrategyOrchestrator:
    """
    Manages the lifecycle of multiple concurrent strategies.

    Each strategy runs on its own interval, fetching data and emitting signals.
    Signals are executed through a shared HyperliquidVaultExecutor.
    """

    def __init__(
        self,
        client: HyperliquidClient,
        executor: HyperliquidVaultExecutor,
    ):
        self.client = client
        self.executor = executor
        self._strategies: Dict[str, BaseStrategy] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._running = False

        logger.info("StrategyOrchestrator initialized")

    def add_strategy(self, name: str, strategy_type: str, config: StrategyConfig) -> BaseStrategy:
        """Add and instantiate a strategy."""
        strategy = create_strategy(strategy_type, config)
        self._strategies[name] = strategy
        logger.info("Added strategy: %s (%s) on %s", name, strategy_type, config.symbol)
        return strategy

    def remove_strategy(self, name: str):
        """Remove a strategy (stops it first if running)."""
        if name in self._tasks:
            self._tasks[name].cancel()
            del self._tasks[name]
        if name in self._strategies:
            del self._strategies[name]
            logger.info("Removed strategy: %s", name)

    async def start_strategy(self, name: str):
        """Start a single strategy's run loop."""
        if name not in self._strategies:
            raise ValueError(f"Strategy {name} not found")

        strategy = self._strategies[name]
        await strategy.on_start()

        task = asyncio.create_task(self._run_strategy_loop(name, strategy))
        self._tasks[name] = task
        logger.info("Started strategy: %s", name)

    async def stop_strategy(self, name: str):
        """Stop a single strategy."""
        if name in self._tasks:
            self._tasks[name].cancel()
            try:
                await self._tasks[name]
            except asyncio.CancelledError:
                pass
            del self._tasks[name]

        if name in self._strategies:
            await self._strategies[name].on_stop()
            logger.info("Stopped strategy: %s", name)

    async def start_all(self):
        """Start all enabled strategies."""
        self._running = True
        for name, strategy in self._strategies.items():
            if strategy.config.enabled and name not in self._tasks:
                await self.start_strategy(name)
        logger.info("Started %d strategies", len(self._tasks))

    async def stop_all(self):
        """Stop all running strategies."""
        self._running = False
        names = list(self._tasks.keys())
        for name in names:
            await self.stop_strategy(name)
        logger.info("Stopped all strategies")

    async def _run_strategy_loop(self, name: str, strategy: BaseStrategy):
        """Main loop for a single strategy."""
        config = strategy.config
        symbol = config.symbol
        interval = config.timeframe
        sleep_seconds = config.interval_seconds

        logger.info(
            "Strategy loop started: %s | %s %s | interval=%ds",
            name, symbol, interval, sleep_seconds,
        )

        while True:
            try:
                # Fetch OHLCV data
                data = await asyncio.to_thread(
                    self.client.get_ohlcv,
                    symbol,
                    interval,
                    config.lookback_days,
                )

                if data.empty:
                    logger.warning("No data for %s %s, skipping", symbol, interval)
                    await asyncio.sleep(sleep_seconds)
                    continue

                # Get current position
                position = await self.executor.get_position(symbol)

                # Run strategy iteration
                signal = await strategy.run_iteration(data, position)

                # Execute signal if present
                if signal and signal.signal_type.value != "none":
                    result = await self.executor.execute_signal(signal, strategy)
                    if result.success:
                        logger.info(
                            "Executed | %s | %s | %s",
                            name, signal.signal_type.value, signal.reason,
                        )
                    else:
                        logger.warning(
                            "Execution failed | %s | %s | error=%s",
                            name, signal.signal_type.value, result.error,
                        )

                await asyncio.sleep(sleep_seconds)

            except asyncio.CancelledError:
                logger.info("Strategy loop cancelled: %s", name)
                break
            except Exception as e:
                logger.error("Strategy loop error | %s | %s", name, e)
                await strategy.on_error(e)

                if not strategy.is_healthy:
                    logger.critical(
                        "Strategy %s unhealthy (%d consecutive errors), stopping",
                        name, strategy.state.consecutive_errors,
                    )
                    break

                await asyncio.sleep(sleep_seconds)

    def get_strategy(self, name: str) -> Optional[BaseStrategy]:
        return self._strategies.get(name)

    def get_all_stats(self) -> List[Dict]:
        return [s.get_stats() for s in self._strategies.values()]

    def get_running_count(self) -> int:
        return len(self._tasks)

    def get_total_pnl(self) -> float:
        return sum(s.state.total_pnl for s in self._strategies.values())

    def get_total_trades(self) -> int:
        return sum(s.state.total_trades for s in self._strategies.values())
