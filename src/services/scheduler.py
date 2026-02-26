"""
Scheduled Trading Engine — Time-based scale-in order execution.

Instead of instant market orders, supports splitting orders over time
with limit/maker-only execution. Orders are cancelled and re-placed
if price moves to ensure we always sit on the bid/ask.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ScaleMode(str, Enum):
    IMMEDIATE = "immediate"
    SCALE_1H = "scale_1h"      # Split over 1 hour
    SCALE_2H = "scale_2h"      # Split over 2 hours
    SCALE_4H = "scale_4h"      # Split over 4 hours


# Mapping: mode -> (total_seconds, num_slices)
SCALE_CONFIG = {
    ScaleMode.IMMEDIATE: (0, 1),
    ScaleMode.SCALE_1H: (3600, 6),        # 6 slices, ~10 min apart
    ScaleMode.SCALE_2H: (7200, 8),        # 8 slices, ~15 min apart
    ScaleMode.SCALE_4H: (14400, 12),      # 12 slices, ~20 min apart
}


@dataclass
class ScheduledOrder:
    """Represents a single slice of a scheduled order."""
    id: str
    parent_id: str
    symbol: str
    is_buy: bool
    size: float
    slice_index: int
    total_slices: int
    scheduled_at: float  # Unix timestamp
    status: str = "pending"  # pending, active, filled, cancelled, failed
    limit_price: Optional[float] = None
    order_id: Optional[str] = None
    fill_price: Optional[float] = None
    attempts: int = 0
    max_attempts: int = 10
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ScheduledJob:
    """A complete scheduled order job with all slices."""
    id: str
    symbol: str
    is_buy: bool
    total_size: float
    scale_mode: ScaleMode
    strategy_name: str
    leverage: int = 3
    slices: List[ScheduledOrder] = field(default_factory=list)
    status: str = "active"  # active, completed, cancelled
    filled_size: float = 0.0
    avg_fill_price: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ScheduledOrderEngine:
    """
    Manages time-based scale-in execution.

    Splits orders into equal slices spread over the configured duration.
    Each slice is placed as a limit order on the bid/ask (maker only).
    If price moves, the order is cancelled and re-placed at the new level.
    """

    def __init__(self, executor=None, price_getter=None):
        """
        Args:
            executor: Object with place_limit_order(symbol, is_buy, size, price) and
                     cancel_order(order_id) methods.
            price_getter: Async callable(symbol) -> {"bid": float, "ask": float, "mid": float}
        """
        self._executor = executor
        self._price_getter = price_getter
        self._jobs: Dict[str, ScheduledJob] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._job_counter = 0

    def _next_job_id(self) -> str:
        self._job_counter += 1
        return f"sched_{self._job_counter}_{int(time.time())}"

    def schedule_order(
        self,
        symbol: str,
        is_buy: bool,
        total_size: float,
        scale_mode: ScaleMode,
        strategy_name: str = "",
        leverage: int = 3,
    ) -> ScheduledJob:
        """
        Schedule a new order with the specified scaling mode.

        Returns the ScheduledJob with all planned slices.
        """
        job_id = self._next_job_id()
        total_seconds, num_slices = SCALE_CONFIG[scale_mode]

        if scale_mode == ScaleMode.IMMEDIATE:
            num_slices = 1
            interval = 0
        else:
            interval = total_seconds / num_slices

        slice_size = total_size / num_slices
        now = time.time()

        slices = []
        for i in range(num_slices):
            slice_id = f"{job_id}_s{i}"
            scheduled_time = now + (i * interval)
            slices.append(ScheduledOrder(
                id=slice_id,
                parent_id=job_id,
                symbol=symbol,
                is_buy=is_buy,
                size=round(slice_size, 8),
                slice_index=i,
                total_slices=num_slices,
                scheduled_at=scheduled_time,
            ))

        job = ScheduledJob(
            id=job_id,
            symbol=symbol,
            is_buy=is_buy,
            total_size=total_size,
            scale_mode=scale_mode,
            strategy_name=strategy_name,
            leverage=leverage,
            slices=slices,
        )

        self._jobs[job_id] = job
        logger.info(
            "Scheduled order | %s | %s %s %.6f | mode=%s slices=%d",
            strategy_name, "BUY" if is_buy else "SELL", symbol,
            total_size, scale_mode.value, num_slices,
        )

        return job

    async def start(self):
        """Start the scheduler loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Scheduled order engine started")

    async def stop(self):
        """Stop the scheduler loop and cancel all pending orders."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # Cancel all active orders
        for job in self._jobs.values():
            if job.status == "active":
                await self._cancel_job(job)
        logger.info("Scheduled order engine stopped")

    async def _run_loop(self):
        """Main loop: check pending slices and manage limit orders."""
        while self._running:
            try:
                now = time.time()
                for job in list(self._jobs.values()):
                    if job.status != "active":
                        continue

                    for sl in job.slices:
                        if sl.status == "pending" and sl.scheduled_at <= now:
                            await self._execute_slice(job, sl)
                        elif sl.status == "active":
                            await self._check_and_refresh(job, sl)

                    # Check if job is complete
                    all_done = all(
                        s.status in ("filled", "cancelled", "failed")
                        for s in job.slices
                    )
                    if all_done:
                        job.status = "completed"
                        logger.info(
                            "Scheduled job complete | %s | filled=%.6f avg_price=%.2f",
                            job.id, job.filled_size, job.avg_fill_price,
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Scheduler loop error: %s", e)

            await asyncio.sleep(5)  # Check every 5 seconds

    async def _execute_slice(self, job: ScheduledJob, sl: ScheduledOrder):
        """Place a limit order for a single slice."""
        if not self._executor or not self._price_getter:
            # No executor configured — simulate fill
            sl.status = "filled"
            sl.fill_price = 0
            job.filled_size += sl.size
            return

        try:
            price_info = await self._price_getter(sl.symbol)
            if not price_info:
                sl.attempts += 1
                if sl.attempts >= sl.max_attempts:
                    sl.status = "failed"
                return

            # Always sit on the bid (buy) or ask (sell) for maker-only
            if sl.is_buy:
                limit_price = price_info.get("bid", price_info.get("mid", 0))
            else:
                limit_price = price_info.get("ask", price_info.get("mid", 0))

            if limit_price <= 0:
                sl.attempts += 1
                return

            sl.limit_price = limit_price

            result = await self._executor.place_limit_order(
                symbol=sl.symbol,
                is_buy=sl.is_buy,
                size=sl.size,
                price=limit_price,
            )

            if result and result.get("success"):
                sl.order_id = result.get("order_id")
                sl.status = "active"
                logger.info(
                    "Slice placed | %s | slice %d/%d | price=%.2f size=%.6f",
                    job.id, sl.slice_index + 1, sl.total_slices,
                    limit_price, sl.size,
                )
            else:
                sl.attempts += 1
                if sl.attempts >= sl.max_attempts:
                    sl.status = "failed"
                    logger.warning("Slice failed after %d attempts | %s", sl.attempts, sl.id)

        except Exception as e:
            logger.error("Execute slice error | %s | %s", sl.id, e)
            sl.attempts += 1
            if sl.attempts >= sl.max_attempts:
                sl.status = "failed"

    async def _check_and_refresh(self, job: ScheduledJob, sl: ScheduledOrder):
        """Check if active limit order needs to be refreshed (price moved)."""
        if not self._executor or not self._price_getter:
            # Simulate: mark as filled after being "active"
            sl.status = "filled"
            job.filled_size += sl.size
            return

        try:
            price_info = await self._price_getter(sl.symbol)
            if not price_info:
                return

            if sl.is_buy:
                current_level = price_info.get("bid", 0)
            else:
                current_level = price_info.get("ask", 0)

            if current_level <= 0 or sl.limit_price is None:
                return

            # If price moved more than 0.05%, cancel and re-place
            price_diff_pct = abs(current_level - sl.limit_price) / sl.limit_price * 100
            if price_diff_pct > 0.05:
                # Cancel existing order
                if sl.order_id:
                    await self._executor.cancel_order(sl.order_id)

                # Re-place at new level
                sl.limit_price = current_level
                result = await self._executor.place_limit_order(
                    symbol=sl.symbol,
                    is_buy=sl.is_buy,
                    size=sl.size,
                    price=current_level,
                )

                if result and result.get("success"):
                    sl.order_id = result.get("order_id")
                    logger.debug(
                        "Slice refreshed | %s | new_price=%.2f",
                        sl.id, current_level,
                    )
                else:
                    sl.attempts += 1

            # Check if order was filled (in real integration, query order status)
            if sl.order_id:
                order_status = await self._executor.get_order_status(sl.order_id)
                if order_status and order_status.get("filled"):
                    sl.status = "filled"
                    sl.fill_price = order_status.get("fill_price", sl.limit_price)
                    job.filled_size += sl.size
                    # Update average fill price
                    if job.filled_size > 0:
                        job.avg_fill_price = (
                            (job.avg_fill_price * (job.filled_size - sl.size) +
                             sl.fill_price * sl.size) / job.filled_size
                        )

        except Exception as e:
            logger.error("Check/refresh error | %s | %s", sl.id, e)

    async def _cancel_job(self, job: ScheduledJob):
        """Cancel all pending/active slices in a job."""
        for sl in job.slices:
            if sl.status in ("pending", "active"):
                if sl.order_id and self._executor:
                    try:
                        await self._executor.cancel_order(sl.order_id)
                    except Exception:
                        pass
                sl.status = "cancelled"
        job.status = "cancelled"
        logger.info("Job cancelled | %s", job.id)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a scheduled job by ID."""
        job = self._jobs.get(job_id)
        if not job:
            return False
        asyncio.create_task(self._cancel_job(job))
        return True

    def get_job(self, job_id: str) -> Optional[Dict]:
        """Get job status and details."""
        job = self._jobs.get(job_id)
        if not job:
            return None
        return self._job_to_dict(job)

    def get_all_jobs(self) -> List[Dict]:
        """Get all scheduled jobs."""
        return [self._job_to_dict(j) for j in self._jobs.values()]

    def get_active_jobs(self) -> List[Dict]:
        """Get only active scheduled jobs."""
        return [self._job_to_dict(j) for j in self._jobs.values() if j.status == "active"]

    def _job_to_dict(self, job: ScheduledJob) -> Dict:
        return {
            "id": job.id,
            "symbol": job.symbol,
            "side": "buy" if job.is_buy else "sell",
            "total_size": job.total_size,
            "filled_size": job.filled_size,
            "avg_fill_price": round(job.avg_fill_price, 2),
            "scale_mode": job.scale_mode.value,
            "strategy_name": job.strategy_name,
            "status": job.status,
            "slices_total": len(job.slices),
            "slices_filled": sum(1 for s in job.slices if s.status == "filled"),
            "slices_pending": sum(1 for s in job.slices if s.status == "pending"),
            "slices_active": sum(1 for s in job.slices if s.status == "active"),
            "created_at": job.created_at.isoformat(),
        }


# Global singleton
_scheduler: Optional[ScheduledOrderEngine] = None


def get_scheduler() -> ScheduledOrderEngine:
    global _scheduler
    if _scheduler is None:
        _scheduler = ScheduledOrderEngine()
    return _scheduler
