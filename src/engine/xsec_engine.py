"""
XsecCarryEngine — cross-sectional funding carry in the live paper book.

Signal logic mirrors scripts/edge_probe.py:xsec_funding_carry
(OOS Sharpe 1.89, 53 trades, corr 0.08):
  - 1h cadence, lb=24, XSEC_Q=0.30
  - LONG bottom-30% by smoothed funding (cheapest / most-negative carry)
  - SHORT top-30% (most crowded / expensive to hold)
  - Equal-weight, dollar-neutral

Standalone class — NOT a BaseStrategy subclass.
Uses a thin _StratShim to satisfy executor.execute_signal's duck-type.
"""

import asyncio
import logging
import os
import time
from collections import deque
from typing import Dict, Optional, Set, Tuple

import requests

logger = logging.getLogger(__name__)

# ── Constants (env-overridable, no magic hardcode) ───────────────────────────
REBALANCE_INTERVAL: int = int(os.getenv("XSEC_REBALANCE_SECS", "3600"))  # 1h
XSEC_Q: float = 0.30         # bottom/top 30% each
LB: int = 24                 # smoothing window (bars @ 1h = 24h)
PER_LEG_USD: float = float(os.getenv("XSEC_PER_LEG_USD", "50"))

# Same coin universe as edge_probe.py:COINS
DEFAULT_COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "AVAX", "LINK", "SUI", "ARB"]


# ── Pure ranking logic (isolated for testing) ────────────────────────────────

def rank_basket(
    smoothed: Dict[str, float],
    q: float = XSEC_Q,
) -> Tuple[Set[str], Set[str]]:
    """Return (long_set, short_set) from a coin->smoothed_funding mapping.

    Long  = bottom-q (lowest/negative funding = cheapest carry).
    Short = top-q    (highest funding = most crowded/expensive).
    Both sets are equal size (dollar-neutral); mid-rank coins are excluded.

    Pure function — no side effects, no network. Testable.
    """
    coins = sorted(smoothed)  # deterministic order
    n = len(coins)
    if n < 4:
        return set(), set()
    ordered = sorted(coins, key=lambda c: smoothed[c])
    k = max(1, int(n * q))
    return set(ordered[:k]), set(ordered[-k:])


# ── Shim: minimal duck-type for PaperTradingExecutor.execute_signal ─────────

class _StratShim:
    """Provides .config (StrategyConfig) + .record_trade() for the executor.

    The executor's _execute_exit calls strategy.record_trade(pnl) to update
    circuit-breaker state.  xsec_carry has no circuit breaker of its own,
    so this is intentionally a no-op.
    ponytail: no circuit-breaker for market-neutral engine; add if CB needed.
    """

    def __init__(self, config):
        self.config = config

    def record_trade(self, pnl: float) -> None:
        pass


# ── Engine ───────────────────────────────────────────────────────────────────

class XsecCarryEngine:
    """Async cross-sectional funding carry engine.

    Lifecycle:
        engine = XsecCarryEngine(executor, client)
        task   = asyncio.create_task(engine.run())
        ...
        engine.stop()
        await task
    """

    def __init__(self, executor, client, coins=None):
        self._executor = executor
        self._client = client
        self._coins: list = list(coins or DEFAULT_COINS)
        # Rolling funding history (max LB+1 snapshots so rolling mean covers LB)
        self._history: deque = deque(maxlen=LB + 1)
        # Currently open legs: coin -> "long" | "short"
        self._open_legs: Dict[str, str] = {}
        self._stop_event = asyncio.Event()

    # ── Private helpers ──────────────────────────────────────────────────────

    def _fetch_all_funding(self) -> Optional[Dict[str, float]]:
        """Single metaAndAssetCtxs call → {coin: funding_rate} for our coin list.

        Returns None on any network/parse error (caller skips the tick).
        """
        try:
            url = f"{self._client.base_url}/info"
            r = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json={"type": "metaAndAssetCtxs"},
                timeout=10,
            )
            r.raise_for_status()
            result = r.json()
            if not (isinstance(result, list) and len(result) >= 2):
                logger.warning("xsec_carry: unexpected metaAndAssetCtxs shape")
                return None
            universe = result[0]["universe"]
            ctxs = result[1]
            name_to_rate: Dict[str, float] = {
                asset["name"]: float(ctxs[i].get("funding", 0.0))
                for i, asset in enumerate(universe)
                if i < len(ctxs)
            }
            snapshot = {c: name_to_rate[c] for c in self._coins if c in name_to_rate}
            return snapshot if snapshot else None
        except Exception as exc:
            logger.warning("xsec_carry: funding fetch failed — %s", exc)
            return None

    def _seed_history(self) -> None:
        """Pre-fill history from HL fundingHistory so the FIRST live tick has full
        lb24 smoothing — matches the validated edge_probe backtest exactly and
        removes the 24h deque warm-up. Builds LB hourly snapshots keyed by coin.
        Best-effort: on any failure, fall back to live tick-built history."""
        try:
            per_coin: Dict[str, list] = {}
            end = int(time.time() * 1000)
            start = end - (LB + 2) * 3600 * 1000  # hourly funding -> LB+2 bars
            for coin in self._coins:
                try:
                    r = requests.post(
                        f"{self._client.base_url}/info",
                        headers={"Content-Type": "application/json"},
                        json={"type": "fundingHistory", "coin": coin, "startTime": start},
                        timeout=10,
                    )
                    r.raise_for_status()
                    rows = r.json()
                    if isinstance(rows, list) and rows:
                        per_coin[coin] = [float(x["fundingRate"]) for x in rows[-LB:]]
                except Exception as exc:
                    logger.debug("xsec_carry: seed fetch %s failed — %s", coin, exc)
            if len(per_coin) < 4:
                logger.warning("xsec_carry: seed got <4 coins — falling back to live warm-up")
                return
            depth = min(len(v) for v in per_coin.values())
            for j in range(depth):
                # align from the most-recent backwards so the last snapshot is newest
                self._history.append({c: per_coin[c][len(per_coin[c]) - depth + j] for c in per_coin})
            logger.info("xsec_carry: seeded %d funding snapshots across %d coins",
                        depth, len(per_coin))
        except Exception as exc:
            logger.warning("xsec_carry: history seed failed (non-fatal) — %s", exc)

    def _smoothed(self) -> Optional[Dict[str, float]]:
        """Rolling mean of funding over accumulated history (up to LB bars)."""
        if not self._history:
            return None
        # Only include coins present in all snapshots (inner join)
        common = set(self._history[0]).intersection(*[set(s) for s in self._history])
        if not common:
            return None
        return {
            c: sum(s[c] for s in self._history) / len(self._history)
            for c in common
        }

    async def _open_leg(self, coin: str, side: str, funding_val: float) -> None:
        from src.strategies.base_strategy import Signal, SignalType, StrategyConfig, StrategyTier
        sig = Signal(
            signal_type=SignalType.LONG if side == "long" else SignalType.SHORT,
            symbol=coin,
            strength=1.0,
            size_usd=PER_LEG_USD,
            reason=f"xsec_carry {side} | funding={funding_val:.6f}",
        )
        cfg = StrategyConfig(
            name="xsec_carry",
            symbol=coin,
            tier=StrategyTier.E,
            size_usd=PER_LEG_USD,
        )
        result = await self._executor.execute_signal(sig, _StratShim(cfg))
        if result.success:
            self._open_legs[coin] = side
            logger.info("xsec_carry: opened %s %s | funding=%.6f", side, coin, funding_val)
        else:
            logger.debug("xsec_carry: open %s %s skipped — %s", side, coin, result.error)

    async def _close_leg(self, coin: str, side: str) -> None:
        from src.strategies.base_strategy import Signal, SignalType, StrategyConfig, StrategyTier
        sig = Signal(
            signal_type=SignalType.CLOSE_LONG if side == "long" else SignalType.CLOSE_SHORT,
            symbol=coin,
            reason=f"xsec_carry: {coin} exited basket",
        )
        cfg = StrategyConfig(
            name="xsec_carry",
            symbol=coin,
            tier=StrategyTier.E,
            size_usd=PER_LEG_USD,
        )
        result = await self._executor.execute_signal(sig, _StratShim(cfg))
        if result.success:
            self._open_legs.pop(coin, None)
            logger.info("xsec_carry: closed %s %s", side, coin)
        else:
            # Position may not exist yet (e.g. entry was rejected); still remove tracking
            self._open_legs.pop(coin, None)
            logger.debug("xsec_carry: close %s %s — %s", side, coin, result.error)

    async def _tick(self) -> None:
        """One rebalance tick: fetch funding, smooth, diff basket, execute."""
        snapshot = await asyncio.to_thread(self._fetch_all_funding)
        if snapshot is None:
            return  # error already logged; skip this tick

        self._history.append(snapshot)

        if len(self._history) < 2:
            logger.info(
                "xsec_carry: warming up (%d/%d bars)", len(self._history), LB
            )
            return

        smoothed = self._smoothed()
        if not smoothed or len(smoothed) < 4:
            logger.warning("xsec_carry: insufficient coin data (%d coins)", len(smoothed or {}))
            return

        want_long, want_short = rank_basket(smoothed)

        # 1. Close legs no longer in desired basket (act on last CLOSED bar — inherent in live)
        closes = [
            (coin, side)
            for coin, side in list(self._open_legs.items())
            if (side == "long" and coin not in want_long)
            or (side == "short" and coin not in want_short)
        ]
        for coin, side in closes:
            try:
                await self._close_leg(coin, side)
            except Exception as exc:
                logger.warning("xsec_carry: close %s %s error — %s", side, coin, exc)

        # 2. Open new basket legs (skip already-open)
        opens = (
            [(c, "long")  for c in want_long  if c not in self._open_legs] +
            [(c, "short") for c in want_short if c not in self._open_legs]
        )
        for coin, side in opens:
            try:
                await self._open_leg(coin, side, smoothed[coin])
            except Exception as exc:
                logger.warning("xsec_carry: open %s %s error — %s", side, coin, exc)

    # ── Public interface ─────────────────────────────────────────────────────

    async def run(self) -> None:
        """Main loop — run as asyncio.create_task(engine.run())."""
        logger.info(
            "XsecCarryEngine started | coins=%s | interval=%ds | lb=%d | q=%.0f%% | per_leg=$%.0f",
            self._coins, REBALANCE_INTERVAL, LB, XSEC_Q * 100, PER_LEG_USD,
        )
        # Seed lb24 history up-front so the first tick trades with full smoothing.
        await asyncio.to_thread(self._seed_history)
        try:
            while not self._stop_event.is_set():
                try:
                    await self._tick()
                except Exception as exc:
                    logger.error("xsec_carry: tick unhandled error — %s", exc)
                # Sleep for REBALANCE_INTERVAL, but wake immediately on stop()
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=REBALANCE_INTERVAL
                    )
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            pass
        finally:
            logger.info("XsecCarryEngine stopped")

    def stop(self) -> None:
        """Signal the run loop to exit cleanly."""
        self._stop_event.set()
