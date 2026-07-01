"""
XsecDriverEngine — generic runtime-instantiable cross-sectional driver engine.

Pluggable driver over a coin basket (Hyperliquid candleSnapshot history):
  - "realized_vol_carry": trailing realized vol; sign=-1 = LONG low-vol / SHORT high-vol
  - "dollar_volume":      trailing mean dollar volume; sign=+1 = LONG high / SHORT low

sign convention: score = -sign * driver_value, then rank_basket(score, q) gives
long=bottom-q:
  realized_vol, sign=-1 → score=+vol → long=low-vol ✓
  dollar_volume, sign=+1 → score=-dv  → long=high-dv ✓

Lifecycle + executor duck-type identical to XsecCarryEngine.
Standalone class — NOT a BaseStrategy subclass.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional

import requests

from .xsec_engine import _StratShim, rank_basket  # reuse proven pure helpers

logger = logging.getLogger(__name__)

SUPPORTED_DRIVERS = {"realized_vol_carry", "dollar_volume", "st_reversal", "amihud_illiq"}

# candle duration per supported timeframe; the fetch window is lookback bars of
# THIS size (was hardcoded 1h — a 4h instance then fetched ~lookback/4 bars,
# _compute_driver_value returned None for every coin, and the instance skipped
# every tick forever: live 0-trade bug on disc_realized_vol_carry_4h_lb96)
BAR_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}

DEFAULT_COINS = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "AVAX", "LINK", "SUI", "ARB"
]


class XsecDriverEngine:
    """Async cross-sectional driver engine with pluggable signal.

    Lifecycle:
        engine = XsecDriverEngine(executor, client, name="vol_carry",
                                   driver="realized_vol_carry", ...)
        task   = asyncio.create_task(engine.run())
        ...
        engine.stop()
        await task
    """

    def __init__(
        self,
        executor,
        client,
        name: str,
        driver: str,
        lookback: int,
        q: float,
        sign: int,
        coins: Optional[List[str]],
        per_leg_usd: float,
        rebalance_secs: int,
        timeframe: str = "1h",
    ):
        if driver not in SUPPORTED_DRIVERS:
            raise ValueError(
                f"Unknown driver '{driver}'. Supported: {sorted(SUPPORTED_DRIVERS)}"
            )
        if timeframe not in BAR_MS:
            raise ValueError(
                f"Unknown timeframe '{timeframe}'. Supported: {sorted(BAR_MS)}"
            )
        self._executor = executor
        self._client = client
        self._name = name
        self._driver = driver
        self._lookback = lookback
        self._q = q
        self._sign = sign
        self._coins: List[str] = list(coins or DEFAULT_COINS)
        self._per_leg_usd = per_leg_usd
        self._rebalance_secs = rebalance_secs
        self._timeframe = timeframe
        self._open_legs: Dict[str, str] = {}
        self._stop_event = asyncio.Event()

    # ── Driver computation ───────────────────────────────────────────────────

    def _compute_driver_value(self, candles: list) -> Optional[float]:
        """Compute driver value for one coin from its closed candle list.

        No lookahead: uses only closed bars (last N candles, index -lookback onwards).
        Returns None if insufficient data.
        """
        if len(candles) < self._lookback:
            return None
        bars = candles[-self._lookback:]

        closes = [float(c["c"]) for c in bars]
        if len(closes) < 2:
            return None
        returns = [
            (closes[i] - closes[i - 1]) / closes[i - 1]
            for i in range(1, len(closes))
        ]

        if self._driver == "realized_vol_carry":
            mean_r = sum(returns) / len(returns)
            variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
            return variance ** 0.5

        if self._driver == "st_reversal":
            # trailing cumulative return (matches edge_engine rolling-sum backtest);
            # sign=-1 -> long recent losers / short recent winners
            return sum(returns)

        if self._driver == "amihud_illiq":
            # Amihud (2002) price-impact proxy: mean |ret| per dollar traded
            ratios = []
            for i in range(1, len(bars)):
                dv = float(bars[i]["c"]) * float(bars[i]["v"])
                if dv > 0:
                    ratios.append(abs(returns[i - 1]) / dv)
            return sum(ratios) / len(ratios) if ratios else None

        # driver == "dollar_volume"
        dvols = [float(c["c"]) * float(c["v"]) for c in bars]
        return sum(dvols) / len(dvols)

    def _fetch_coin_scores(self) -> Optional[Dict[str, float]]:
        """Fetch candleSnapshot for all coins, compute driver values, apply sign.

        Returns {coin: score} where score = -sign * driver_value, or None on
        insufficient data (<4 coins).  rank_basket(score) then gives long=bottom-q
        matching the intended sign convention.
        """
        end_ms = int(time.time() * 1000)
        # +2 bars of slack so the last bar is always a closed bar
        bar_ms = BAR_MS[self._timeframe]
        start_ms = end_ms - (self._lookback + 2) * bar_ms

        scores: Dict[str, float] = {}
        for coin in self._coins:
            try:
                r = requests.post(
                    f"{self._client.base_url}/info",
                    headers={"Content-Type": "application/json"},
                    json={
                        "type": "candleSnapshot",
                        "req": {
                            "coin": coin,
                            "interval": self._timeframe,
                            "startTime": start_ms,
                            "endTime": end_ms,
                        },
                    },
                    timeout=10,
                )
                r.raise_for_status()
                candles = r.json()
                val = self._compute_driver_value(candles)
                if val is not None:
                    scores[coin] = -self._sign * val  # ponytail: sign inversion; see module docstring
            except Exception as exc:
                logger.debug(
                    "xsec_driver(%s): candle fetch %s failed — %s", self._name, coin, exc
                )

        return scores if len(scores) >= 4 else None

    # ── Leg management (mirrors XsecCarryEngine) ────────────────────────────

    async def _open_leg(self, coin: str, side: str, score: float) -> None:
        from src.strategies.base_strategy import Signal, SignalType, StrategyConfig, StrategyTier
        sig = Signal(
            signal_type=SignalType.LONG if side == "long" else SignalType.SHORT,
            symbol=coin,
            strength=1.0,
            size_usd=self._per_leg_usd,
            reason=(
                f"xsec_driver({self._name}) {side} | "
                f"driver={self._driver} score={score:.4f}"
            ),
        )
        cfg = StrategyConfig(
            name=self._name,
            symbol=coin,
            tier=StrategyTier.E,
            size_usd=self._per_leg_usd,
        )
        result = await self._executor.execute_signal(sig, _StratShim(cfg))
        if result.success:
            self._open_legs[coin] = side
            logger.info("xsec_driver(%s): opened %s %s", self._name, side, coin)
        else:
            logger.debug(
                "xsec_driver(%s): open %s %s skipped — %s",
                self._name, side, coin, result.error,
            )

    async def _close_leg(self, coin: str, side: str) -> None:
        from src.strategies.base_strategy import Signal, SignalType, StrategyConfig, StrategyTier
        sig = Signal(
            signal_type=SignalType.CLOSE_LONG if side == "long" else SignalType.CLOSE_SHORT,
            symbol=coin,
            reason=f"xsec_driver({self._name}): {coin} exited basket",
        )
        cfg = StrategyConfig(
            name=self._name,
            symbol=coin,
            tier=StrategyTier.E,
            size_usd=self._per_leg_usd,
        )
        result = await self._executor.execute_signal(sig, _StratShim(cfg))
        if result.success:
            self._open_legs.pop(coin, None)
            logger.info("xsec_driver(%s): closed %s %s", self._name, side, coin)
        else:
            # Position may not exist yet (entry was rejected); still clear tracking
            self._open_legs.pop(coin, None)
            logger.debug(
                "xsec_driver(%s): close %s %s — %s", self._name, side, coin, result.error
            )

    async def _tick(self) -> None:
        """One rebalance tick: fetch driver values, rank, diff basket, execute."""
        scores = await asyncio.to_thread(self._fetch_coin_scores)
        if scores is None:
            logger.warning(
                "xsec_driver(%s): insufficient coin data — skipping tick", self._name
            )
            return

        want_long, want_short = rank_basket(scores, self._q)

        # Close legs no longer in desired basket
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
                logger.warning(
                    "xsec_driver(%s): close %s %s error — %s", self._name, side, coin, exc
                )

        # Open new basket legs (skip already-open)
        opens = (
            [(c, "long")  for c in want_long  if c not in self._open_legs] +
            [(c, "short") for c in want_short if c not in self._open_legs]
        )
        for coin, side in opens:
            try:
                await self._open_leg(coin, side, scores[coin])
            except Exception as exc:
                logger.warning(
                    "xsec_driver(%s): open %s %s error — %s", self._name, side, coin, exc
                )

    # ── Public interface ─────────────────────────────────────────────────────

    async def run(self) -> None:
        """Main loop — run as asyncio.create_task(engine.run())."""
        logger.info(
            "XsecDriverEngine started | name=%s | driver=%s | sign=%+d | "
            "lb=%d | q=%.0f%% | per_leg=$%.0f | interval=%ds",
            self._name, self._driver, self._sign,
            self._lookback, self._q * 100, self._per_leg_usd, self._rebalance_secs,
        )
        try:
            while not self._stop_event.is_set():
                try:
                    await self._tick()
                except Exception as exc:
                    logger.error(
                        "xsec_driver(%s): tick unhandled — %s", self._name, exc
                    )
                # Sleep for rebalance interval, wake immediately on stop()
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self._rebalance_secs
                    )
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            pass
        finally:
            logger.info("XsecDriverEngine(%s) stopped", self._name)

    def stop(self) -> None:
        """Signal the run loop to exit cleanly."""
        self._stop_event.set()
