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
import os
import time
from typing import Dict, List, Optional

import requests

from .xsec_engine import _StratShim, rank_basket  # reuse proven pure helpers

logger = logging.getLogger(__name__)

# A failed tick (insufficient data — HL 429 burst) retries after this many
# seconds instead of the full rebalance interval; a daily-rebalance ensemble
# must not sit dark for 24h because one fetch window got rate-limited.
TICK_RETRY_SECS = int(os.getenv("XSEC_TICK_RETRY_SECS", "900"))

BASE_DRIVERS = {"realized_vol_carry", "dollar_volume", "st_reversal",
                "amihud_illiq", "return_skew"}
# "ensemble" = factor-momentum blend of base drivers (params.members): each
# member's cross-sectional rank is flipped when its own trailing trail_days
# gross return is negative (Gupta-Kelly factor momentum), then ranks average
# and the basket is top/bottom-q of the blend. Members too weak to pass the
# discovery gate individually can pass it combined, and the combined unit
# trades live as ONE pnl stream. Mirrors edge_engine._eval_xsec_ensemble.
SUPPORTED_DRIVERS = BASE_DRIVERS | {"ensemble"}
DEFAULT_TRAIL_DAYS = 14

# candle duration per supported timeframe; the fetch window is lookback bars of
# THIS size (was hardcoded 1h — a 4h instance then fetched ~lookback/4 bars,
# _compute_driver_value returned None for every coin, and the instance skipped
# every tick forever: live 0-trade bug on disc_realized_vol_carry_4h_lb96)
BAR_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}

DEFAULT_COINS = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "AVAX", "LINK", "SUI", "ARB"
]


def compute_driver_value(candles: list, driver: str, lookback: int) -> Optional[float]:
    """Base-driver value for one coin from its closed candle list.

    No lookahead: uses only closed bars (last `lookback` candles).
    Returns None if insufficient data. Matches edge_engine backtest math.
    """
    if len(candles) < lookback:
        return None
    bars = candles[-lookback:]

    closes = [float(c["c"]) for c in bars]
    if len(closes) < 2:
        return None
    returns = [
        (closes[i] - closes[i - 1]) / closes[i - 1]
        for i in range(1, len(closes))
    ]

    if driver == "realized_vol_carry":
        mean_r = sum(returns) / len(returns)
        variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
        return variance ** 0.5

    if driver == "st_reversal":
        # trailing cumulative return (matches edge_engine rolling-sum backtest);
        # sign=-1 -> long recent losers / short recent winners
        return sum(returns)

    if driver == "return_skew":
        # sample skewness of returns (matches pandas rolling.skew, which is the
        # bias-corrected G1); sign=-1 -> long low-skew / short high-skew.
        n = len(returns)
        if n < 3:
            return None
        mean_r = sum(returns) / n
        m2 = sum((r - mean_r) ** 2 for r in returns) / n
        if m2 <= 0:
            return None
        m3 = sum((r - mean_r) ** 3 for r in returns) / n
        g1 = m3 / (m2 ** 1.5)
        # bias correction to match pandas .skew()
        return g1 * ((n * (n - 1)) ** 0.5) / (n - 2)

    if driver == "amihud_illiq":
        # Amihud (2002) price-impact proxy: mean |ret| per dollar traded
        ratios = []
        for i in range(1, len(bars)):
            dv = float(bars[i]["c"]) * float(bars[i]["v"])
            if dv > 0:
                ratios.append(abs(returns[i - 1]) / dv)
        return sum(ratios) / len(ratios) if ratios else None

    if driver == "dollar_volume":
        dvols = [float(c["c"]) * float(c["v"]) for c in bars]
        return sum(dvols) / len(dvols)

    return None


def _normalized_ranks(scores: Dict[str, float]) -> Dict[str, float]:
    """Ascending normalized rank in [0,1] of a {coin: value} dict (higher value
    = higher rank)."""
    denom = max(1, len(scores) - 1)
    ranked = sorted(scores, key=lambda c: scores[c])
    return {coin: r / denom for r, coin in enumerate(ranked)}


def ensemble_scores(candles_by_coin: Dict[str, list],
                    members: List[dict],
                    bars_per_day: int,
                    trail_days: int,
                    q: float) -> Optional[Dict[str, float]]:
    """Factor-momentum signed-rank blend over HL candle lists ({t,c,v} dicts).

    Pure function (unit-testable, no I/O). Mirrors edge_engine's
    _eval_xsec_ensemble: daily member rebalance grid anchored at the latest
    bar, member trailing gross over trail_days decides each member's rank
    direction, blend = mean of (rank | 1-rank), returned with LOWER = more
    long-attractive so rank_basket(scores, q) picks long = bottom-q.
    Returns None on insufficient data.

    Coins are TIMESTAMP-aligned (HL omits empty bars — raw index alignment
    skews every rank and return): grid = last `need` timestamps of the densest
    coin; a coin participates only if it has every grid bar.
    """
    if not members or not candles_by_coin:
        return None
    max_lb = max(int(m["lookback"]) for m in members)
    trail = trail_days * bars_per_day
    need = max_lb + trail + 1
    dense = max(candles_by_coin, key=lambda c: len(candles_by_coin[c]))
    grid_ts = [int(b["t"]) for b in candles_by_coin[dense]][-need:]
    if len(grid_ts) < need:
        return None
    cl: Dict[str, List[float]] = {}
    vol: Dict[str, List[float]] = {}
    for coin, candles in candles_by_coin.items():
        by_t = {int(b["t"]): b for b in candles}
        if all(t in by_t for t in grid_ts):
            cl[coin] = [float(by_t[t]["c"]) for t in grid_ts]
            vol[coin] = [float(by_t[t]["v"]) for t in grid_ts]
    coins = sorted(cl)
    if len(coins) < 4:
        return None
    L = need

    def member_value(coin: str, m: dict, t: int) -> Optional[float]:
        lb = int(m["lookback"])
        candles = [{"c": cl[coin][i], "v": vol[coin][i]} for i in range(t - lb + 1, t + 1)]
        return compute_driver_value(candles, m["driver"], lb)

    # rebalance grid: latest bar and every bars_per_day back through the trail
    grid = [L - 1 - j * bars_per_day for j in range(trail_days, -1, -1)]
    if grid[0] < max_lb:
        return None

    # per-member basket weights at each grid point + gross over each span
    trailing_gross: List[float] = [0.0] * len(members)
    member_now_ranks: List[Dict[str, float]] = []
    for mi, m in enumerate(members):
        sign = int(m.get("sign", -1))
        for gi, t in enumerate(grid):
            vals = {}
            for coin in coins:
                v = member_value(coin, m, t)
                if v is not None:
                    vals[coin] = sign * v          # higher = long-attractive
            if len(vals) < 4:
                return None
            ranks = _normalized_ranks(vals)
            if gi == len(grid) - 1:
                member_now_ranks.append(ranks)
                break                              # current bar: rank only, no span after it
            ranked = sorted(vals, key=lambda c: vals[c])
            k = max(1, int(len(ranked) * q))
            w = {c: 0.0 for c in coins}
            for c in ranked[-k:]: w[c] =  1.0 / k
            for c in ranked[:k]:  w[c] = -1.0 / k
            # gross over (t, next grid point], weights applied from t+1 (no lookahead)
            t_next = grid[gi + 1]
            g = 0.0
            for b in range(t + 1, t_next + 1):
                for c in coins:
                    if w[c]:
                        g += w[c] * (cl[c][b] / cl[c][b - 1] - 1.0)
            trailing_gross[mi] += g

    common = set(member_now_ranks[0])
    for r in member_now_ranks[1:]:
        common &= set(r)
    if len(common) < 4:
        return None
    blended: Dict[str, float] = {}
    for coin in common:
        acc = 0.0
        for mi, ranks in enumerate(member_now_ranks):
            r = ranks[coin]
            acc += r if trailing_gross[mi] >= 0 else 1.0 - r
        blended[coin] = acc / len(members)
    # backtest longs TOP of blended; rank_basket longs BOTTOM of score
    return {c: -v for c, v in blended.items()}


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
        initial_legs: Optional[Dict[str, str]] = None,
        members: Optional[List[dict]] = None,
        trail_days: int = DEFAULT_TRAIL_DAYS,
    ):
        if driver not in SUPPORTED_DRIVERS:
            raise ValueError(
                f"Unknown driver '{driver}'. Supported: {sorted(SUPPORTED_DRIVERS)}"
            )
        if timeframe not in BAR_MS:
            raise ValueError(
                f"Unknown timeframe '{timeframe}'. Supported: {sorted(BAR_MS)}"
            )
        if driver == "ensemble":
            if not members:
                raise ValueError("driver 'ensemble' requires non-empty members")
            bad = [m.get("driver") for m in members if m.get("driver") not in BASE_DRIVERS]
            if bad:
                raise ValueError(f"ensemble members use unknown drivers: {bad}")
        self._executor = executor
        self._client = client
        self._name = name
        self._driver = driver
        self._members: List[dict] = list(members or [])
        self._trail_days = int(trail_days)
        self._bars_per_day = max(1, 86_400_000 // BAR_MS[timeframe])
        # candle fetch window must cover the deepest member PLUS the factor-
        # momentum trail (member gross is reconstructed from candles each tick);
        # +5%+10 slack because HL omits empty bars and the window is time-based
        if self._members:
            base = (max(int(m["lookback"]) for m in self._members)
                    + self._trail_days * self._bars_per_day)
            self._fetch_lookback = int(base * 1.05) + 10
        else:
            self._fetch_lookback = lookback
        self._lookback = lookback
        self._q = q
        self._sign = sign
        self._coins: List[str] = list(coins or DEFAULT_COINS)
        self._per_leg_usd = per_leg_usd
        self._rebalance_secs = rebalance_secs
        self._timeframe = timeframe
        # Seed from durable open positions on restart — a redeploy otherwise
        # blanks this map while the paper legs persist, so rotation can never
        # close pre-restart legs (they orphan forever).
        self._open_legs: Dict[str, str] = dict(initial_legs or {})
        self._stop_event = asyncio.Event()

    # ── Driver computation ───────────────────────────────────────────────────

    def _compute_driver_value(self, candles: list) -> Optional[float]:
        """Base-driver value using this instance's driver/lookback (delegates)."""
        return compute_driver_value(candles, self._driver, self._lookback)

    def _fetch_coin_candles(self) -> Dict[str, list]:
        """Fetch candleSnapshot per coin, window sized for the deepest lookback."""
        end_ms = int(time.time() * 1000)
        # +2 bars of slack so the last bar is always a closed bar
        bar_ms = BAR_MS[self._timeframe]
        start_ms = end_ms - (self._fetch_lookback + 2) * bar_ms

        out: Dict[str, list] = {}
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
                out[coin] = r.json()
            except Exception as exc:
                logger.debug(
                    "xsec_driver(%s): candle fetch %s failed — %s", self._name, coin, exc
                )
        return out

    def _fetch_coin_scores(self) -> Optional[Dict[str, float]]:
        """Fetch candles, compute driver scores ({coin: score}, lower = long).

        Base driver: score = -sign * driver_value, so rank_basket(score, q)
        gives long=bottom-q matching the module docstring sign convention.
        Ensemble: equal-weight rank blend of member scores (same convention).
        Returns None on insufficient data (<4 coins).
        """
        candles_by_coin = self._fetch_coin_candles()

        if self._driver == "ensemble":
            scores = ensemble_scores(candles_by_coin, self._members,
                                     self._bars_per_day, self._trail_days, self._q)
            if scores is None:
                logger.warning(
                    "xsec_driver(%s): insufficient aligned history for ensemble — skipping",
                    self._name,
                )
                return None
        else:
            scores = {}
            for coin, candles in candles_by_coin.items():
                val = compute_driver_value(candles, self._driver, self._lookback)
                if val is not None:
                    scores[coin] = -self._sign * val  # ponytail: sign inversion; see module docstring

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

    async def _tick(self) -> bool:
        """One rebalance tick: fetch driver values, rank, diff basket, execute.
        Returns False when the tick could not rebalance (insufficient data,
        e.g. an HL 429 burst) so the run loop retries sooner than the full
        rebalance interval — a failed daily-ensemble tick must not wait 24h."""
        scores = await asyncio.to_thread(self._fetch_coin_scores)
        if scores is None:
            logger.warning(
                "xsec_driver(%s): insufficient coin data — skipping tick", self._name
            )
            return False

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
        return True

    def _sleep_secs(self, tick_ok: bool) -> int:
        """Full rebalance interval after a good tick; the (never longer) retry
        interval after a failed one."""
        return self._rebalance_secs if tick_ok else min(TICK_RETRY_SECS, self._rebalance_secs)

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
                    ok = await self._tick()
                except Exception as exc:
                    ok = False
                    logger.error(
                        "xsec_driver(%s): tick unhandled — %s", self._name, exc
                    )
                # Sleep for rebalance interval (short retry after a failed
                # tick), wake immediately on stop()
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self._sleep_secs(ok)
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
