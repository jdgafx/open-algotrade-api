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
REBALANCE_INTERVAL: int = int(os.getenv("XSEC_REBALANCE_SECS", "3600"))  # 1h between full rebalances
RETRY_INTERVAL: int = int(os.getenv("XSEC_RETRY_SECS", "120"))  # short backoff when a tick skips (fetch fail / warm-up) — never go dark for a full hour on a transient miss
XSEC_Q: float = 0.30         # bottom/top 30% each
LB: int = 24                 # smoothing window (bars @ 1h = 24h)
PER_LEG_USD: float = float(os.getenv("XSEC_PER_LEG_USD", "50"))
# Liquidity floor: a coin must have >= this 24h notional volume to enter the
# basket on a given tick, so legs fill/exit cleanly. Sized to exclude only dead/
# delisted markets — HL is a smaller venue, so real majors (BNB/LINK/DOGE) run
# ~$1.5-3M/day here with deep OI, and a $150 leg is <0.4% of even that flow.
# $1M keeps every live market, cuts only the truly illiquid. Env-overridable.
MIN_DAY_VOL_USD: float = float(os.getenv("XSEC_MIN_DAY_VOL_USD", "1000000"))  # $1M

# Master on/off. Defaults OFF as of 2026-07-24: the sleeve was decertified.
# Its promotion evidence (OOS Sharpe 1.89 / 53 trades) was computed BEFORE the
# edge_probe funding-pagination fix (2026-07-02), i.e. on stale forward-filled
# rates, and was never re-validated. Live it lost -0.589% of $8,290 turnover
# over 72 legs (2026-07-18..24) with funding income already folded into exit
# pnl — the same ~-0.5%/turnover rate it had lost since inception at $5 legs,
# only visible once sizing went 30x. Allocation follows gate verdicts: this
# stays off until xsec_funding_carry re-clears the honest gate on paginated
# funding data, at which point set XSEC_CARRY_ENABLED=1.
CARRY_ENABLED: bool = os.getenv("XSEC_CARRY_ENABLED", "0").strip().lower() in (
    "1", "true", "yes", "on",
)

# Universe = majors (one correlated cluster, near-zero funding dispersion) +
# liquid alts that carry genuine funding dispersion (the tails where carry pays:
# e.g. kBONK deeply negative, CASHCAT/ZEC/VVV/XMR elevated). The MIN_DAY_VOL_USD
# floor in _fetch_all_funding keeps fills clean even as an alt's liquidity moves.
_MAJORS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "AVAX", "LINK", "SUI", "ARB"]
_LIQUID_ALTS = ["HYPE", "ZEC", "CASHCAT", "kBONK", "ONDO", "LIT", "FARTCOIN",
                "WLD", "NEAR", "LTC", "VVV", "AAVE", "XMR", "TRUMP"]
DEFAULT_COINS = _MAJORS + _LIQUID_ALTS


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


def liquid_snapshot(
    name_to_rate: Dict[str, float],
    name_to_vol: Dict[str, float],
    coins,
    min_vol: float = MIN_DAY_VOL_USD,
) -> Dict[str, float]:
    """Funding snapshot for `coins`, keeping only names with 24h notional
    volume >= min_vol (clean fills). Pure function — no side effects, testable.
    """
    return {
        c: name_to_rate[c]
        for c in coins
        if c in name_to_rate and name_to_vol.get(c, 0.0) >= min_vol
    }


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
            name_to_rate: Dict[str, float] = {}
            name_to_vol: Dict[str, float] = {}
            for i, asset in enumerate(universe):
                if i >= len(ctxs):
                    break
                name = asset["name"]
                name_to_rate[name] = float(ctxs[i].get("funding", 0.0) or 0.0)
                name_to_vol[name] = float(ctxs[i].get("dayNtlVlm", 0.0) or 0.0)
            snapshot = liquid_snapshot(name_to_rate, name_to_vol, self._coins, MIN_DAY_VOL_USD)
            if len(snapshot) < 4:
                logger.warning(
                    "xsec_carry: only %d coins cleared the $%.0fM liquidity floor",
                    len(snapshot), MIN_DAY_VOL_USD / 1e6,
                )
                return None
            return snapshot
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
            # Dollar-neutral structural sleeve: exempt from the directional
            # half-Kelly observation-floor haircut so legs trade at PER_LEG_USD,
            # not 10% of it. See paper_executor sizing.
            metadata={"market_neutral": True},
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

    async def _tick(self) -> bool:
        """One rebalance tick: fetch funding, smooth, diff basket, execute.

        Returns True only if it actually evaluated & rebalanced the basket.
        Returns False when the tick is skipped (fetch failed or still warming
        up) so the caller can retry after a SHORT backoff rather than staying
        dark for a full rebalance interval.
        """
        snapshot = await asyncio.to_thread(self._fetch_all_funding)
        if snapshot is None:
            return False  # error already logged; retry soon

        self._history.append(snapshot)

        if len(self._history) < 2:
            logger.info(
                "xsec_carry: warming up (%d/%d bars)", len(self._history), LB
            )
            return False  # not yet rebalancing — retry soon, don't sleep the full hour

        smoothed = self._smoothed()
        if not smoothed or len(smoothed) < 4:
            logger.warning("xsec_carry: insufficient coin data (%d coins)", len(smoothed or {}))
            return False

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

        return True  # full rebalance evaluated — caller sleeps the full interval

    async def _reconcile_open_legs(self) -> None:
        """Adopt pre-existing executor positions at boot.

        _open_legs is in-memory only, so before this every redeploy orphaned the
        live basket: the engine could neither rotate nor close legs a previous
        boot had opened (observed live 2026-07-18: $5 legs from the $50/leg era
        still open weeks later). Legs whose notional no longer matches
        PER_LEG_USD (config changed between boots) are closed here; the next
        tick reopens the ranked basket at the configured size.
        """
        try:
            positions = await self._executor.get_all_positions()
        except Exception as exc:
            logger.warning("xsec_carry: reconcile skipped (non-fatal) — %s", exc)
            return
        for p in positions:
            if p.get("strategy_name") != "xsec_carry":
                continue
            coin, side = p["symbol"], p["side"]
            size_usd = float(p.get("size_usd") or 0.0)
            if abs(size_usd - PER_LEG_USD) > PER_LEG_USD * 0.2:
                logger.info(
                    "xsec_carry: closing stale-size leg %s %s ($%.0f, config $%.0f)",
                    side, coin, size_usd, PER_LEG_USD,
                )
                await self._close_leg(coin, side)
            else:
                self._open_legs[coin] = side
        if self._open_legs:
            logger.info(
                "xsec_carry: reconciled %d open legs from executor", len(self._open_legs)
            )

    async def flush_all_legs(self) -> int:
        """Close every live xsec_carry leg and return how many were closed.

        Used when the sleeve is disabled: without this, gating the engine off
        would orphan its open basket in the executor with no owner left to
        close it (the exact failure 3bcb46a fixed for redeploys — $5 legs from
        the $50/leg era sat open for weeks). Unlike _reconcile_open_legs this
        closes regardless of leg notional. Never raises: a flush failure must
        not block application startup.
        """
        closed = 0
        try:
            positions = await self._executor.get_all_positions()
        except Exception as exc:
            logger.warning("xsec_carry: flush skipped (non-fatal) — %s", exc)
            return 0
        for p in positions:
            if p.get("strategy_name") != "xsec_carry":
                continue
            try:
                await self._close_leg(p["symbol"], p["side"])
                closed += 1
            except Exception as exc:
                logger.warning("xsec_carry: flush close %s failed — %s",
                               p.get("symbol"), exc)
        if closed:
            logger.warning("xsec_carry: flushed %d open legs (sleeve disabled)", closed)
        return closed

    # ── Public interface ─────────────────────────────────────────────────────

    async def run(self) -> None:
        """Main loop — run as asyncio.create_task(engine.run())."""
        logger.info(
            "XsecCarryEngine started | coins=%d | interval=%ds | lb=%d | q=%.0f%% | per_leg=$%.0f | min_vol=$%.0fM",
            len(self._coins), REBALANCE_INTERVAL, LB, XSEC_Q * 100, PER_LEG_USD, MIN_DAY_VOL_USD / 1e6,
        )
        # Adopt any legs a previous boot left open BEFORE trading decisions.
        await self._reconcile_open_legs()
        # Seed lb24 history up-front so the first tick trades with full smoothing.
        await asyncio.to_thread(self._seed_history)
        try:
            while not self._stop_event.is_set():
                did_rebalance = False
                try:
                    did_rebalance = await self._tick()
                except Exception as exc:
                    logger.error("xsec_carry: tick unhandled error — %s", exc)
                # Full interval after a real rebalance; short retry after a skip
                # (fetch fail / warm-up) so a transient miss never goes dark for 1h.
                timeout = REBALANCE_INTERVAL if did_rebalance else RETRY_INTERVAL
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=timeout
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
