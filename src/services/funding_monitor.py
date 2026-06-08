"""
Funding Rate Monitor -- Tracks Hyperliquid perpetual funding rates across all symbols.

Fetches funding rates every 15 minutes from the metaAndAssetCtxs endpoint,
detects crowded positioning via extreme rates, and provides confidence
adjustments the orchestrator can use as a signal filter.

Thresholds (annualized):
    |rate| > 100%  = extreme
    |rate| > 50%   = high
    |rate| > 20%   = elevated
    else            = normal

Contrarian logic:
    Going WITH extreme funding  -> reduce confidence (crowd is on same side)
    Going AGAINST extreme funding -> boost confidence (contrarian edge)
"""

import asyncio
import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Hyperliquid funding rates are 8-hour rates. Annualize = rate * 3 * 365.
HOURS_PER_FUNDING_PERIOD = 8
FUNDING_PERIODS_PER_YEAR = (365 * 24) / HOURS_PER_FUNDING_PERIOD  # 1095

# Annualized threshold constants
EXTREME_THRESHOLD = 1.00   # 100%
HIGH_THRESHOLD = 0.50      # 50%
ELEVATED_THRESHOLD = 0.20  # 20%

# Confidence adjustment magnitudes
CROWD_PENALTY = 0.3    # reduce when agreeing with extreme crowd
CONTRARIAN_BONUS = 0.2  # boost when fading extreme crowd

DEFAULT_REFRESH_INTERVAL = 900  # 15 minutes in seconds
HISTORY_RETENTION = timedelta(hours=24)

# Funding-cost gate: block long entries when 8h funding exceeds this threshold.
# 0.0001 = 0.01%/8h — longs pay this much per 8-hour period.
FUNDING_LONG_SUPPRESS_THRESHOLD = 0.0001

HL_MAINNET_INFO_URL = "https://api.hyperliquid.xyz/info"


class FundingMonitor:
    """
    Periodically fetches and caches Hyperliquid funding rates for every listed
    perpetual.  Exposes helpers the orchestrator (or any caller) can use to
    gate or adjust signal confidence based on funding-rate positioning.
    """

    def __init__(
        self,
        base_url: str = HL_MAINNET_INFO_URL,
        refresh_interval: int = DEFAULT_REFRESH_INTERVAL,
        auto_start: bool = True,
    ):
        self._base_url = base_url
        self._refresh_interval = refresh_interval

        # symbol -> latest annualized funding rate
        self._rates: Dict[str, float] = {}

        # symbol -> list of (datetime, annualized_rate) tuples, capped at 24h
        self._funding_history: Dict[str, List[Tuple[datetime, float]]] = {}

        # symbol -> raw 8h rate (before annualizing) kept for diagnostics
        self._raw_rates: Dict[str, float] = {}

        self._last_refresh: Optional[datetime] = None
        self._universe: List[str] = []

        # Background refresh thread
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        if auto_start:
            self.start()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start the background refresh thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._thread.start()
        logger.info(
            "FundingMonitor started | url=%s | interval=%ds",
            self._base_url, self._refresh_interval,
        )

    def stop(self):
        """Signal the background thread to stop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("FundingMonitor stopped")

    # ------------------------------------------------------------------
    # Background refresh
    # ------------------------------------------------------------------

    def _refresh_loop(self):
        """Fetch funding rates on a fixed interval until stopped."""
        # Do an immediate first fetch, then sleep between subsequent ones.
        while not self._stop_event.is_set():
            try:
                self._fetch_funding_rates()
            except Exception as e:
                logger.error("FundingMonitor refresh failed: %s", e)
            self._stop_event.wait(timeout=self._refresh_interval)

    def _fetch_funding_rates(self):
        """
        POST to metaAndAssetCtxs and parse per-asset funding rates.

        Response shape:
            [
                {"universe": [{"name": "BTC", ...}, ...]},  # meta
                [{"funding": "0.00001234", ...}, ...]         # assetCtxs
            ]
        """
        payload = {"type": "metaAndAssetCtxs"}
        headers = {"Content-Type": "application/json"}

        resp = requests.post(self._base_url, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        result = resp.json()

        if not isinstance(result, list) or len(result) < 2:
            logger.warning("Unexpected metaAndAssetCtxs response shape")
            return

        universe = result[0].get("universe", [])
        ctxs = result[1]

        now = datetime.now(timezone.utc)
        fetched = 0

        for i, asset_meta in enumerate(universe):
            symbol = asset_meta.get("name", "")
            if not symbol or i >= len(ctxs):
                continue

            raw_rate = float(ctxs[i].get("funding", 0))
            annualized = raw_rate * FUNDING_PERIODS_PER_YEAR

            self._raw_rates[symbol] = raw_rate
            self._rates[symbol] = annualized

            # Append to history and prune entries older than 24h
            if symbol not in self._funding_history:
                self._funding_history[symbol] = []
            self._funding_history[symbol].append((now, annualized))
            cutoff = now - HISTORY_RETENTION
            self._funding_history[symbol] = [
                (ts, r) for ts, r in self._funding_history[symbol] if ts >= cutoff
            ]

            fetched += 1

        self._universe = [a["name"] for a in universe if "name" in a]
        self._last_refresh = now
        logger.info(
            "FundingMonitor refreshed | %d symbols | ts=%s",
            fetched, now.strftime("%H:%M:%S"),
        )

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    def get_funding_rate(self, symbol: str) -> Optional[float]:
        """
        Current annualized funding rate for *symbol*.

        Returns None if the symbol has not been fetched yet.
        """
        return self._rates.get(symbol)

    def get_funding_bias(self, symbol: str) -> str:
        """
        Crowd-positioning label derived from the current funding rate.

        Returns one of: "long_crowded", "short_crowded", "neutral".
        """
        rate = self._rates.get(symbol)
        if rate is None:
            return "neutral"
        if rate > HIGH_THRESHOLD:
            return "long_crowded"
        if rate < -HIGH_THRESHOLD:
            return "short_crowded"
        return "neutral"

    def classify_rate(self, rate: float) -> str:
        """
        Classify an annualized rate into a severity bucket.

        Returns one of: "extreme", "high", "elevated", "normal".
        """
        abs_rate = abs(rate)
        if abs_rate > EXTREME_THRESHOLD:
            return "extreme"
        if abs_rate > HIGH_THRESHOLD:
            return "high"
        if abs_rate > ELEVATED_THRESHOLD:
            return "elevated"
        return "normal"

    def get_top_anomalies(self, n: int = 5) -> List[dict]:
        """
        Return the *n* symbols with the most extreme annualized funding rates,
        sorted by absolute magnitude descending.
        """
        items = [
            {
                "symbol": sym,
                "annualized_rate": rate,
                "raw_8h_rate": self._raw_rates.get(sym, 0.0),
                "bias": self.get_funding_bias(sym),
                "severity": self.classify_rate(rate),
            }
            for sym, rate in self._rates.items()
        ]
        items.sort(key=lambda x: abs(x["annualized_rate"]), reverse=True)
        return items[:n]

    async def get_current_funding(self, symbol: str) -> float:
        """
        Fetch the live 8h funding rate for *symbol* and return it as a decimal.

        Makes a fresh HTTP request (via asyncio.to_thread so the async caller is
        not blocked) rather than relying on the 15-minute background cache, giving
        the entry gate an up-to-the-second reading.

        Returns 0.0 on any error (fail-open: entry is not suppressed when data is
        unavailable).

        Example: 0.0001 = 0.01%/8h (longs are paying shorts).
        """
        try:
            def _fetch() -> float:
                payload = {"type": "metaAndAssetCtxs"}
                headers = {"Content-Type": "application/json"}
                resp = requests.post(
                    self._base_url, headers=headers, json=payload, timeout=10
                )
                resp.raise_for_status()
                result = resp.json()
                if not isinstance(result, list) or len(result) < 2:
                    return 0.0
                universe = result[0].get("universe", [])
                ctxs = result[1]
                for i, asset_meta in enumerate(universe):
                    if asset_meta.get("name") == symbol and i < len(ctxs):
                        return float(ctxs[i].get("funding", 0))
                return 0.0

            return await asyncio.to_thread(_fetch)
        except Exception as exc:
            logger.warning(
                "get_current_funding(%s) failed — fail-open (returning 0.0): %s",
                symbol, exc,
            )
            return 0.0

    # ------------------------------------------------------------------
    # Summary / serialization
    # ------------------------------------------------------------------

    def get_funding_summary(self) -> dict:
        """
        Full snapshot suitable for the /funding/rates API endpoint.

        Returns dict with keys: last_refresh, symbol_count, rates, anomalies, biases.
        """
        rates = {}
        for sym in sorted(self._rates):
            rate = self._rates[sym]
            rates[sym] = {
                "annualized_rate": rate,
                "raw_8h_rate": self._raw_rates.get(sym, 0.0),
                "severity": self.classify_rate(rate),
                "bias": self.get_funding_bias(sym),
            }

        return {
            "last_refresh": (
                self._last_refresh.isoformat() if self._last_refresh else None
            ),
            "symbol_count": len(self._rates),
            "rates": rates,
            "anomalies": self.get_top_anomalies(10),
            "biases": {
                sym: self.get_funding_bias(sym)
                for sym in sorted(self._rates)
            },
        }

    def get_history(self, symbol: str, limit: int = 96) -> List[dict]:
        """
        Return recent funding rate history for *symbol*.

        Default limit of 96 = 24 hours at 15-min intervals.
        """
        entries = self._funding_history.get(symbol, [])
        return [
            {"timestamp": ts.isoformat(), "annualized_rate": r}
            for ts, r in entries[-limit:]
        ]


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------

_monitor: Optional[FundingMonitor] = None


def get_funding_monitor(
    base_url: str = HL_MAINNET_INFO_URL,
    auto_start: bool = True,
) -> FundingMonitor:
    """Get or create the global FundingMonitor singleton."""
    global _monitor
    if _monitor is None:
        _monitor = FundingMonitor(base_url=base_url, auto_start=auto_start)
    return _monitor
