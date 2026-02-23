"""
MoonDev Data Layer API Client for Open Algotrade
Wraps Moon Dev's API (https://api.moondev.com) for:
- Liquidation data (HL, Binance, Bybit, OKX, HIP3)
- Whale tracking & depositor data
- Order flow & trade imbalance
- Smart money signals & rankings
- Market data (prices, orderbooks, candles — no rate limits!)
- HLP sentiment & positioning
- User positions & fills

Enhanced with:
- Retry logic with exponential backoff
- Async wrappers via asyncio.to_thread
- Module-level singleton factory
- Structured logging
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class MoonDevAPI:
    """Moon Dev Data Layer API client — enhanced for Open Algotrade."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.moondev.com",
        max_retries: int = 3,
        timeout: int = 30,
    ):
        self.api_key = api_key or os.getenv("MOONDEV_API_KEY")
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        self.headers = {"X-API-Key": self.api_key} if self.api_key else {}
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

        logger.info(
            "MoonDevAPI initialized | base_url=%s | key=%s",
            base_url,
            f"...{self.api_key[-4:]}" if self.api_key else "none",
        )

    def _get(
        self, endpoint: str, auth_required: bool = True, **kwargs
    ) -> requests.Response:
        """Make GET request with retry logic and exponential backoff."""
        url = f"{self.base_url}{endpoint}"
        headers = self.headers if auth_required else {}

        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    url, headers=headers, timeout=self.timeout, **kwargs
                )
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    logger.error(
                        "MoonDevAPI | %s | failed after %d attempts: %s",
                        endpoint,
                        self.max_retries,
                        e,
                    )
                    raise
                wait = 2**attempt
                logger.warning(
                    "MoonDevAPI | %s | attempt %d failed: %s | retrying in %ds",
                    endpoint,
                    attempt + 1,
                    e,
                    wait,
                )
                time.sleep(wait)

        # Should never reach here, but satisfy type checker
        raise RuntimeError("Unreachable")

    def _post(self, url: str, data: Dict, **kwargs) -> requests.Response:
        """Make POST request (for direct Hyperliquid API calls)."""
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    url, json=data, timeout=self.timeout, **kwargs
                )
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError("Unreachable")

    # ==================== HEALTH ====================

    def health(self) -> Dict:
        """Check API health status (no auth required)."""
        return self._get("/health", auth_required=False).json()

    # ==================== LIQUIDATIONS ====================

    def get_liquidations(self, timeframe: str = "1h") -> Dict:
        """Get liquidation data. Timeframes: 10m, 1h, 4h, 12h, 24h, 2d, 7d, 14d, 30d."""
        return self._get(f"/api/liquidations/{timeframe}.json").json()

    def get_liquidation_stats(self) -> Dict:
        """Get aggregated liquidation stats across all timeframes."""
        return self._get("/api/liquidations/stats.json").json()

    # ==================== MULTI-EXCHANGE LIQUIDATIONS ====================

    def get_all_liquidations(self, timeframe: str = "1h") -> Dict:
        """Get COMBINED liquidation data from ALL exchanges (HL, Binance, Bybit, OKX)."""
        return self._get(f"/api/all_liquidations/{timeframe}.json").json()

    def get_all_liquidation_stats(self) -> Dict:
        """Get combined liquidation stats across ALL exchanges."""
        return self._get("/api/all_liquidations/stats.json").json()

    def get_binance_liquidations(self, timeframe: str = "1h") -> Dict:
        """Get Binance Futures liquidation data."""
        return self._get(f"/api/binance_liquidations/{timeframe}.json").json()

    def get_bybit_liquidations(self, timeframe: str = "1h") -> Dict:
        """Get Bybit liquidation data."""
        return self._get(f"/api/bybit_liquidations/{timeframe}.json").json()

    def get_okx_liquidations(self, timeframe: str = "1h") -> Dict:
        """Get OKX liquidation data."""
        return self._get(f"/api/okx_liquidations/{timeframe}.json").json()

    # ==================== HIP3 LIQUIDATIONS ====================

    def get_hip3_liquidations(self, timeframe: str = "1h") -> Dict:
        """Get HIP3 liquidation data (Stocks, Commodities, Indices, FX). Timeframes: 10m, 1h, 24h, 7d."""
        return self._get(f"/api/hip3_liquidations/{timeframe}.json").json()

    def get_hip3_liquidation_stats(self) -> Dict:
        """Get HIP3 liquidation statistics."""
        return self._get("/api/hip3_liquidations/stats.json").json()

    # ==================== POSITIONS ====================

    def get_positions(self) -> Dict:
        """Get large positions near liquidation ($200k+) — top 50 across ALL symbols."""
        return self._get("/api/positions.json").json()

    def get_all_positions(self) -> Dict:
        """Get ALL positions for all 148 symbols — top 50 longs/shorts per symbol."""
        return self._get("/api/positions/all.json").json()

    # ==================== WHALES ====================

    def get_whales(self) -> Dict:
        """Get recent whale trades ($25k+)."""
        return self._get("/api/whales.json").json()

    def get_whale_addresses(self) -> List[str]:
        """Get plain text list of known whale addresses."""
        response = self._get("/api/whale_addresses.txt")
        return [
            addr.strip() for addr in response.text.strip().split("\n") if addr.strip()
        ]

    def get_buyers(self) -> Dict:
        """Get recent $5k+ buyers on HYPE/SOL/XRP/ETH."""
        return self._get("/api/buyers.json").json()

    def get_depositors(self) -> Dict:
        """Get all Hyperliquid depositors — every address that bridged USDC."""
        return self._get("/api/depositors.json").json()

    # ==================== EVENTS & CONTRACTS ====================

    def get_events(self) -> Dict:
        """Get real-time blockchain events."""
        return self._get("/api/events.json").json()

    def get_contracts(self) -> Dict:
        """Get contract registry with metadata and activity tracking."""
        return self._get("/api/contracts.json").json()

    # ==================== TICK DATA ====================

    def get_tick_stats(self) -> Dict:
        """Get tick data collection stats and summary."""
        return self._get("/api/ticks/stats.json").json()

    def get_tick_latest(self) -> Dict:
        """Get latest prices for all symbols."""
        return self._get("/api/ticks/latest.json").json()

    def get_ticks(
        self,
        symbol: str = "BTC",
        duration: str = "1h",
        limit: int = 10000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> Dict:
        """
        Get historical tick data for any of 80 tracked symbols.

        Args:
            symbol: Any tracked symbol (BTC, ETH, SOL, DOGE, etc.)
            duration: Time window — 10m, 1h, 4h, 24h, 7d
            limit: Max ticks to return (default: 10000)
            start_time: Start time in Unix ms (optional)
            end_time: End time in Unix ms (optional)
        """
        params = [f"duration={duration}", f"limit={limit}"]
        if start_time is not None:
            params.append(f"startTime={start_time}")
        if end_time is not None:
            params.append(f"endTime={end_time}")
        query = "?" + "&".join(params)
        return self._get(f"/api/ticks/{symbol.upper()}{query}").json()

    # ==================== ORDER FLOW & TRADES ====================

    def get_trades(self) -> Any:
        """Get recent 500 trades (real-time)."""
        return self._get("/api/trades.json").json()

    def get_large_trades(self) -> Any:
        """Get large trades >$100k (24h)."""
        return self._get("/api/large_trades.json").json()

    def get_orderflow(self) -> Dict:
        """Get order flow imbalance by timeframe + per coin."""
        return self._get("/api/orderflow.json").json()

    def get_orderflow_stats(self) -> Dict:
        """Get order flow service stats (uptime, trades/sec)."""
        return self._get("/api/orderflow/stats.json").json()

    def get_imbalance(self, timeframe: str = "1h") -> Dict:
        """Get buy/sell imbalance. Timeframes: 5m, 15m, 1h, 4h, 24h."""
        return self._get(f"/api/imbalance/{timeframe}.json").json()

    # ==================== SMART MONEY ====================

    def get_smart_money_rankings(self) -> Dict:
        """Get Top 100 smart money + Bottom 100 dumb money rankings."""
        return self._get("/api/smart_money/rankings.json").json()

    def get_smart_money_leaderboard(self) -> Dict:
        """Get Top 50 performers with details."""
        return self._get("/api/smart_money/leaderboard.json").json()

    def get_smart_money_signals(self, timeframe: str = "1h") -> Dict:
        """Get smart money trading signals. Timeframes: 10m, 1h, 24h."""
        return self._get(f"/api/smart_money/signals_{timeframe}.json").json()

    # ==================== USER DATA (HYPERLIQUID) ====================

    def get_user_positions(self, address: str) -> Dict:
        """Get all open positions for a Hyperliquid wallet (direct API call)."""
        url = "https://api.hyperliquid.xyz/info"
        payload = {"type": "clearinghouseState", "user": address}
        logger.debug("Fetching positions for %s...%s", address[:6], address[-4:])
        return self._post(url, payload).json()

    def get_user_positions_api(self, address: str) -> Dict:
        """Get positions via Moon Dev's API (faster, local node data)."""
        return self._get(f"/api/user/{address}/positions").json()

    def get_user_fills(self, address: str, limit: int = 100) -> Dict:
        """Get historical fills for a wallet. limit=-1 for ALL fills."""
        params = f"?limit={limit}" if limit != 100 else ""
        return self._get(f"/api/user/{address}/fills{params}").json()

    # ==================== POSITION SNAPSHOTS ====================

    def get_position_snapshots(
        self,
        symbol: str,
        hours: int = 24,
        limit: int = 1000,
        min_distance_pct: Optional[float] = None,
        max_distance_pct: Optional[float] = None,
        side: Optional[str] = None,
    ) -> Dict:
        """Get historical position snapshots near liquidation (1-min intervals)."""
        params = f"?hours={hours}&limit={limit}"
        if min_distance_pct is not None:
            params += f"&min_distance_pct={min_distance_pct}"
        if max_distance_pct is not None:
            params += f"&max_distance_pct={max_distance_pct}"
        if side is not None:
            params += f"&side={side}"
        return self._get(f"/api/position_snapshots/symbol/{symbol}{params}").json()

    def get_position_snapshot_stats(self, hours: int = 24) -> Dict:
        """Get aggregate stats for position snapshots across all tracked symbols."""
        return self._get(f"/api/position_snapshots/stats?hours={hours}").json()

    # ==================== MARKET DATA (NO RATE LIMITS!) ====================

    def get_prices(self) -> Dict:
        """Get all 224 coin prices, funding rates, and open interest. No rate limits!"""
        return self._get("/api/prices").json()

    def get_price(self, coin: str) -> Dict:
        """Get quick price for a single coin (bid/ask/mid/spread)."""
        return self._get(f"/api/price/{coin}").json()

    def get_orderbook(self, coin: str) -> Dict:
        """Get full L2 orderbook (~20 levels each side). No rate limits!"""
        return self._get(f"/api/orderbook/{coin}").json()

    def get_account(self, address: str) -> Dict:
        """Get full account state for any wallet. No rate limits!"""
        return self._get(f"/api/account/{address}").json()

    def get_fills(self, address: str, limit: int = 100) -> Any:
        """Get trade fills in Hyperliquid-compatible format. No rate limits!"""
        params = f"?limit={limit}" if limit != 100 else ""
        return self._get(f"/api/fills/{address}{params}").json()

    def get_candle_symbols(self) -> Dict:
        """Get list of all 80 tracked symbols available for candles/ticks."""
        return self._get("/api/candles/symbols").json()

    def get_candles(
        self,
        coin: str,
        interval: str = "5m",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> Any:
        """
        Get OHLCV candles for any of 80 tracked symbols.

        Args:
            coin: Any tracked symbol (BTC, ETH, SOL, etc.)
            interval: Candle interval — 1m, 5m, 15m, 1h, 4h, 1d
            start_time: Start timestamp in ms (optional)
            end_time: End timestamp in ms (optional)
        """
        params = [f"interval={interval}"]
        if start_time is not None:
            params.append(f"startTime={start_time}")
        if end_time is not None:
            params.append(f"endTime={end_time}")
        query = "?" + "&".join(params) if params else ""
        return self._get(f"/api/candles/{coin}{query}").json()

    # ==================== HLP (HYPERLIQUIDITY PROVIDER) ====================

    def get_hlp_positions(self, include_strategies: bool = True) -> Dict:
        """Get all HLP positions across all 7 strategies."""
        params = "" if include_strategies else "?include_strategies=false"
        return self._get(f"/api/hlp/positions{params}").json()

    def get_hlp_trades(self, limit: int = 100) -> Dict:
        """Get historical HLP trade fills."""
        params = f"?limit={limit}" if limit != 100 else ""
        return self._get(f"/api/hlp/trades{params}").json()

    def get_hlp_trade_stats(self) -> Dict:
        """Get HLP trade volume and fee statistics."""
        return self._get("/api/hlp/trades/stats").json()

    def get_hlp_position_history(self, hours: int = 24) -> Dict:
        """Get historical position snapshots over time."""
        params = f"?hours={hours}" if hours != 24 else ""
        return self._get(f"/api/hlp/positions/history{params}").json()

    def get_hlp_liquidators(self) -> Dict:
        """Get HLP liquidator activation events."""
        return self._get("/api/hlp/liquidators").json()

    def get_hlp_deltas(self, hours: int = 24) -> Dict:
        """Get HLP net exposure (delta) changes over time."""
        params = f"?hours={hours}" if hours != 24 else ""
        return self._get(f"/api/hlp/deltas{params}").json()

    def get_hlp_sentiment(self) -> Dict:
        """
        Get HLP sentiment indicator — THE BIG ONE!
        Z-score showing how positioned HLP is vs historical norms.
        Z-score of 2.2 = HLP is 2.2σ more long = retail heavily SHORT.
        """
        return self._get("/api/hlp/sentiment").json()

    def get_hlp_liquidator_status(self) -> Dict:
        """Get real-time HLP liquidator status (active/idle + PnL)."""
        return self._get("/api/hlp/liquidators/status").json()

    def get_hlp_market_maker(self) -> Dict:
        """Get HLP Strategy B market maker tracker for BTC/ETH/SOL."""
        return self._get("/api/hlp/market-maker").json()

    def get_hlp_timing(self) -> Dict:
        """Get HLP hourly/session profitability analysis."""
        return self._get("/api/hlp/timing").json()

    def get_hlp_correlation(self) -> Dict:
        """Get HLP delta-price correlation analysis by coin."""
        return self._get("/api/hlp/correlation").json()

    def get_hlp_delta(self) -> Dict:
        """Get live HLP net delta calculation (polls every 30s)."""
        return self._get("/api/hlp/delta").json()

    def get_hlp_flips(self) -> Any:
        """Get historical HLP flip events (when delta crosses zero)."""
        return self._get("/api/hlp/flips").json()

    def get_hlp_flip_stats(self) -> Dict:
        """Get aggregated HLP flip statistics."""
        return self._get("/api/hlp/flip-stats").json()

    # ==================== HIP3 MARKET DATA ====================

    def get_hip3_meta(self, include_delisted: bool = False) -> Dict:
        """Get all 51 HIP3 symbols from all 4 dexes with current prices."""
        params = "?include_delisted=true" if include_delisted else ""
        return self._get(f"/api/hip3/meta{params}").json()

    def get_hip3_tick_stats(self) -> Dict:
        """Get HIP3 tick collector statistics with dex breakdown."""
        return self._get("/api/hip3_ticks/stats.json").json()

    def get_hip3_ticks(self, dex: str, ticker: str) -> Any:
        """
        Get raw tick data for a specific HIP3 symbol.

        Args:
            dex: Dex prefix (xyz, flx, hyna, km)
            ticker: Symbol ticker (tsla, btc, gold, us500, etc.)
        """
        return self._get(f"/api/hip3_ticks/{dex.lower()}_{ticker.lower()}.json").json()


# ──────────────────────────────────────────────
# Module-level singleton factory
# ──────────────────────────────────────────────

_default_client: Optional[MoonDevAPI] = None


def get_moondev_client(**kwargs) -> MoonDevAPI:
    """Get or create the default MoonDevAPI singleton."""
    global _default_client
    if _default_client is None:
        _default_client = MoonDevAPI(**kwargs)
    return _default_client


# ──────────────────────────────────────────────
# Async wrappers for use in asyncio strategies
# ──────────────────────────────────────────────


async def async_get_prices(client: MoonDevAPI) -> Dict:
    return await asyncio.to_thread(client.get_prices)


async def async_get_price(client: MoonDevAPI, coin: str) -> Dict:
    return await asyncio.to_thread(client.get_price, coin)


async def async_get_orderbook(client: MoonDevAPI, coin: str) -> Dict:
    return await asyncio.to_thread(client.get_orderbook, coin)


async def async_get_candles(client: MoonDevAPI, coin: str, interval: str = "5m") -> Any:
    return await asyncio.to_thread(client.get_candles, coin, interval)


async def async_get_hlp_sentiment(client: MoonDevAPI) -> Dict:
    return await asyncio.to_thread(client.get_hlp_sentiment)


async def async_get_smart_money_signals(
    client: MoonDevAPI, timeframe: str = "1h"
) -> Dict:
    return await asyncio.to_thread(client.get_smart_money_signals, timeframe)


async def async_get_liquidations(client: MoonDevAPI, timeframe: str = "1h") -> Dict:
    return await asyncio.to_thread(client.get_liquidations, timeframe)


async def async_get_imbalance(client: MoonDevAPI, timeframe: str = "1h") -> Dict:
    return await asyncio.to_thread(client.get_imbalance, timeframe)


async def async_get_whales(client: MoonDevAPI) -> Dict:
    return await asyncio.to_thread(client.get_whales)


async def async_get_large_trades(client: MoonDevAPI) -> Any:
    return await asyncio.to_thread(client.get_large_trades)
