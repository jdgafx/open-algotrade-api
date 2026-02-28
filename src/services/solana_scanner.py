"""
Solana DEX Token Scanner — Full Solana new-token discovery and filtering service.

Ported from MoonDev's solana-sniper-2025 bot (get_new_tokens.py, nice_funcs.py, config.py)
and adapted for the Open Algotrade async API architecture.

Pipeline:
1. Fetch new tokens from Jupiter API (lite-api.jup.ag/tokens/v1/new)
2. Security check via BirdEye (freeze authority, mutable metadata, Token2022, top10 holders)
3. Metrics filter via BirdEye token_overview (market cap, liquidity, trades, wallets, sell %)
4. OHLCV technical analysis via BirdEye (MA20/MA40, higher highs/lows, price vs avg)
5. Score and rank surviving tokens

Paper mode: fetches real data from Jupiter/BirdEye, but simulates trades locally.
No wallet or private key needed for scanning or paper trading.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import httpx
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration (env-var overridable)
# ---------------------------------------------------------------------------

class ScannerConfig:
    """All scanner params, populated from env vars with sensible defaults
    matching MoonDev's original config.py values."""

    def __init__(self, overrides: Optional[Dict[str, Any]] = None):
        o = overrides or {}

        # API keys
        self.birdeye_api_key: str = o.get("birdeye_api_key", os.getenv("BIRDEYE_API_KEY", ""))
        self.solana_rpc_url: str = o.get("solana_rpc_url", os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"))
        self.scanner_enabled: bool = o.get("scanner_enabled", os.getenv("SOLANA_SCANNER_ENABLED", "true").lower() == "true")

        # Token discovery
        self.hours_lookback: float = o.get("hours_lookback", float(os.getenv("SOLANA_HOURS_LOOKBACK", "0.5")))

        # Security filters
        self.max_top10_holder_pct: float = o.get("max_top10_holder_pct", float(os.getenv("SOLANA_MAX_TOP10_HOLDER_PCT", "0.7")))
        self.drop_if_mutable_metadata: bool = o.get("drop_if_mutable_metadata", os.getenv("SOLANA_DROP_MUTABLE_METADATA", "false").lower() == "true")
        self.drop_if_token_2022: bool = o.get("drop_if_token_2022", os.getenv("SOLANA_DROP_TOKEN_2022", "true").lower() == "true")

        # Metrics filters
        self.max_market_cap: float = o.get("max_market_cap", float(os.getenv("SOLANA_MAX_MARKET_CAP", "30000")))
        self.min_market_cap: float = o.get("min_market_cap", float(os.getenv("SOLANA_MIN_MARKET_CAP", "50")))
        self.min_liquidity: float = o.get("min_liquidity", float(os.getenv("SOLANA_MIN_LIQUIDITY", "400")))
        self.min_trades_1h: int = o.get("min_trades_1h", int(os.getenv("SOLANA_MIN_TRADES_1H", "9")))
        self.min_unique_wallets_24h: int = o.get("min_unique_wallets_24h", int(os.getenv("SOLANA_MIN_UNIQUE_WALLETS_24H", "30")))
        self.max_sell_pct: float = o.get("max_sell_pct", float(os.getenv("SOLANA_MAX_SELL_PCT", "70")))

        # OHLCV / Technical filters
        self.ohlcv_timeframe: str = o.get("ohlcv_timeframe", os.getenv("SOLANA_OHLCV_TIMEFRAME", "3m"))
        self.max_bars_before_drop: int = o.get("max_bars_before_drop", int(os.getenv("SOLANA_MAX_BARS", "120")))
        self.require_above_avg_close: bool = o.get("require_above_avg_close", os.getenv("SOLANA_REQUIRE_ABOVE_AVG", "true").lower() == "true")
        self.sma_fast: int = o.get("sma_fast", int(os.getenv("SOLANA_SMA_FAST", "20")))
        self.sma_slow: int = o.get("sma_slow", int(os.getenv("SOLANA_SMA_SLOW", "40")))

        # Paper trading
        self.paper_balance: float = o.get("paper_balance", float(os.getenv("SOLANA_PAPER_BALANCE", "1000")))
        self.position_size_usdc: float = o.get("position_size_usdc", float(os.getenv("SOLANA_POSITION_SIZE", "10")))
        self.sell_at_multiple: float = o.get("sell_at_multiple", float(os.getenv("SOLANA_SELL_AT_MULTIPLE", "9.0")))
        self.stop_loss_pct: float = o.get("stop_loss_pct", float(os.getenv("SOLANA_STOP_LOSS_PCT", "-0.6")))

        # Scanning interval
        self.scan_interval_seconds: int = o.get("scan_interval_seconds", int(os.getenv("SOLANA_SCAN_INTERVAL", "600")))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scanner_enabled": self.scanner_enabled,
            "birdeye_api_key_set": bool(self.birdeye_api_key),
            "solana_rpc_url": self.solana_rpc_url,
            "hours_lookback": self.hours_lookback,
            "max_top10_holder_pct": self.max_top10_holder_pct,
            "drop_if_mutable_metadata": self.drop_if_mutable_metadata,
            "drop_if_token_2022": self.drop_if_token_2022,
            "max_market_cap": self.max_market_cap,
            "min_market_cap": self.min_market_cap,
            "min_liquidity": self.min_liquidity,
            "min_trades_1h": self.min_trades_1h,
            "min_unique_wallets_24h": self.min_unique_wallets_24h,
            "max_sell_pct": self.max_sell_pct,
            "ohlcv_timeframe": self.ohlcv_timeframe,
            "max_bars_before_drop": self.max_bars_before_drop,
            "require_above_avg_close": self.require_above_avg_close,
            "sma_fast": self.sma_fast,
            "sma_slow": self.sma_slow,
            "paper_balance": self.paper_balance,
            "position_size_usdc": self.position_size_usdc,
            "sell_at_multiple": self.sell_at_multiple,
            "stop_loss_pct": self.stop_loss_pct,
            "scan_interval_seconds": self.scan_interval_seconds,
        }


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class TokenStatus(str, Enum):
    DISCOVERED = "discovered"
    SECURITY_PASSED = "security_passed"
    METRICS_PASSED = "metrics_passed"
    OHLCV_PASSED = "ohlcv_passed"
    REJECTED_SECURITY = "rejected_security"
    REJECTED_METRICS = "rejected_metrics"
    REJECTED_OHLCV = "rejected_ohlcv"
    BOUGHT = "bought"
    SOLD_TP = "sold_tp"
    SOLD_SL = "sold_sl"
    BLACKLISTED = "blacklisted"


@dataclass
class ScannedToken:
    """A token that has been discovered and is progressing through the filter pipeline."""
    address: str
    name: str = ""
    symbol: str = ""
    discovered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: TokenStatus = TokenStatus.DISCOVERED
    rejection_reason: str = ""

    # Security data (from BirdEye)
    freeze_authority: Optional[bool] = None
    mutable_metadata: Optional[bool] = None
    is_token_2022: Optional[bool] = None
    top10_holder_pct: Optional[float] = None

    # Metrics data (from BirdEye token_overview)
    market_cap: Optional[float] = None
    liquidity: Optional[float] = None
    buy_1h: Optional[int] = None
    sell_1h: Optional[int] = None
    trade_1h: Optional[int] = None
    buy_pct: Optional[float] = None
    sell_pct: Optional[float] = None
    unique_wallets_24h: Optional[int] = None

    # OHLCV / Technical data
    price: Optional[float] = None
    ma_fast: Optional[float] = None
    ma_slow: Optional[float] = None
    price_above_ma_fast: Optional[bool] = None
    price_above_ma_slow: Optional[bool] = None
    ma_fast_above_slow: Optional[bool] = None
    higher_highs: Optional[bool] = None
    higher_lows: Optional[bool] = None
    price_above_avg_close: Optional[bool] = None
    num_bars: Optional[int] = None

    # Composite score (0-100)
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "address": self.address,
            "name": self.name,
            "symbol": self.symbol,
            "discovered_at": self.discovered_at.isoformat(),
            "status": self.status.value,
            "rejection_reason": self.rejection_reason,
            "freeze_authority": self.freeze_authority,
            "mutable_metadata": self.mutable_metadata,
            "is_token_2022": self.is_token_2022,
            "top10_holder_pct": self.top10_holder_pct,
            "market_cap": self.market_cap,
            "liquidity": self.liquidity,
            "buy_1h": self.buy_1h,
            "sell_1h": self.sell_1h,
            "trade_1h": self.trade_1h,
            "buy_pct": self.buy_pct,
            "sell_pct": self.sell_pct,
            "unique_wallets_24h": self.unique_wallets_24h,
            "price": self.price,
            "ma_fast": self.ma_fast,
            "ma_slow": self.ma_slow,
            "price_above_ma_fast": self.price_above_ma_fast,
            "price_above_ma_slow": self.price_above_ma_slow,
            "ma_fast_above_slow": self.ma_fast_above_slow,
            "higher_highs": self.higher_highs,
            "higher_lows": self.higher_lows,
            "price_above_avg_close": self.price_above_avg_close,
            "num_bars": self.num_bars,
            "score": round(self.score, 2),
        }


@dataclass
class SolanaPaperPosition:
    """A simulated position in a Solana token."""
    token_address: str
    token_name: str
    token_symbol: str
    side: str = "long"
    size_usdc: float = 0.0
    token_amount: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    pnl_pct: float = 0.0
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    score_at_entry: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_address": self.token_address,
            "token_name": self.token_name,
            "token_symbol": self.token_symbol,
            "side": self.side,
            "size_usdc": round(self.size_usdc, 2),
            "token_amount": self.token_amount,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "unrealized_pnl": round(self.unrealized_pnl, 4),
            "pnl_pct": round(self.pnl_pct, 2),
            "entry_time": self.entry_time.isoformat(),
            "score_at_entry": round(self.score_at_entry, 2),
        }


@dataclass
class SolanaPaperTrade:
    """A completed paper trade."""
    id: int
    token_address: str
    token_name: str
    action: str  # "buy" or "sell"
    price: float
    size_usdc: float
    token_amount: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    reason: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "token_address": self.token_address,
            "token_name": self.token_name,
            "action": self.action,
            "price": self.price,
            "size_usdc": round(self.size_usdc, 2),
            "token_amount": self.token_amount,
            "pnl": round(self.pnl, 4),
            "pnl_pct": round(self.pnl_pct, 2),
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }


# ---------------------------------------------------------------------------
# The Scanner Service
# ---------------------------------------------------------------------------

class SolanaScanner:
    """
    Async Solana DEX token scanner.

    Fetches new tokens from Jupiter, filters via BirdEye security + metrics + OHLCV,
    and optionally paper-trades the survivors.
    """

    def __init__(self, config: Optional[ScannerConfig] = None):
        self.config = config or ScannerConfig()
        self._http: Optional[httpx.AsyncClient] = None

        # State
        self._all_tokens: Dict[str, ScannedToken] = {}  # address -> token
        self._blacklist: Set[str] = set()
        self._positions: Dict[str, SolanaPaperPosition] = {}  # address -> position
        self._trades: List[SolanaPaperTrade] = []
        self._trade_counter: int = 0
        self._paper_balance: float = self.config.paper_balance
        self._initial_balance: float = self.config.paper_balance

        # Scan state
        self._last_scan_time: Optional[datetime] = None
        self._total_scans: int = 0
        self._total_tokens_seen: int = 0
        self._total_tokens_passed: int = 0
        self._scanning: bool = False
        self._scan_task: Optional[asyncio.Task] = None

        logger.info(
            "SolanaScanner initialized | enabled=%s | birdeye_key=%s | balance=$%.2f",
            self.config.scanner_enabled,
            "SET" if self.config.birdeye_api_key else "NOT SET",
            self._paper_balance,
        )

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=15.0)
        return self._http

    async def close(self):
        if self._http and not self._http.is_closed:
            await self._http.aclose()
        if self._scan_task and not self._scan_task.done():
            self._scan_task.cancel()

    # -------------------------------------------------------------------
    # Token Discovery: DexScreener + Jupiter fallback
    # -------------------------------------------------------------------

    async def fetch_new_tokens(self) -> List[Dict[str, Any]]:
        """Fetch newly listed Solana tokens from multiple sources.

        Sources (tried in order):
        1. DexScreener token-profiles/latest (free, no key, returns recent tokens)
        2. Jupiter lite-api.jup.ag/tokens/v1/new (original MoonDev source)
        3. BirdEye tokenlist sorted by creation_time (requires API key)

        Returns a unified list of dicts with: mint, name, symbol fields.
        """
        # Try DexScreener first (most reliable for new token discovery)
        tokens = await self._fetch_dexscreener_latest()
        if tokens:
            return tokens

        # Fallback: Jupiter new tokens endpoint
        tokens = await self._fetch_jupiter_new()
        if tokens:
            return tokens

        # Fallback: BirdEye token list (requires API key)
        tokens = await self._fetch_birdeye_new()
        if tokens:
            return tokens

        logger.warning("All token discovery sources returned empty")
        return []

    async def _fetch_dexscreener_latest(self) -> List[Dict[str, Any]]:
        """Fetch latest token profiles from DexScreener (free, no API key).

        Returns Solana tokens from the latest profiles endpoint.
        """
        http = await self._get_http()
        url = "https://api.dexscreener.com/token-profiles/latest/v1"

        try:
            resp = await http.get(url)
            resp.raise_for_status()
            data = resp.json()

            if not isinstance(data, list):
                return []

            # Filter for Solana tokens only
            solana_tokens = []
            seen = set()
            for item in data:
                if item.get("chainId") != "solana":
                    continue
                addr = item.get("tokenAddress", "")
                if not addr or addr in seen:
                    continue
                seen.add(addr)

                solana_tokens.append({
                    "mint": addr,
                    "name": item.get("description", "")[:50] if item.get("description") else "",
                    "symbol": "",
                    "source": "dexscreener",
                })

            logger.info("DexScreener: found %d Solana tokens from latest profiles", len(solana_tokens))
            return solana_tokens

        except Exception as e:
            logger.debug("DexScreener fetch error: %s", e)
            return []

    async def _fetch_jupiter_new(self) -> List[Dict[str, Any]]:
        """Fetch new tokens from Jupiter API (original MoonDev source).

        The lite-api.jup.ag/tokens/v1/new endpoint. Returns tokens with
        mint, name, symbol, created_at fields.
        """
        http = await self._get_http()
        url = "https://lite-api.jup.ag/tokens/v1/new"

        try:
            resp = await http.get(url)
            if resp.status_code != 200:
                logger.debug("Jupiter /tokens/v1/new returned %d", resp.status_code)
                return []

            tokens = resp.json()
            if not isinstance(tokens, list):
                return []

            # Filter by time window
            cutoff = datetime.now(timezone.utc) - timedelta(hours=self.config.hours_lookback)
            filtered = []
            for t in tokens:
                created_at = t.get("created_at")
                if created_at is not None:
                    try:
                        token_time = datetime.fromtimestamp(int(created_at), tz=timezone.utc)
                        if token_time >= cutoff:
                            t["source"] = "jupiter"
                            filtered.append(t)
                    except (ValueError, TypeError):
                        pass

            logger.info(
                "Jupiter: fetched %d total new tokens, %d within %.1f hour window",
                len(tokens), len(filtered), self.config.hours_lookback,
            )
            return filtered

        except Exception as e:
            logger.debug("Jupiter fetch error: %s", e)
            return []

    async def _fetch_birdeye_new(self) -> List[Dict[str, Any]]:
        """Fetch new tokens from BirdEye token list sorted by creation time.

        Requires a BirdEye API key.
        """
        if not self.config.birdeye_api_key:
            return []

        http = await self._get_http()
        url = (
            "https://public-api.birdeye.so/defi/tokenlist"
            "?sort_by=creation_time&sort_type=desc&offset=0&limit=50"
            "&min_liquidity=100"
        )
        headers = {"X-API-KEY": self.config.birdeye_api_key, "x-chain": "solana"}

        try:
            resp = await http.get(url, headers=headers)
            if resp.status_code != 200:
                logger.debug("BirdEye tokenlist returned %d", resp.status_code)
                return []

            data = resp.json()
            tokens_data = data.get("data", {}).get("tokens", [])
            if not tokens_data:
                return []

            result = []
            for t in tokens_data:
                result.append({
                    "mint": t.get("address", ""),
                    "name": t.get("name", ""),
                    "symbol": t.get("symbol", ""),
                    "source": "birdeye",
                })

            logger.info("BirdEye: fetched %d new tokens from tokenlist", len(result))
            return result

        except Exception as e:
            logger.debug("BirdEye tokenlist error: %s", e)
            return []

    # -------------------------------------------------------------------
    # BirdEye: Security check
    # -------------------------------------------------------------------

    async def security_check(self, address: str) -> Optional[Dict[str, Any]]:
        """Run BirdEye security check on a token.

        Returns security data dict or None if check fails.
        When no BirdEye API key is set, returns a simulated pass (paper mode).
        """
        if not self.config.birdeye_api_key:
            # Paper mode without BirdEye: simulate a pass with neutral values
            logger.debug("No BirdEye key: simulating security pass for %s", address[:8])
            return {
                "freezeAuthority": None,
                "mutableMetadata": False,
                "isToken2022": False,
                "top10HolderPercent": 0.3,
                "simulated": True,
            }

        http = await self._get_http()
        url = f"https://public-api.birdeye.so/defi/token_security?address={address}"
        headers = {"X-API-KEY": self.config.birdeye_api_key}

        try:
            resp = await http.get(url, headers=headers)

            if resp.status_code == 521:
                logger.warning("BirdEye 521 for %s (Cloudflare) — will retry", address[:8])
                return None

            if resp.status_code != 200:
                logger.warning("BirdEye security %d for %s", resp.status_code, address[:8])
                return None

            data = resp.json()
            if not data.get("success") or "data" not in data:
                return None

            security = data["data"]

            # Original MoonDev check: if freezeable, drop immediately
            if security.get("freezeable", False):
                logger.debug("Token %s is freezeable — dropping", address[:8])
                return None

            return security

        except Exception as e:
            logger.error("BirdEye security error for %s: %s", address[:8], e)
            return None

    def _apply_security_filters(self, token: ScannedToken, security: Dict[str, Any]) -> bool:
        """Apply security filters. Returns True if token passes."""
        token.freeze_authority = bool(security.get("freezeAuthority"))
        token.mutable_metadata = bool(security.get("mutableMetadata"))
        token.is_token_2022 = bool(security.get("isToken2022"))
        token.top10_holder_pct = security.get("top10HolderPercent", 0)

        if token.freeze_authority:
            token.status = TokenStatus.REJECTED_SECURITY
            token.rejection_reason = "freeze_authority"
            return False

        if token.top10_holder_pct is not None and token.top10_holder_pct > self.config.max_top10_holder_pct:
            token.status = TokenStatus.REJECTED_SECURITY
            token.rejection_reason = f"top10_holder_pct={token.top10_holder_pct:.1%} > {self.config.max_top10_holder_pct:.1%}"
            return False

        if self.config.drop_if_mutable_metadata and token.mutable_metadata:
            token.status = TokenStatus.REJECTED_SECURITY
            token.rejection_reason = "mutable_metadata"
            return False

        if self.config.drop_if_token_2022 and token.is_token_2022:
            token.status = TokenStatus.REJECTED_SECURITY
            token.rejection_reason = "token_2022_program"
            return False

        token.status = TokenStatus.SECURITY_PASSED
        return True

    # -------------------------------------------------------------------
    # BirdEye: Token overview / metrics
    # -------------------------------------------------------------------

    async def token_overview(self, address: str) -> Optional[Dict[str, Any]]:
        """Fetch token overview metrics from BirdEye.

        Returns overview dict or None on failure.
        When no BirdEye API key is set, returns simulated data.
        """
        if not self.config.birdeye_api_key:
            # Simulate metrics in paper mode
            logger.debug("No BirdEye key: simulating metrics for %s", address[:8])
            return {
                "mc": 5000.0,
                "liquidity": 800.0,
                "buy1h": 25,
                "sell1h": 10,
                "uniqueWallet24h": 50,
                "top10HolderPercent": 0.3,
                "simulated": True,
            }

        http = await self._get_http()
        url = f"https://public-api.birdeye.so/defi/token_overview?address={address}"
        headers = {"X-API-KEY": self.config.birdeye_api_key}

        try:
            resp = await http.get(url, headers=headers)

            if resp.status_code == 521:
                logger.warning("BirdEye 521 for overview %s", address[:8])
                return None

            if resp.status_code != 200:
                logger.warning("BirdEye overview %d for %s", resp.status_code, address[:8])
                return None

            data = resp.json()
            return data.get("data", {})

        except Exception as e:
            logger.error("BirdEye overview error for %s: %s", address[:8], e)
            return None

    def _apply_metrics_filters(self, token: ScannedToken, overview: Dict[str, Any]) -> bool:
        """Apply metrics filters from BirdEye token_overview. Returns True if passes."""
        mc = overview.get("mc", 0) or 0
        liquidity = overview.get("liquidity", 0) or 0
        buy1h = overview.get("buy1h", 0) or 0
        sell1h = overview.get("sell1h", 0) or 0
        trade1h = buy1h + sell1h
        unique_wallets = overview.get("uniqueWallet24h", 0) or 0

        token.market_cap = mc
        token.liquidity = liquidity
        token.buy_1h = buy1h
        token.sell_1h = sell1h
        token.trade_1h = trade1h
        token.unique_wallets_24h = unique_wallets

        if trade1h > 0:
            token.buy_pct = (buy1h / trade1h) * 100
            token.sell_pct = (sell1h / trade1h) * 100
        else:
            token.buy_pct = 0
            token.sell_pct = 0

        # Market cap too high
        if mc > self.config.max_market_cap:
            token.status = TokenStatus.REJECTED_METRICS
            token.rejection_reason = f"market_cap=${mc:,.0f} > ${self.config.max_market_cap:,.0f}"
            return False

        # Market cap too low
        if mc < self.config.min_market_cap and mc > 0:
            token.status = TokenStatus.REJECTED_METRICS
            token.rejection_reason = f"market_cap=${mc:,.0f} < ${self.config.min_market_cap:,.0f}"
            return False

        # Top 10 holder check (also in overview)
        top10 = overview.get("top10HolderPercent", 0) or 0
        if top10 > self.config.max_top10_holder_pct:
            token.status = TokenStatus.REJECTED_METRICS
            token.rejection_reason = f"top10_holders={top10:.1%} (from overview)"
            return False

        # Sell pressure too high
        if token.sell_pct and token.sell_pct > self.config.max_sell_pct:
            token.status = TokenStatus.REJECTED_METRICS
            token.rejection_reason = f"sell_pct={token.sell_pct:.1f}% > {self.config.max_sell_pct}%"
            return False

        # Not enough trades
        if trade1h < self.config.min_trades_1h:
            token.status = TokenStatus.REJECTED_METRICS
            token.rejection_reason = f"trades_1h={trade1h} < {self.config.min_trades_1h}"
            return False

        # Not enough unique wallets
        if unique_wallets < self.config.min_unique_wallets_24h:
            token.status = TokenStatus.REJECTED_METRICS
            token.rejection_reason = f"wallets_24h={unique_wallets} < {self.config.min_unique_wallets_24h}"
            return False

        # Insufficient liquidity
        if liquidity < self.config.min_liquidity:
            token.status = TokenStatus.REJECTED_METRICS
            token.rejection_reason = f"liquidity=${liquidity:,.0f} < ${self.config.min_liquidity:,.0f}"
            return False

        token.status = TokenStatus.METRICS_PASSED
        return True

    # -------------------------------------------------------------------
    # BirdEye: OHLCV + Technical Analysis
    # -------------------------------------------------------------------

    async def fetch_ohlcv(self, address: str) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from BirdEye for a token.

        Returns DataFrame with Open, High, Low, Close, Volume columns or None.
        When no BirdEye API key is set, returns None (OHLCV filter skipped).
        """
        if not self.config.birdeye_api_key:
            return None

        http = await self._get_http()
        now = datetime.now(timezone.utc)
        time_from = int((now - timedelta(days=10)).timestamp())
        time_to = int(now.timestamp())

        url = (
            f"https://public-api.birdeye.so/defi/ohlcv"
            f"?address={address}&type={self.config.ohlcv_timeframe}"
            f"&time_from={time_from}&time_to={time_to}"
        )
        headers = {"X-API-KEY": self.config.birdeye_api_key}

        try:
            resp = await http.get(url, headers=headers)
            if resp.status_code != 200:
                logger.debug("BirdEye OHLCV %d for %s", resp.status_code, address[:8])
                return None

            data = resp.json()
            items = data.get("data", {}).get("items", [])
            if not items:
                return None

            rows = [{
                "datetime": datetime.utcfromtimestamp(item["unixTime"]).strftime("%Y-%m-%d %H:%M:%S"),
                "open": item["o"],
                "high": item["h"],
                "low": item["l"],
                "close": item["c"],
                "volume": item["v"],
            } for item in items]

            df = pd.DataFrame(rows)

            # Pad if too few rows (same as original MoonDev code)
            if len(df) < 40 and len(df) > 0:
                rows_to_add = 40 - len(df)
                first_row = pd.concat([df.iloc[0:1]] * rows_to_add, ignore_index=True)
                df = pd.concat([first_row, df], ignore_index=True)

            # Technical indicators
            fast = self.config.sma_fast
            slow = self.config.sma_slow
            df["ma_fast"] = df["close"].rolling(window=fast).mean()
            df["ma_slow"] = df["close"].rolling(window=slow).mean()
            df["price_above_ma_fast"] = df["close"] > df["ma_fast"]
            df["price_above_ma_slow"] = df["close"] > df["ma_slow"]
            df["ma_fast_above_slow"] = df["ma_fast"] > df["ma_slow"]

            return df

        except Exception as e:
            logger.error("BirdEye OHLCV error for %s: %s", address[:8], e)
            return None

    def _apply_ohlcv_filters(self, token: ScannedToken, df: Optional[pd.DataFrame]) -> bool:
        """Apply OHLCV / technical analysis filters. Returns True if passes.

        If df is None (no BirdEye key), auto-passes with simulated values.
        """
        if df is None or df.empty:
            # No OHLCV data available — pass through (e.g. no BirdEye key)
            token.status = TokenStatus.OHLCV_PASSED
            token.num_bars = 0
            return True

        num_bars = len(df)
        token.num_bars = num_bars

        # Too many bars = token is too old
        if num_bars > self.config.max_bars_before_drop:
            token.status = TokenStatus.REJECTED_OHLCV
            token.rejection_reason = f"bars={num_bars} > {self.config.max_bars_before_drop}"
            return False

        # Latest values
        last = df.iloc[-1]
        token.price = last["close"]
        token.ma_fast = last.get("ma_fast")
        token.ma_slow = last.get("ma_slow")
        token.price_above_ma_fast = bool(last.get("price_above_ma_fast", False))
        token.price_above_ma_slow = bool(last.get("price_above_ma_slow", False))
        token.ma_fast_above_slow = bool(last.get("ma_fast_above_slow", False))

        # Price vs average close
        avg_close = df["close"].mean()
        token.price_above_avg_close = token.price > avg_close

        if self.config.require_above_avg_close and not token.price_above_avg_close:
            token.status = TokenStatus.REJECTED_OHLCV
            token.rejection_reason = "price_below_avg_close"
            return False

        # Higher highs / higher lows
        token.higher_highs = True
        token.higher_lows = True
        prev_high = df["high"].iloc[0]
        prev_low = df["low"].iloc[0]
        for i in range(1, len(df)):
            if df["high"].iloc[i] <= prev_high:
                token.higher_highs = False
            if df["low"].iloc[i] <= prev_low:
                token.higher_lows = False
            prev_high = df["high"].iloc[i]
            prev_low = df["low"].iloc[i]

        # MA conditions over last 30 bars (same as original)
        if len(df) >= 30:
            tail = df.tail(30)
            cond_a = (tail["price_above_ma_fast"].sum() / 30) > 0.5
            cond_b = (tail["price_above_ma_slow"].sum() / 30) > 0.5
            cond_c = (tail["ma_fast_above_slow"].sum() / 30) > 0.5
            cond_d = df["close"].iloc[-1] > df["open"].iloc[0]  # price increase from launch

            if any([cond_a, cond_b, cond_c, cond_d]):
                token.status = TokenStatus.OHLCV_PASSED
                return True

        # Less than 30 bars: simpler check
        if token.price_above_ma_fast or token.price_above_ma_slow or token.ma_fast_above_slow:
            token.status = TokenStatus.OHLCV_PASSED
            return True

        token.status = TokenStatus.REJECTED_OHLCV
        token.rejection_reason = "failed_technical_conditions"
        return False

    # -------------------------------------------------------------------
    # BirdEye: Get current price for a token
    # -------------------------------------------------------------------

    async def get_token_price(self, address: str) -> Optional[float]:
        """Fetch current price from BirdEye. Returns None on failure."""
        if not self.config.birdeye_api_key:
            return None

        http = await self._get_http()
        url = f"https://public-api.birdeye.so/defi/price?address={address}"
        headers = {"X-API-KEY": self.config.birdeye_api_key}

        try:
            resp = await http.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                value = data.get("data", {}).get("value")
                return float(value) if value is not None else None
        except Exception as e:
            logger.debug("Price fetch error for %s: %s", address[:8], e)
        return None

    # -------------------------------------------------------------------
    # Scoring
    # -------------------------------------------------------------------

    def _compute_score(self, token: ScannedToken) -> float:
        """Compute a 0-100 composite score for a token that passed all filters."""
        score = 50.0  # base score for passing all filters

        # Security bonus (lower top10 holder = better)
        if token.top10_holder_pct is not None:
            holder_bonus = max(0, (self.config.max_top10_holder_pct - token.top10_holder_pct)) * 20
            score += holder_bonus

        # Liquidity bonus (higher liquidity = better, capped)
        if token.liquidity:
            liq_bonus = min(token.liquidity / 5000.0, 1.0) * 10
            score += liq_bonus

        # Trade activity bonus
        if token.trade_1h:
            trade_bonus = min(token.trade_1h / 100.0, 1.0) * 5
            score += trade_bonus

        # Buy pressure bonus
        if token.buy_pct and token.buy_pct > 50:
            score += min((token.buy_pct - 50) / 50 * 10, 10)

        # Unique wallets bonus
        if token.unique_wallets_24h:
            wallet_bonus = min(token.unique_wallets_24h / 200.0, 1.0) * 5
            score += wallet_bonus

        # Technical analysis bonus
        ta_points = 0
        if token.price_above_ma_fast:
            ta_points += 3
        if token.price_above_ma_slow:
            ta_points += 3
        if token.ma_fast_above_slow:
            ta_points += 3
        if token.higher_highs:
            ta_points += 3
        if token.higher_lows:
            ta_points += 3
        if token.price_above_avg_close:
            ta_points += 2
        score += ta_points

        return min(score, 100.0)

    # -------------------------------------------------------------------
    # Full Scan Pipeline
    # -------------------------------------------------------------------

    async def run_scan(self) -> Dict[str, Any]:
        """Run the full scan pipeline.

        Returns a summary dict with counts and the list of tokens that passed.
        """
        if self._scanning:
            return {"status": "already_scanning", "message": "A scan is already in progress."}

        self._scanning = True
        scan_start = time.monotonic()

        try:
            # Step 1: Fetch new tokens from Jupiter
            raw_tokens = await self.fetch_new_tokens()
            if not raw_tokens:
                return {
                    "status": "complete",
                    "tokens_fetched": 0,
                    "tokens_passed": 0,
                    "passed_tokens": [],
                    "scan_time_seconds": round(time.monotonic() - scan_start, 2),
                }

            # Remove already-blacklisted tokens
            raw_tokens = [t for t in raw_tokens if t.get("mint") not in self._blacklist]

            discovered = []
            for t in raw_tokens:
                mint = t.get("mint", "")
                if not mint:
                    continue

                token = ScannedToken(
                    address=mint,
                    name=t.get("name", ""),
                    symbol=t.get("symbol", ""),
                )
                discovered.append(token)
                self._all_tokens[mint] = token

            self._total_tokens_seen += len(discovered)

            # Step 2: Security checks
            security_passed = []
            for token in discovered:
                security_data = await self.security_check(token.address)
                if security_data is None:
                    token.status = TokenStatus.REJECTED_SECURITY
                    token.rejection_reason = "security_check_failed"
                    self._blacklist.add(token.address)
                    continue

                if self._apply_security_filters(token, security_data):
                    security_passed.append(token)
                else:
                    self._blacklist.add(token.address)

                # Rate limiting for BirdEye
                if self.config.birdeye_api_key:
                    await asyncio.sleep(0.5)

            # Step 3: Metrics checks
            metrics_passed = []
            for token in security_passed:
                overview = await self.token_overview(token.address)
                if overview is None:
                    continue

                if self._apply_metrics_filters(token, overview):
                    metrics_passed.append(token)

                if self.config.birdeye_api_key:
                    await asyncio.sleep(0.5)

            # Step 4: OHLCV / Technical analysis
            final_passed = []
            for token in metrics_passed:
                ohlcv_df = await self.fetch_ohlcv(token.address)
                if self._apply_ohlcv_filters(token, ohlcv_df):
                    token.score = self._compute_score(token)
                    final_passed.append(token)

                if self.config.birdeye_api_key:
                    await asyncio.sleep(0.5)

            # Sort by score descending
            final_passed.sort(key=lambda t: t.score, reverse=True)

            self._total_tokens_passed += len(final_passed)
            self._total_scans += 1
            self._last_scan_time = datetime.now(timezone.utc)

            scan_time = round(time.monotonic() - scan_start, 2)

            logger.info(
                "Scan complete: %d fetched -> %d security -> %d metrics -> %d final (%.1fs)",
                len(discovered), len(security_passed), len(metrics_passed),
                len(final_passed), scan_time,
            )

            return {
                "status": "complete",
                "tokens_fetched": len(discovered),
                "security_passed": len(security_passed),
                "metrics_passed": len(metrics_passed),
                "tokens_passed": len(final_passed),
                "passed_tokens": [t.to_dict() for t in final_passed],
                "scan_time_seconds": scan_time,
            }

        except Exception as e:
            logger.error("Scan error: %s", e, exc_info=True)
            return {"status": "error", "error": str(e)}
        finally:
            self._scanning = False

    # -------------------------------------------------------------------
    # Paper Trading
    # -------------------------------------------------------------------

    async def paper_buy(self, token_address: str) -> Dict[str, Any]:
        """Simulate buying a token in paper mode."""
        if token_address in self._positions:
            return {"status": "error", "message": f"Already holding position in {token_address[:8]}"}

        token = self._all_tokens.get(token_address)
        if not token:
            return {"status": "error", "message": f"Token {token_address[:8]} not found in scanned tokens"}

        size = self.config.position_size_usdc
        if size > self._paper_balance:
            return {"status": "error", "message": f"Insufficient balance: ${self._paper_balance:.2f} < ${size:.2f}"}

        # Get price (from token data or BirdEye)
        price = token.price
        if price is None or price <= 0:
            price = await self.get_token_price(token_address)
        if price is None or price <= 0:
            # Use a simulated price for paper mode
            price = 0.001
            logger.debug("Using simulated price for %s", token_address[:8])

        token_amount = size / price
        self._paper_balance -= size

        pos = SolanaPaperPosition(
            token_address=token_address,
            token_name=token.name,
            token_symbol=token.symbol,
            size_usdc=size,
            token_amount=token_amount,
            entry_price=price,
            current_price=price,
            score_at_entry=token.score,
        )
        self._positions[token_address] = pos

        self._trade_counter += 1
        trade = SolanaPaperTrade(
            id=self._trade_counter,
            token_address=token_address,
            token_name=token.name,
            action="buy",
            price=price,
            size_usdc=size,
            token_amount=token_amount,
            reason=f"Scanner score={token.score:.1f}",
        )
        self._trades.append(trade)

        token.status = TokenStatus.BOUGHT

        logger.info(
            "[SOL PAPER] BUY | %s (%s) | %.6f tokens @ $%.8f ($%.2f) | balance=$%.2f",
            token.symbol, token_address[:8], token_amount, price, size, self._paper_balance,
        )

        return {
            "status": "bought",
            "position": pos.to_dict(),
            "balance": round(self._paper_balance, 2),
        }

    async def paper_sell(self, token_address: str, reason: str = "manual") -> Dict[str, Any]:
        """Simulate selling a token position in paper mode."""
        pos = self._positions.get(token_address)
        if not pos:
            return {"status": "error", "message": f"No position for {token_address[:8]}"}

        # Get current price
        current_price = await self.get_token_price(token_address)
        if current_price is None or current_price <= 0:
            current_price = pos.entry_price  # fallback

        exit_value = pos.token_amount * current_price
        pnl = exit_value - pos.size_usdc
        pnl_pct = (pnl / pos.size_usdc) * 100 if pos.size_usdc > 0 else 0

        self._paper_balance += exit_value

        self._trade_counter += 1
        trade = SolanaPaperTrade(
            id=self._trade_counter,
            token_address=token_address,
            token_name=pos.token_name,
            action="sell",
            price=current_price,
            size_usdc=exit_value,
            token_amount=pos.token_amount,
            pnl=pnl,
            pnl_pct=pnl_pct,
            reason=reason,
        )
        self._trades.append(trade)

        token = self._all_tokens.get(token_address)
        if token:
            token.status = TokenStatus.SOLD_TP if pnl >= 0 else TokenStatus.SOLD_SL

        del self._positions[token_address]

        logger.info(
            "[SOL PAPER] SELL | %s (%s) | pnl=$%.4f (%.2f%%) | reason=%s | balance=$%.2f",
            pos.token_symbol, token_address[:8], pnl, pnl_pct, reason, self._paper_balance,
        )

        return {
            "status": "sold",
            "pnl": round(pnl, 4),
            "pnl_pct": round(pnl_pct, 2),
            "balance": round(self._paper_balance, 2),
        }

    async def update_positions(self) -> List[Dict[str, Any]]:
        """Update all open position prices and check TP/SL."""
        results = []
        addresses_to_sell = []

        for address, pos in self._positions.items():
            price = await self.get_token_price(address)
            if price is not None and price > 0:
                pos.current_price = price
                pos.unrealized_pnl = (pos.token_amount * price) - pos.size_usdc
                pos.pnl_pct = (pos.unrealized_pnl / pos.size_usdc) * 100 if pos.size_usdc > 0 else 0

                # Check take profit
                multiple = price / pos.entry_price if pos.entry_price > 0 else 1
                if multiple >= self.config.sell_at_multiple:
                    addresses_to_sell.append((address, f"TP: {multiple:.1f}x >= {self.config.sell_at_multiple}x"))
                    continue

                # Check stop loss
                if pos.pnl_pct <= self.config.stop_loss_pct * 100:
                    addresses_to_sell.append((address, f"SL: {pos.pnl_pct:.1f}% <= {self.config.stop_loss_pct * 100:.0f}%"))

        # Execute sells outside the iteration
        for address, reason in addresses_to_sell:
            result = await self.paper_sell(address, reason=reason)
            results.append(result)

        return results

    # -------------------------------------------------------------------
    # Auto-buy tokens that pass scan
    # -------------------------------------------------------------------

    async def auto_buy_passed_tokens(self, passed_tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Automatically paper-buy tokens that passed all filters."""
        results = []
        for token_dict in passed_tokens:
            address = token_dict.get("address", "")
            if not address or address in self._positions:
                continue
            if self._paper_balance < self.config.position_size_usdc:
                break
            result = await self.paper_buy(address)
            results.append(result)
        return results

    # -------------------------------------------------------------------
    # Background scanning loop
    # -------------------------------------------------------------------

    async def start_background_scanner(self):
        """Start the background scanning task."""
        if self._scan_task and not self._scan_task.done():
            logger.warning("Background scanner already running")
            return

        self._scan_task = asyncio.create_task(self._scan_loop())
        logger.info("Background Solana scanner started (interval=%ds)", self.config.scan_interval_seconds)

    async def stop_background_scanner(self):
        """Stop the background scanning task."""
        if self._scan_task and not self._scan_task.done():
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass
        logger.info("Background Solana scanner stopped")

    async def _scan_loop(self):
        """Background loop that scans and auto-buys."""
        while True:
            try:
                if self.config.scanner_enabled:
                    result = await self.run_scan()
                    passed = result.get("passed_tokens", [])
                    if passed:
                        await self.auto_buy_passed_tokens(passed)

                    # Also update existing positions
                    await self.update_positions()

                await asyncio.sleep(self.config.scan_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Scanner loop error: %s", e, exc_info=True)
                await asyncio.sleep(30)

    # -------------------------------------------------------------------
    # Query methods
    # -------------------------------------------------------------------

    def get_all_tokens(self, status_filter: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all scanned tokens, optionally filtered by status."""
        tokens = list(self._all_tokens.values())

        if status_filter:
            tokens = [t for t in tokens if t.status.value == status_filter]

        # Sort by score descending, then discovery time
        tokens.sort(key=lambda t: (-t.score, t.discovered_at))
        tokens = tokens[:limit]

        return [t.to_dict() for t in tokens]

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all open Solana paper positions."""
        return [p.to_dict() for p in self._positions.values()]

    def get_trade_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get paper trade history."""
        trades = list(reversed(self._trades))[:limit]
        return [t.to_dict() for t in trades]

    def get_stats(self) -> Dict[str, Any]:
        """Get scanner statistics."""
        total_pnl = sum(t.pnl for t in self._trades if t.action == "sell")
        winning = sum(1 for t in self._trades if t.action == "sell" and t.pnl > 0)
        losing = sum(1 for t in self._trades if t.action == "sell" and t.pnl <= 0)
        total_closed = winning + losing

        return {
            "scanner_enabled": self.config.scanner_enabled,
            "birdeye_api_key_set": bool(self.config.birdeye_api_key),
            "total_scans": self._total_scans,
            "last_scan_time": self._last_scan_time.isoformat() if self._last_scan_time else None,
            "total_tokens_seen": self._total_tokens_seen,
            "total_tokens_passed": self._total_tokens_passed,
            "blacklisted_count": len(self._blacklist),
            "currently_scanning": self._scanning,
            "paper_balance": round(self._paper_balance, 2),
            "initial_balance": round(self._initial_balance, 2),
            "open_positions": len(self._positions),
            "total_trades": len(self._trades),
            "total_closed_trades": total_closed,
            "winning_trades": winning,
            "losing_trades": losing,
            "win_rate": round((winning / total_closed * 100) if total_closed > 0 else 0, 1),
            "total_realized_pnl": round(total_pnl, 4),
            "total_return_pct": round(
                ((self._paper_balance - self._initial_balance) / self._initial_balance) * 100
                if self._initial_balance > 0 else 0, 2
            ),
        }

    def update_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update scanner configuration dynamically."""
        for key, value in updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info("Scanner config updated: %s = %s", key, value)
        return self.config.to_dict()

    def reset(self):
        """Reset scanner state: clear positions, trades, balance."""
        self._positions.clear()
        self._trades.clear()
        self._trade_counter = 0
        self._paper_balance = self._initial_balance
        self._all_tokens.clear()
        self._blacklist.clear()
        self._total_scans = 0
        self._total_tokens_seen = 0
        self._total_tokens_passed = 0
        self._last_scan_time = None
        logger.info("Solana scanner reset | balance=$%.2f", self._paper_balance)
