"""
Canonical Hyperliquid SDK Wrapper — nice_funcs.py
Merged from Harvard + ALGOS + Extended versions into one production-ready module.

Key design decisions:
- Account parameter pattern (not global key) for multi-vault/subaccount support
- Testnet/mainnet toggle via env var or constructor param
- Logging instead of print()
- Async wrappers via asyncio.to_thread()
- All HL-native API calls (no ccxt dependency)
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import eth_account
import pandas as pd
import requests
from dotenv import load_dotenv
from eth_account.signers.local import LocalAccount
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

load_dotenv()

logger = logging.getLogger(__name__)

# The HL SDK eagerly processes spot metadata in Info/Exchange __init__. Current mainnet
# spotMeta has a pair (@367) referencing a token index beyond tokens[] (479 > 464), which
# IndexErrors on construct in some SDK versions (e.g. 0.22.0) and is a moving target as HL
# adds spot listings. This is a PERP-only bot — no code path needs spot metadata — so we
# pass an empty spot_meta to every Info/Exchange construction. This makes client
# construction immune to that SDK/data inconsistency regardless of the installed SDK
# version, removing a latent deploy landmine (unpinned SDK + a fresh pip install could
# otherwise brick all client construction on startup).
_EMPTY_SPOT_META = {"universe": [], "tokens": []}

# HL /info POST timeout (seconds). Without it a hung socket stalls the trade
# loop indefinitely (the call runs in a to_thread, so the whole strategy's
# next iteration never fires). ponytail: single value, lift to config if a
# second caller ever needs a different budget.
_HTTP_TIMEOUT_SECONDS = 15


@dataclass
class OrderResult:
    """Structured order result"""

    success: bool
    oid: Optional[int] = None
    status: str = ""
    raw: Optional[Dict] = None
    error: Optional[str] = None


class HyperliquidClient:
    """
    Production-ready Hyperliquid SDK wrapper.

    Supports:
    - Testnet and mainnet
    - Vault/subaccount trading via vault_address
    - All order types (limit, market, TP/SL)
    - Position management and risk controls
    - OHLCV data via native candleSnapshot API
    - L2 orderbook data
    """

    def __init__(
        self,
        account: Optional[LocalAccount] = None,
        private_key: Optional[str] = None,
        network: str = "testnet",
        vault_address: Optional[str] = None,
    ):
        """
        Initialize Hyperliquid client.

        Args:
            account: Pre-created LocalAccount (preferred for multi-account)
            private_key: Alternative to account — will create LocalAccount from key
            network: "testnet" or "mainnet"
            vault_address: If trading on behalf of a vault/subaccount
        """
        if account:
            self.account = account
        elif private_key:
            self.account = eth_account.Account.from_key(private_key)
        else:
            key = os.getenv("HYPERLIQUID_PRIVATE_KEY")
            if not key:
                raise ValueError(
                    "Must provide account, private_key, or set HYPERLIQUID_PRIVATE_KEY env var"
                )
            self.account = eth_account.Account.from_key(key)

        self.network = network.lower()
        if self.network == "mainnet":
            self.base_url = constants.MAINNET_API_URL
        else:
            self.base_url = constants.TESTNET_API_URL

        self.vault_address = vault_address or os.getenv("VAULT_ADDRESS")
        self.info = Info(self.base_url, skip_ws=True, spot_meta=_EMPTY_SPOT_META)
        self.exchange = Exchange(self.account, self.base_url, spot_meta=_EMPTY_SPOT_META)

        logger.info(
            "HyperliquidClient initialized | network=%s | account=%s | vault=%s",
            self.network,
            self.account.address,
            self.vault_address or "none",
        )

    # ──────────────────────────────────────────────
    # Market Data
    # ──────────────────────────────────────────────

    def ask_bid(self, symbol: str) -> Tuple[float, float, Dict]:
        """
        Get current ask and bid prices from L2 orderbook.

        Returns:
            (ask, bid, l2_data)
        """
        url = f"{self.base_url}/info"
        headers = {"Content-Type": "application/json"}
        data = {"type": "l2Book", "coin": symbol}

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        l2_data = response.json()["levels"]

        bid = float(l2_data[0][0]["px"])
        ask = float(l2_data[1][0]["px"])

        logger.debug("ask_bid | %s | ask=%.6f bid=%.6f", symbol, ask, bid)
        return ask, bid, l2_data

    def get_all_mids(self) -> Dict[str, str]:
        """Get mid prices for all coins."""
        url = f"{self.base_url}/info"
        data = {"type": "allMids"}
        response = requests.post(
            url, headers={"Content-Type": "application/json"}, json=data
        )
        response.raise_for_status()
        return response.json()

    def get_sz_px_decimals(self, symbol: str) -> Tuple[int, int]:
        """
        Get size and price decimal precision for a symbol.

        Returns:
            (sz_decimals, px_decimals)
        """
        url = f"{self.base_url}/info"
        headers = {"Content-Type": "application/json"}
        data = {"type": "meta"}

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        meta = response.json()

        symbols = meta["universe"]
        symbol_info = next((s for s in symbols if s["name"] == symbol), None)
        if not symbol_info:
            raise ValueError(f"Symbol {symbol} not found in universe")

        sz_decimals = symbol_info["szDecimals"]

        # Compute price decimals from current ask
        ask = self.ask_bid(symbol)[0]
        ask_str = str(ask)
        if "." in ask_str:
            px_decimals = len(ask_str.split(".")[1])
        else:
            px_decimals = 0

        logger.debug(
            "get_sz_px_decimals | %s | sz=%d px=%d", symbol, sz_decimals, px_decimals
        )
        return sz_decimals, px_decimals

    def get_meta(self) -> Dict:
        """Get full exchange metadata (universe, margin info, etc.)."""
        url = f"{self.base_url}/info"
        data = {"type": "meta"}
        response = requests.post(
            url, headers={"Content-Type": "application/json"}, json=data
        )
        response.raise_for_status()
        return response.json()

    def get_ohlcv(
        self,
        symbol: str,
        interval: str = "1h",
        lookback_days: int = 7,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Get OHLCV candle data via native HL candleSnapshot API.

        Args:
            symbol: Coin symbol (e.g. "BTC", "ETH")
            interval: "1m","3m","5m","15m","30m","1h","2h","4h","8h","12h","1d","3d","1w","1M"
            lookback_days: How many days of history
            start_time: Override start time (epoch ms)
            end_time: Override end time (epoch ms)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume, num_trades
        """
        if end_time is None:
            end_time = int(datetime.now().timestamp() * 1000)
        if start_time is None:
            start_dt = datetime.now() - timedelta(days=lookback_days)
            start_time = int(start_dt.timestamp() * 1000)

        url = f"{self.base_url}/info"
        headers = {"Content-Type": "application/json"}
        data = {
            "type": "candleSnapshot",
            "req": {
                "coin": symbol,
                "interval": interval,
                "startTime": start_time,
                "endTime": end_time,
            },
        }

        response = requests.post(
            url, headers=headers, json=data, timeout=_HTTP_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        snapshot = response.json()

        if not snapshot:
            logger.warning("get_ohlcv | %s | no data returned", symbol)
            return pd.DataFrame()

        df = pd.DataFrame(snapshot)
        df = df.rename(
            columns={
                "t": "timestamp",
                "T": "close_timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "n": "num_trades",
                "s": "symbol",
                "i": "interval",
            }
        )
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = df[col].astype(float)

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        logger.info("get_ohlcv | %s %s | %d candles", symbol, interval, len(df))
        return df

    def get_l2_book(self, symbol: str, n_sig_figs: Optional[int] = None) -> Dict:
        """Get L2 orderbook snapshot (up to 20 levels per side)."""
        url = f"{self.base_url}/info"
        data = {"type": "l2Book", "coin": symbol}
        if n_sig_figs:
            data["nSigFigs"] = n_sig_figs
        response = requests.post(
            url, headers={"Content-Type": "application/json"}, json=data
        )
        response.raise_for_status()
        return response.json()

    def get_funding_rate(self, symbol: str) -> Optional[float]:
        """Get current funding rate for a symbol."""
        meta = self.get_meta()
        for asset_ctx in meta.get("assetCtxs", []):
            # Match by index
            pass
        # Alternative: use the meta endpoint with asset contexts
        url = f"{self.base_url}/info"
        data = {"type": "metaAndAssetCtxs"}
        response = requests.post(
            url, headers={"Content-Type": "application/json"}, json=data
        )
        response.raise_for_status()
        result = response.json()

        if isinstance(result, list) and len(result) >= 2:
            universe = result[0]["universe"]
            ctxs = result[1]
            for i, asset in enumerate(universe):
                if asset["name"] == symbol and i < len(ctxs):
                    return float(ctxs[i].get("funding", 0))
        return None

    # ──────────────────────────────────────────────
    # Account & Position Management
    # ──────────────────────────────────────────────

    def get_user_state(self, address: Optional[str] = None) -> Dict:
        """Get full user state including positions, margin, balances."""
        addr = address or self.account.address
        return self.info.user_state(addr)

    def get_account_value(self, address: Optional[str] = None) -> float:
        """Get total account value in USD."""
        state = self.get_user_state(address)
        return float(state["marginSummary"]["accountValue"])

    def get_withdrawable(self, address: Optional[str] = None) -> float:
        """Get withdrawable balance."""
        state = self.get_user_state(address)
        return float(state.get("withdrawable", 0))

    def get_position(
        self, symbol: str, address: Optional[str] = None
    ) -> Tuple[List, bool, float, Optional[str], float, float, Optional[bool]]:
        """
        Get current position info for a symbol.

        Returns:
            (positions, in_pos, size, pos_sym, entry_px, pnl_perc, long)
            - positions: list of position dicts
            - in_pos: True if has open position
            - size: position size (negative = short)
            - pos_sym: symbol of position
            - entry_px: entry price
            - pnl_perc: return on equity as percentage
            - long: True if long, False if short, None if no position
        """
        addr = address or self.account.address
        user_state = self.info.user_state(addr)
        acct_value = float(user_state["marginSummary"]["accountValue"])
        logger.debug("get_position | account_value=%.2f", acct_value)

        positions = []
        for position in user_state.get("assetPositions", []):
            pos = position["position"]
            if pos["coin"] == symbol and float(pos["szi"]) != 0:
                positions.append(pos)
                size = float(pos["szi"])
                entry_px = float(pos["entryPx"])
                pnl_perc = float(pos["returnOnEquity"]) * 100
                long = size > 0

                logger.info(
                    "get_position | %s | size=%.6f entry=%.2f pnl=%.2f%% long=%s",
                    symbol,
                    size,
                    entry_px,
                    pnl_perc,
                    long,
                )
                return positions, True, size, symbol, entry_px, pnl_perc, long

        return [], False, 0.0, None, 0.0, 0.0, None

    def get_all_positions(self, address: Optional[str] = None) -> List[Dict]:
        """Get all open positions."""
        addr = address or self.account.address
        user_state = self.info.user_state(addr)
        positions = []
        for position in user_state.get("assetPositions", []):
            pos = position["position"]
            if float(pos["szi"]) != 0:
                positions.append(
                    {
                        "coin": pos["coin"],
                        "size": float(pos["szi"]),
                        "entry_px": float(pos["entryPx"]),
                        "unrealized_pnl": float(pos.get("unrealizedPnl", 0)),
                        "return_on_equity": float(pos.get("returnOnEquity", 0)) * 100,
                        "leverage": float(pos.get("leverage", {}).get("value", 1)),
                        "long": float(pos["szi"]) > 0,
                    }
                )
        return positions

    def get_open_orders(self, address: Optional[str] = None) -> List[Dict]:
        """Get all open orders."""
        addr = address or self.account.address
        return self.info.open_orders(addr)

    def get_user_fills(self, address: Optional[str] = None) -> List[Dict]:
        """Get recent fills (up to 2000)."""
        addr = address or self.account.address
        url = f"{self.base_url}/info"
        data = {"type": "userFills", "user": addr}
        response = requests.post(
            url, headers={"Content-Type": "application/json"}, json=data
        )
        response.raise_for_status()
        return response.json()

    def get_subaccounts(self, address: Optional[str] = None) -> List[Dict]:
        """Get user's subaccounts."""
        addr = address or self.account.address
        url = f"{self.base_url}/info"
        data = {"type": "subAccounts", "user": addr}
        response = requests.post(
            url, headers={"Content-Type": "application/json"}, json=data
        )
        response.raise_for_status()
        return response.json()

    # ──────────────────────────────────────────────
    # Vault Operations
    # ──────────────────────────────────────────────

    def get_vault_details(self, vault_address: Optional[str] = None) -> Dict:
        """Get vault details including followers, PnL, etc."""
        addr = vault_address or self.vault_address
        if not addr:
            raise ValueError("No vault address provided")
        url = f"{self.base_url}/info"
        data = {
            "type": "vaultDetails",
            "vaultAddress": addr,
            "user": self.account.address,
        }
        response = requests.post(
            url, headers={"Content-Type": "application/json"}, json=data
        )
        response.raise_for_status()
        return response.json()

    def get_user_vault_equities(self, address: Optional[str] = None) -> List[Dict]:
        """Get user's vault deposit equities."""
        addr = address or self.account.address
        url = f"{self.base_url}/info"
        data = {"type": "userVaultEquities", "user": addr}
        response = requests.post(
            url, headers={"Content-Type": "application/json"}, json=data
        )
        response.raise_for_status()
        return response.json()

    # ──────────────────────────────────────────────
    # Order Management
    # ──────────────────────────────────────────────

    def limit_order(
        self,
        coin: str,
        is_buy: bool,
        sz: float,
        limit_px: float,
        reduce_only: bool = False,
        tif: str = "Gtc",
        vault_address: Optional[str] = None,
    ) -> OrderResult:
        """
        Place a limit order.

        Args:
            coin: Symbol (e.g. "BTC", "ETH")
            is_buy: True for buy, False for sell
            sz: Order size in asset units
            limit_px: Limit price
            reduce_only: If True, only reduces existing position
            tif: Time in force — "Gtc", "Alo" (post only), "Ioc"
            vault_address: Override vault address for this order

        Returns:
            OrderResult with success status and order ID
        """
        # Round size to correct decimals
        sz_decimals = self.get_sz_px_decimals(coin)[0]
        sz = round(sz, sz_decimals)

        if sz == 0:
            return OrderResult(success=False, error="Size rounds to 0")

        logger.info(
            "limit_order | %s %s %.6f @ %.2f | reduce_only=%s tif=%s",
            coin,
            "BUY" if is_buy else "SELL",
            sz,
            limit_px,
            reduce_only,
            tif,
        )

        vault = vault_address or self.vault_address

        order_result = self.exchange.order(
            coin,
            is_buy,
            sz,
            limit_px,
            {"limit": {"tif": tif}},
            reduce_only=reduce_only,
            vault_address=vault,
        )

        if isinstance(order_result, dict) and "response" in order_result:
            statuses = order_result["response"]["data"]["statuses"]
            if statuses:
                status = statuses[0]
                if "resting" in status:
                    return OrderResult(
                        success=True,
                        oid=status["resting"]["oid"],
                        status="resting",
                        raw=order_result,
                    )
                elif "filled" in status:
                    return OrderResult(
                        success=True,
                        oid=status["filled"]["oid"],
                        status="filled",
                        raw=order_result,
                    )
                elif "error" in status:
                    return OrderResult(
                        success=False,
                        error=status["error"],
                        raw=order_result,
                    )

        return OrderResult(success=True, status="unknown", raw=order_result)

    def market_order(
        self,
        coin: str,
        is_buy: bool,
        sz: float,
        slippage: float = 0.01,
        reduce_only: bool = False,
        vault_address: Optional[str] = None,
    ) -> OrderResult:
        """
        Place a market order (IOC limit at slippage from current price).

        Args:
            coin: Symbol
            is_buy: True for buy, False for sell
            sz: Size in asset units
            slippage: Slippage tolerance (0.01 = 1%)
            reduce_only: If True, only reduces position
            vault_address: Override vault address
        """
        ask, bid, _ = self.ask_bid(coin)

        if is_buy:
            limit_px = round(ask * (1 + slippage), self.get_sz_px_decimals(coin)[1])
        else:
            limit_px = round(bid * (1 - slippage), self.get_sz_px_decimals(coin)[1])

        return self.limit_order(
            coin,
            is_buy,
            sz,
            limit_px,
            reduce_only=reduce_only,
            tif="Ioc",
            vault_address=vault_address,
        )

    def trigger_order(
        self,
        coin: str,
        is_buy: bool,
        sz: float,
        trigger_px: float,
        tpsl: str = "sl",
        is_market: bool = True,
        reduce_only: bool = True,
        vault_address: Optional[str] = None,
    ) -> OrderResult:
        """
        Place a trigger (TP/SL) order.

        Args:
            coin: Symbol
            is_buy: True for buy, False for sell
            sz: Size
            trigger_px: Trigger price
            tpsl: "tp" for take profit, "sl" for stop loss
            is_market: If True, executes as market when triggered
            reduce_only: Usually True for TP/SL
            vault_address: Override vault address
        """
        sz_decimals = self.get_sz_px_decimals(coin)[0]
        sz = round(sz, sz_decimals)

        logger.info(
            "trigger_order | %s %s %.6f trigger=%.2f %s",
            coin,
            "BUY" if is_buy else "SELL",
            sz,
            trigger_px,
            tpsl.upper(),
        )

        vault = vault_address or self.vault_address
        # For trigger orders, limit_px is ignored when is_market=True
        order_result = self.exchange.order(
            coin,
            is_buy,
            sz,
            trigger_px,  # limit price (ignored for market triggers)
            {
                "trigger": {
                    "isMarket": is_market,
                    "triggerPx": str(trigger_px),
                    "tpsl": tpsl,
                }
            },
            reduce_only=reduce_only,
            vault_address=vault,
        )

        return OrderResult(success=True, status="trigger_placed", raw=order_result)

    def cancel_order(
        self,
        coin: str,
        oid: int,
        vault_address: Optional[str] = None,
    ) -> bool:
        """Cancel a specific order by OID."""
        vault = vault_address or self.vault_address
        result = self.exchange.cancel(coin, oid, vault_address=vault)
        logger.info("cancel_order | %s oid=%d | result=%s", coin, oid, result)
        return True

    def cancel_all_orders(
        self,
        coin: Optional[str] = None,
        vault_address: Optional[str] = None,
    ) -> int:
        """
        Cancel all open orders, optionally filtered by coin.

        Returns:
            Number of orders cancelled
        """
        addr = self.account.address
        open_orders = self.info.open_orders(addr)

        if coin:
            open_orders = [o for o in open_orders if o["coin"] == coin]

        cancelled = 0
        vault = vault_address or self.vault_address
        for order in open_orders:
            try:
                self.exchange.cancel(order["coin"], order["oid"], vault_address=vault)
                cancelled += 1
            except Exception as e:
                logger.warning("Failed to cancel order %s: %s", order["oid"], e)

        logger.info("cancel_all_orders | coin=%s | cancelled=%d", coin, cancelled)
        return cancelled

    # ──────────────────────────────────────────────
    # Leverage & Margin
    # ──────────────────────────────────────────────

    def update_leverage(
        self,
        symbol: str,
        leverage: int,
        is_cross: bool = False,
        vault_address: Optional[str] = None,
    ) -> Dict:
        """Update leverage for a symbol."""
        vault = vault_address or self.vault_address
        result = self.exchange.update_leverage(
            leverage, symbol, is_cross=is_cross, vault_address=vault
        )
        logger.info(
            "update_leverage | %s | leverage=%d cross=%s", symbol, leverage, is_cross
        )
        return result

    def adjust_leverage_usd_size(
        self,
        symbol: str,
        usd_size: float,
        leverage: int,
    ) -> Tuple[int, float]:
        """
        Calculate position size from USD amount and leverage.

        Returns:
            (leverage, size_in_asset_units)
        """
        self.update_leverage(symbol, leverage, is_cross=False)

        price = self.ask_bid(symbol)[0]
        size = (usd_size / price) * leverage
        sz_decimals = self.get_sz_px_decimals(symbol)[0]
        size = round(size, sz_decimals)

        logger.info(
            "adjust_leverage_usd_size | %s | usd=$%.2f lev=%dx size=%.6f",
            symbol,
            usd_size,
            leverage,
            size,
        )
        return leverage, size

    def usd_to_asset_size(self, symbol: str, usd_amount: float) -> float:
        """Convert USD amount to asset size at current mid price."""
        ask, bid, _ = self.ask_bid(symbol)
        mid = (ask + bid) / 2
        if mid <= 0:
            raise ValueError(f"Invalid mid price for {symbol}: {mid}")

        sz_decimals = self.get_sz_px_decimals(symbol)[0]
        size = round(usd_amount / mid, sz_decimals)
        return size

    # ──────────────────────────────────────────────
    # Risk Management
    # ──────────────────────────────────────────────

    def kill_switch(
        self,
        symbol: str,
        max_retries: int = 5,
        vault_address: Optional[str] = None,
    ) -> bool:
        """
        Emergency close all positions for a symbol.
        Cancels orders, then aggressively market-closes the position.

        Returns:
            True if position successfully closed
        """
        logger.warning("KILL SWITCH ACTIVATED | %s", symbol)

        vault = vault_address or self.vault_address

        for attempt in range(max_retries):
            self.cancel_all_orders(symbol, vault_address=vault)

            _, in_pos, size, _, _, _, long = self.get_position(symbol)

            if not in_pos:
                logger.info("kill_switch | %s | position closed successfully", symbol)
                return True

            abs_size = abs(size)
            is_buy = not long  # Buy to close short, sell to close long

            result = self.market_order(
                symbol,
                is_buy,
                abs_size,
                slippage=0.03,  # 3% slippage tolerance for emergency
                reduce_only=True,
                vault_address=vault,
            )

            logger.info(
                "kill_switch | %s | attempt %d | %s %.6f | result=%s",
                symbol,
                attempt + 1,
                "BUY" if is_buy else "SELL",
                abs_size,
                result.status,
            )

            time.sleep(2)

        logger.error("kill_switch | %s | FAILED after %d attempts", symbol, max_retries)
        return False

    def pnl_close(
        self,
        symbol: str,
        target_pct: float = 5.0,
        max_loss_pct: float = -5.0,
        vault_address: Optional[str] = None,
    ) -> Tuple[bool, bool, float, Optional[bool]]:
        """
        Check PnL and close position if target or max loss is hit.

        Args:
            symbol: Trading symbol
            target_pct: Profit target percentage (e.g. 5.0 = 5%)
            max_loss_pct: Max loss percentage (e.g. -5.0 = -5%)

        Returns:
            (pnl_closed, in_pos, size, long)
        """
        _, in_pos, size, _, entry_px, pnl_perc, long = self.get_position(symbol)

        if not in_pos:
            return False, False, 0.0, None

        logger.info(
            "pnl_close | %s | pnl=%.2f%% target=%.2f%% max_loss=%.2f%%",
            symbol,
            pnl_perc,
            target_pct,
            max_loss_pct,
        )

        vault = vault_address or self.vault_address

        if pnl_perc >= target_pct:
            logger.info("pnl_close | %s | TARGET HIT at %.2f%%", symbol, pnl_perc)
            self.kill_switch(symbol, vault_address=vault)
            return True, True, size, long

        if pnl_perc <= max_loss_pct:
            logger.warning("pnl_close | %s | MAX LOSS HIT at %.2f%%", symbol, pnl_perc)
            self.kill_switch(symbol, vault_address=vault)
            return True, True, size, long

        return False, True, size, long

    def account_guard(
        self,
        min_account_value: float = 10.0,
        symbols: Optional[List[str]] = None,
        vault_address: Optional[str] = None,
    ) -> bool:
        """
        Emergency guard — if account drops below minimum, close ALL positions.

        Returns:
            True if guard was triggered (positions closed)
        """
        acct_val = self.get_account_value()
        if acct_val < min_account_value:
            logger.critical(
                "ACCOUNT GUARD | value=$%.2f < min=$%.2f | CLOSING ALL",
                acct_val,
                min_account_value,
            )
            if symbols:
                for sym in symbols:
                    self.kill_switch(sym, vault_address=vault_address)
            else:
                # Close all positions
                positions = self.get_all_positions()
                for pos in positions:
                    self.kill_switch(pos["coin"], vault_address=vault_address)
            return True
        return False

    # ──────────────────────────────────────────────
    # Dead Man's Switch (Schedule Cancel)
    # ──────────────────────────────────────────────

    def schedule_cancel(
        self,
        cancel_time_ms: Optional[int] = None,
        seconds_from_now: int = 300,
    ) -> Dict:
        """
        Set a dead man's switch — all orders cancelled if not refreshed.

        Args:
            cancel_time_ms: Exact cancel time in epoch ms
            seconds_from_now: Alternative — cancel N seconds from now
        """
        if cancel_time_ms is None:
            cancel_time_ms = int((time.time() + seconds_from_now) * 1000)

        result = self.exchange.schedule_cancel(cancel_time_ms)
        logger.info(
            "schedule_cancel | cancel at %s",
            datetime.fromtimestamp(cancel_time_ms / 1000),
        )
        return result

    # ──────────────────────────────────────────────
    # Technical Analysis Helpers
    # ──────────────────────────────────────────────

    def get_ohlcv_with_indicators(
        self,
        symbol: str,
        interval: str = "4h",
        lookback_days: int = 30,
        sma_period: int = 20,
    ) -> pd.DataFrame:
        """
        Get OHLCV data with SMA indicator pre-calculated.

        Returns:
            DataFrame with OHLCV + sma column
        """
        df = self.get_ohlcv(symbol, interval, lookback_days)
        if df.empty:
            return df

        df[f"sma{sma_period}"] = df["close"].rolling(sma_period).mean()
        df["support"] = (
            df["close"].iloc[:-2].min() if len(df) > 2 else df["close"].min()
        )
        df["resistance"] = (
            df["close"].iloc[:-2].max() if len(df) > 2 else df["close"].max()
        )

        return df

    def supply_demand_zones(
        self,
        symbol: str,
        timeframe: str = "4h",
        lookback_days: int = 30,
        zone_threshold: float = 0.02,
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        Calculate supply and demand zones from OHLCV data.

        Returns:
            {"supply": [(high, low), ...], "demand": [(high, low), ...]}
        """
        df = self.get_ohlcv(symbol, timeframe, lookback_days)
        if df.empty:
            return {"supply": [], "demand": []}

        supply_zones = []
        demand_zones = []

        for i in range(2, len(df) - 2):
            # Demand zone: price drops then reverses up
            if (
                df.iloc[i]["low"] < df.iloc[i - 1]["low"]
                and df.iloc[i]["low"] < df.iloc[i + 1]["low"]
                and df.iloc[i + 1]["close"] > df.iloc[i]["high"]
            ):
                demand_zones.append(
                    (float(df.iloc[i]["high"]), float(df.iloc[i]["low"]))
                )

            # Supply zone: price rises then reverses down
            if (
                df.iloc[i]["high"] > df.iloc[i - 1]["high"]
                and df.iloc[i]["high"] > df.iloc[i + 1]["high"]
                and df.iloc[i + 1]["close"] < df.iloc[i]["low"]
            ):
                supply_zones.append(
                    (float(df.iloc[i]["high"]), float(df.iloc[i]["low"]))
                )

        return {"supply": supply_zones[-5:], "demand": demand_zones[-5:]}

    def volume_spike(self, df: pd.DataFrame, vol_multiplier: float = 2.0) -> bool:
        """Check if current volume is a spike (above multiplier * average)."""
        if "volume" not in df.columns or len(df) < 20:
            return False
        avg_vol = df["volume"].iloc[-20:].mean()
        current_vol = df["volume"].iloc[-1]
        return current_vol > (avg_vol * vol_multiplier)


class HyperliquidDataClient:
    """
    Read-only data client for Hyperliquid public API.

    No wallet, no private key, no account needed.
    Only fetches market data (OHLCV, prices, orderbooks, funding rates).
    Drop-in replacement for HyperliquidClient where only data methods are needed.
    """

    def __init__(self, network: str = "mainnet"):
        self.network = network.lower()
        if self.network == "mainnet":
            self.base_url = constants.MAINNET_API_URL
        else:
            self.base_url = constants.TESTNET_API_URL

        self.vault_address = None
        self.info = Info(self.base_url, skip_ws=True, spot_meta=_EMPTY_SPOT_META)

        logger.info("HyperliquidDataClient initialized (read-only) | network=%s", self.network)

    def get_ohlcv(
        self,
        symbol: str,
        interval: str = "1h",
        lookback_days: int = 7,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> pd.DataFrame:
        """Get OHLCV candle data (same as HyperliquidClient.get_ohlcv)."""
        if end_time is None:
            end_time = int(datetime.now().timestamp() * 1000)
        if start_time is None:
            start_dt = datetime.now() - timedelta(days=lookback_days)
            start_time = int(start_dt.timestamp() * 1000)

        url = f"{self.base_url}/info"
        headers = {"Content-Type": "application/json"}
        data = {
            "type": "candleSnapshot",
            "req": {
                "coin": symbol,
                "interval": interval,
                "startTime": start_time,
                "endTime": end_time,
            },
        }

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        snapshot = response.json()

        if not snapshot:
            return pd.DataFrame()

        df = pd.DataFrame(snapshot)
        df = df.rename(columns={
            "t": "timestamp", "T": "close_timestamp",
            "o": "open", "h": "high", "l": "low", "c": "close",
            "v": "volume", "n": "num_trades", "s": "symbol", "i": "interval",
        })
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = df[col].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def ask_bid(self, symbol: str) -> Tuple[float, float, Dict]:
        """Get current ask and bid prices."""
        url = f"{self.base_url}/info"
        data = {"type": "l2Book", "coin": symbol}
        response = requests.post(url, headers={"Content-Type": "application/json"}, json=data)
        response.raise_for_status()
        l2_data = response.json()["levels"]
        bid = float(l2_data[0][0]["px"])
        ask = float(l2_data[1][0]["px"])
        return ask, bid, l2_data

    def get_all_mids(self) -> Dict[str, str]:
        """Get mid prices for all coins."""
        url = f"{self.base_url}/info"
        data = {"type": "allMids"}
        response = requests.post(url, headers={"Content-Type": "application/json"}, json=data)
        response.raise_for_status()
        return response.json()

    def get_meta(self) -> Dict:
        """Get exchange metadata."""
        url = f"{self.base_url}/info"
        data = {"type": "meta"}
        response = requests.post(url, headers={"Content-Type": "application/json"}, json=data)
        response.raise_for_status()
        return response.json()

    def get_funding_rate(self, symbol: str) -> Optional[float]:
        """Get current funding rate."""
        url = f"{self.base_url}/info"
        data = {"type": "metaAndAssetCtxs"}
        response = requests.post(url, headers={"Content-Type": "application/json"}, json=data)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and len(result) >= 2:
            universe = result[0]["universe"]
            ctxs = result[1]
            for i, asset in enumerate(universe):
                if asset["name"] == symbol and i < len(ctxs):
                    return float(ctxs[i].get("funding", 0))
        return None


# ──────────────────────────────────────────────
# Module-level convenience functions
# ──────────────────────────────────────────────

_default_client: Optional[HyperliquidClient] = None


def get_client(**kwargs) -> HyperliquidClient:
    """Get or create the default HyperliquidClient singleton."""
    global _default_client
    if _default_client is None:
        network = os.getenv("HYPERLIQUID_NETWORK", "testnet")
        _default_client = HyperliquidClient(network=network, **kwargs)
    return _default_client


# Async wrappers for use in asyncio strategies
async def async_ask_bid(client: HyperliquidClient, symbol: str):
    import asyncio

    return await asyncio.to_thread(client.ask_bid, symbol)


async def async_get_position(client: HyperliquidClient, symbol: str):
    import asyncio

    return await asyncio.to_thread(client.get_position, symbol)


async def async_get_ohlcv(
    client: HyperliquidClient, symbol: str, interval: str = "1h", lookback_days: int = 7
):
    import asyncio

    return await asyncio.to_thread(client.get_ohlcv, symbol, interval, lookback_days)


async def async_limit_order(
    client: HyperliquidClient,
    coin: str,
    is_buy: bool,
    sz: float,
    limit_px: float,
    reduce_only: bool = False,
):
    import asyncio

    return await asyncio.to_thread(
        client.limit_order, coin, is_buy, sz, limit_px, reduce_only
    )


async def async_market_order(
    client: HyperliquidClient,
    coin: str,
    is_buy: bool,
    sz: float,
    slippage: float = 0.01,
    reduce_only: bool = False,
):
    import asyncio

    return await asyncio.to_thread(
        client.market_order, coin, is_buy, sz, slippage, reduce_only
    )


async def async_kill_switch(client: HyperliquidClient, symbol: str):
    import asyncio

    return await asyncio.to_thread(client.kill_switch, symbol)
