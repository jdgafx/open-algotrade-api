"""
hyperliquid_ws_client.py — Hyperliquid native WebSocket candle sidecar.

T020 (from the T009 data-source audit, rank 1). Wraps a REST HyperliquidClient /
HyperliquidDataClient and serves get_ohlcv() from a LIVE Hyperliquid websocket candle
buffer instead of polling REST candleSnapshot every loop iteration. Drop-in: the
get_ohlcv contract is byte-identical and everything else delegates to the wrapped REST
client, so it can replace `client` with zero strategy/orchestrator changes.

Why: the live signal loop (orchestrator._run_strategy_loop -> client.get_ohlcv) was
REST-poll bound, so a setup valid mid-interval waited up to one poll to be acted on and
the poll budget capped how many (symbol, interval) pairs could be tracked. A ws candle
buffer lets strategies evaluate on freshly-closed bars the moment HL emits them, and
each extra subscription is near-free on one socket — raising QUALIFIED entry timeliness
and breadth without changing the qualification logic (same indicators/gates → no churn).

SAFETY (non-negotiable): a dead/slow socket must NEVER yield stale bars. get_ohlcv
detects buffer staleness and reconnect-failure and falls back to REST candleSnapshot,
degrading to EXACTLY today's behavior. The HL SDK's run_forever has no reconnect, so
this REST fallback + a rate-limited reconnect is the liveness guarantee.

Opt-in: only active when the orchestrator is constructed with env HL_WS_CANDLES truthy;
otherwise the orchestrator uses the plain REST client and this module is never imported.
"""

import logging
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import pandas as pd
import requests
from hyperliquid.info import Info

logger = logging.getLogger(__name__)

# HL candle interval -> milliseconds
_INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "8h": 28_800_000,
    "12h": 43_200_000, "1d": 86_400_000, "3d": 259_200_000, "1w": 604_800_000,
    "1M": 2_592_000_000,
}
_DAY_MS = 86_400_000


def _interval_ms(interval: str) -> int:
    return _INTERVAL_MS.get(interval, 3_600_000)


class HyperliquidWsClient:
    """Websocket candle feed wrapping a REST client; same get_ohlcv contract + REST fallback."""

    # how stale (in interval-multiples) before we distrust the buffer and use REST
    STALE_INTERVALS = float(os.getenv("HL_WS_STALE_INTERVALS", "2.5"))
    RECONNECT_COOLDOWN_S = float(os.getenv("HL_WS_RECONNECT_COOLDOWN_S", "30"))
    _MAX_CANDLES_PER_KEY = 5000

    def __init__(self, rest_client, max_keys: Optional[int] = None):
        # set _rest FIRST so __getattr__ delegation never recurses during init
        self._rest = rest_client
        self.base_url = getattr(rest_client, "base_url")
        self._buffers: Dict[Tuple[str, str], Dict[int, dict]] = {}
        self._sub_ids: Dict[Tuple[str, str], int] = {}
        self._lock = threading.RLock()
        # dedicated lock serializing ws (re)connect so concurrent cold-start callers
        # (~25 strategy loops hitting get_ohlcv at once) create exactly ONE shared connection.
        self._conn_lock = threading.Lock()
        self._info: Optional[Info] = None
        self._last_reconnect = 0.0
        self._construct_count = 0   # # of Info(skip_ws=False) constructions (observability)
        self._fallback_count = 0    # # of times get_ohlcv served REST instead of the ws buffer
        self._max_keys = max_keys if max_keys is not None else int(os.getenv("HL_WS_MAX_KEYS", "60"))
        self._ensure_info()

    # ── websocket lifecycle ───────────────────────────────────────────────
    def _ensure_info(self) -> bool:
        """Ensure a live ws connection exists; rate-limited (re)connect. Never raises.

        Uses double-checked locking on _conn_lock so that when many strategy loops call
        get_ohlcv concurrently on a cold container, only ONE thread constructs the shared
        Info(skip_ws=False) connection — the rest wait, then see it ready and return.
        """
        info = self._info
        if info is not None:
            wm = getattr(info, "ws_manager", None)
            if wm is not None and wm.is_alive():
                return True  # fast path, no lock
        with self._conn_lock:
            # re-check under the lock: another thread may have just connected
            info = self._info
            if info is not None:
                wm = getattr(info, "ws_manager", None)
                if wm is not None and wm.is_alive():
                    return True
            now = time.time()
            if info is not None and (now - self._last_reconnect) < self.RECONNECT_COOLDOWN_S:
                return False  # cooling down; REST fallback covers this window
            self._last_reconnect = now
            try:
                # Pass an empty spot_meta: the HL SDK eagerly processes spot metadata in
                # Info.__init__, and current mainnet spotMeta has a pair (@367) referencing a
                # token index beyond tokens[] -> IndexError on construct in some SDK versions
                # (e.g. 0.22.0). This is a PERP bot; candle subs resolve coins from perp `meta`,
                # not spot_meta, so skipping spot processing is both safe and necessary for
                # robustness against that SDK/data inconsistency. Verified live: BTC 1m candle
                # callbacks flow normally with this construction.
                new_info = Info(self.base_url, skip_ws=False, spot_meta={"universe": [], "tokens": []})
                self._construct_count += 1
                self._info = new_info
                with self._lock:
                    keys = list(self._buffers.keys())
                    self._sub_ids.clear()
                for sym, iv in keys:
                    self._do_subscribe(sym, iv)
                logger.info("HL ws connected (#%d, %d key(s) resubscribed)", self._construct_count, len(keys))
                return True
            except Exception as e:  # noqa: BLE001 - liveness must never crash the loop
                logger.warning("HL ws connect failed: %s (REST fallback active)", e)
                return False

    def _do_subscribe(self, symbol: str, interval: str) -> None:
        info = self._info
        if info is None:
            return
        try:
            sub = {"type": "candle", "coin": symbol, "interval": interval}
            sid = info.subscribe(sub, self._make_cb(symbol, interval))
            with self._lock:
                self._sub_ids[(symbol, interval)] = sid
        except Exception as e:  # noqa: BLE001
            logger.warning("HL ws subscribe %s %s failed: %s", symbol, interval, e)

    def _make_cb(self, symbol: str, interval: str):
        key = (symbol, interval)

        def _cb(msg):
            try:
                data = msg.get("data") if isinstance(msg, dict) else None
                if not data or "t" not in data:
                    return
                t = int(data["t"])
                with self._lock:
                    buf = self._buffers.setdefault(key, {})
                    buf[t] = data  # upsert: forming candle updates in place, closed bar stays
                    if len(buf) > self._MAX_CANDLES_PER_KEY:
                        for old in sorted(buf)[: len(buf) - self._MAX_CANDLES_PER_KEY]:
                            buf.pop(old, None)
            except Exception:  # noqa: BLE001
                logger.debug("HL ws candle callback parse error", exc_info=True)

        return _cb

    # ── REST seed / fallback ──────────────────────────────────────────────
    def _rest_snapshot(self, symbol: str, interval: str, lookback_days: int) -> list:
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)
        resp = requests.post(
            f"{self.base_url}/info",
            headers={"Content-Type": "application/json"},
            json={
                "type": "candleSnapshot",
                "req": {"coin": symbol, "interval": interval, "startTime": start_time, "endTime": end_time},
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json() or []

    def _seed(self, symbol: str, interval: str, lookback_days: int) -> int:
        snap = self._rest_snapshot(symbol, interval, lookback_days)
        with self._lock:
            buf = self._buffers.setdefault((symbol, interval), {})
            for c in snap:
                try:
                    buf[int(c["t"])] = c
                except (KeyError, TypeError, ValueError):
                    continue
        return len(snap)

    def _rest_fallback(self, symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
        """Serve from REST candleSnapshot and count it. A high fallback_count relative to
        request volume means the ws isn't serving — the key signal /ws/status exposes."""
        with self._lock:
            self._fallback_count += 1
        return self._rest.get_ohlcv(symbol, interval, lookback_days)

    @staticmethod
    def _to_df(candles: list) -> pd.DataFrame:
        """Native HL candle dicts -> the exact DataFrame shape HyperliquidClient.get_ohlcv returns."""
        if not candles:
            return pd.DataFrame()
        df = pd.DataFrame(candles)
        df = df.rename(
            columns={
                "t": "timestamp", "T": "close_timestamp", "o": "open", "h": "high",
                "l": "low", "c": "close", "v": "volume", "n": "num_trades",
                "s": "symbol", "i": "interval",
            }
        )
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = df[col].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    # ── the get_ohlcv contract (drop-in for HyperliquidClient.get_ohlcv) ──
    def get_ohlcv(
        self,
        symbol: str,
        interval: str = "1h",
        lookback_days: int = 7,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> pd.DataFrame:
        key = (symbol, interval)
        # Explicit time-range requests bypass the live buffer (used for backtests/exports).
        if start_time is not None or end_time is not None:
            return self._rest.get_ohlcv(symbol, interval, lookback_days, start_time, end_time)
        try:
            with self._lock:
                seeded = bool(self._buffers.get(key))
                subscribed = key in self._sub_ids
                at_cap = len(self._buffers) >= self._max_keys and key not in self._buffers

            if not seeded:
                if at_cap:
                    logger.debug("HL ws universe cap (%d) reached; REST for %s %s", self._max_keys, symbol, interval)
                    return self._rest_fallback(symbol, interval, lookback_days)
                self._seed(symbol, interval, lookback_days)
            if not subscribed and self._ensure_info():
                self._do_subscribe(symbol, interval)

            with self._lock:
                buf = dict(self._buffers.get(key, {}))
            if not buf:
                return self._rest_fallback(symbol, interval, lookback_days)

            # staleness guard: newest bar's close-time must be recent, else the socket is
            # dead/slow -> refresh from REST (and attempt a rate-limited reconnect).
            newest_t = max(buf)
            newest_close = int(buf[newest_t].get("T", newest_t + _interval_ms(interval)))
            now_ms = int(datetime.now().timestamp() * 1000)
            if now_ms - newest_close > self.STALE_INTERVALS * _interval_ms(interval):
                logger.debug("HL ws buffer stale for %s %s (%.0fs); REST refresh", symbol, interval, (now_ms - newest_close) / 1000)
                try:
                    self._seed(symbol, interval, lookback_days)
                    self._ensure_info()
                    with self._lock:
                        buf = dict(self._buffers.get(key, {}))
                except Exception:  # noqa: BLE001
                    return self._rest_fallback(symbol, interval, lookback_days)

            cutoff = now_ms - lookback_days * _DAY_MS
            rows = [c for t, c in sorted(buf.items()) if int(c.get("t", t)) >= cutoff]
            df = self._to_df(rows)
            if df.empty:
                return self._rest_fallback(symbol, interval, lookback_days)
            return df
        except Exception as e:  # noqa: BLE001 - any failure degrades to REST, never crashes the loop
            logger.warning("HL ws get_ohlcv %s %s -> REST fallback: %s", symbol, interval, e)
            return self._rest_fallback(symbol, interval, lookback_days)

    @property
    def ws_ready(self) -> bool:
        info = self._info
        wm = getattr(info, "ws_manager", None) if info is not None else None
        return bool(wm is not None and getattr(wm, "ws_ready", False))

    def status(self) -> dict:
        """Live ws-feed health for /ws/status — lets activation be VERIFIED in prod
        (the app's logger does not reach Railway stdout). ws_ready + buffers filling +
        a low fallback_count == the ws is genuinely serving (not silently all-REST)."""
        now_ms = int(datetime.now().timestamp() * 1000)
        info = self._info
        wm = getattr(info, "ws_manager", None) if info is not None else None
        keys: Dict[str, dict] = {}
        with self._lock:
            for (sym, iv), buf in self._buffers.items():
                if buf:
                    nt = max(buf)
                    nclose = int(buf[nt].get("T", nt + _interval_ms(iv)))
                    age = round((now_ms - nclose) / 1000, 1)
                else:
                    age = None
                keys[f"{sym}:{iv}"] = {
                    "buffer": len(buf),
                    "newest_bar_age_s": age,
                    "subscribed": (sym, iv) in self._sub_ids,
                }
            sub_count = len(self._sub_ids)
            fb = self._fallback_count
            cc = self._construct_count
        return {
            "ws_ready": self.ws_ready,
            "connected": bool(wm is not None and wm.is_alive()),
            "construct_count": cc,
            "subscribed_keys": sub_count,
            "buffered_keys": sum(1 for v in keys.values() if v["buffer"] > 0),
            "fallback_count": fb,
            "max_keys": self._max_keys,
            "keys": keys,
        }

    def __getattr__(self, name: str):
        # delegate every non-overridden attribute/method to the wrapped REST client
        if name == "_rest":
            raise AttributeError(name)
        return getattr(self._rest, name)
