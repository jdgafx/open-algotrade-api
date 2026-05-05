import time
from threading import Lock
from typing import Optional
import pandas as pd


class CandleCache:
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 50):
        self._cache: dict[str, tuple[pd.DataFrame, float]] = {}
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._lock = Lock()

    def _key(self, symbol: str, timeframe: str, lookback_days: int) -> str:
        return f"{symbol}:{timeframe}:{lookback_days}"

    def get(self, symbol: str, timeframe: str, lookback_days: int) -> Optional[pd.DataFrame]:
        key = self._key(symbol, timeframe, lookback_days)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            data, ts = entry
            now = time.monotonic()
            if now - ts > self._ttl:
                del self._cache[key]
                return None
            self._cache[key] = (data, now)  # refresh access time for LRU eviction
            return data.copy()

    def set(self, symbol: str, timeframe: str, lookback_days: int, data: pd.DataFrame) -> None:
        key = self._key(symbol, timeframe, lookback_days)
        with self._lock:
            if len(self._cache) >= self._max_size and key not in self._cache:
                oldest = min(self._cache, key=lambda k: self._cache[k][1])
                del self._cache[oldest]
            self._cache[key] = (data.copy(), time.monotonic())

    def invalidate(self, symbol: Optional[str] = None) -> None:
        with self._lock:
            if symbol is None:
                self._cache.clear()
            else:
                for k in [k for k in self._cache if k.startswith(f"{symbol}:")]:
                    del self._cache[k]

    def stats(self) -> dict:
        with self._lock:
            return {"size": len(self._cache), "max_size": self._max_size, "ttl_seconds": self._ttl}


candle_cache = CandleCache()
