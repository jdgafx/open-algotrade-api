import time
import pandas as pd
import pytest
from src.engine.data_cache import CandleCache


def _make_df():
    return pd.DataFrame({"close": [1.0, 2.0, 3.0], "open": [1.0, 2.0, 3.0],
                         "high": [1.1, 2.1, 3.1], "low": [0.9, 1.9, 2.9], "volume": [100, 200, 300]})


def test_cache_miss_returns_none():
    cache = CandleCache()
    assert cache.get("BTC", "1h", 30) is None


def test_cache_stores_and_retrieves():
    cache = CandleCache()
    df = _make_df()
    cache.set("BTC", "1h", 30, df)
    result = cache.get("BTC", "1h", 30)
    assert result is not None
    assert len(result) == 3
    assert list(result.columns) == list(df.columns)


def test_cache_returns_copy_not_reference():
    cache = CandleCache()
    df = _make_df()
    cache.set("BTC", "1h", 30, df)
    result = cache.get("BTC", "1h", 30)
    result["close"] = 999.0
    result2 = cache.get("BTC", "1h", 30)
    assert result2["close"].iloc[0] != 999.0


def test_cache_ttl_expiry():
    cache = CandleCache(ttl_seconds=1)
    cache.set("BTC", "1h", 30, _make_df())
    time.sleep(1.1)
    assert cache.get("BTC", "1h", 30) is None


def test_cache_invalidate_all():
    cache = CandleCache()
    cache.set("BTC", "1h", 30, _make_df())
    cache.set("ETH", "1h", 30, _make_df())
    cache.invalidate()
    assert cache.get("BTC", "1h", 30) is None
    assert cache.get("ETH", "1h", 30) is None


def test_cache_invalidate_by_symbol():
    cache = CandleCache()
    cache.set("BTC", "1h", 30, _make_df())
    cache.set("ETH", "1h", 30, _make_df())
    cache.invalidate(symbol="BTC")
    assert cache.get("BTC", "1h", 30) is None
    assert cache.get("ETH", "1h", 30) is not None


def test_cache_evicts_oldest_at_max_size():
    cache = CandleCache(max_size=2)
    cache.set("BTC", "1h", 30, _make_df())
    cache.set("ETH", "1h", 30, _make_df())
    cache.set("SOL", "1h", 30, _make_df())  # should evict oldest
    assert cache.stats()["size"] == 2


def test_cache_stats():
    cache = CandleCache(max_size=10, ttl_seconds=3600)
    cache.set("BTC", "1h", 30, _make_df())
    stats = cache.stats()
    assert stats["size"] == 1
    assert stats["max_size"] == 10
    assert stats["ttl_seconds"] == 3600
