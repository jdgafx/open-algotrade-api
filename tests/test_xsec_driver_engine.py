"""
Tests for XsecDriverEngine.

Covers:
  - realized_vol driver value computation
  - dollar_volume driver value computation
  - sign convention: sign=-1 → LONG low-vol (score inversion)
  - sign=+1 → LONG high-dollar-vol
  - rank_basket reuse via engine (dollar-neutral, equal baskets)
  - unknown driver rejected at construction
  - POST /xsec/instances endpoint creates+persists+starts an instance

No network I/O — pure unit assertions.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.engine.xsec_driver_engine import XsecDriverEngine, SUPPORTED_DRIVERS
from src.engine.xsec_engine import rank_basket


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_candles(closes, volumes=None):
    """Build minimal candleSnapshot-style dicts for testing."""
    if volumes is None:
        volumes = [1.0] * len(closes)
    return [{"c": str(c), "v": str(v), "o": str(c), "h": str(c), "l": str(c), "t": i}
            for i, (c, v) in enumerate(zip(closes, volumes))]


class _FakeClient:
    base_url = "http://fake"


class _FakeExecutor:
    pass


def _make_engine(driver="realized_vol_carry", sign=-1, coins=None, lookback=5):
    return XsecDriverEngine(
        executor=_FakeExecutor(),
        client=_FakeClient(),
        name="test",
        driver=driver,
        lookback=lookback,
        q=0.30,
        sign=sign,
        coins=coins or ["BTC", "ETH", "SOL", "BNB", "XRP"],
        per_leg_usd=50.0,
        rebalance_secs=3600,
    )


# ── Driver value tests ────────────────────────────────────────────────────────

def test_realized_vol_low_vol_coin():
    """A coin with identical closes has zero realized vol."""
    eng = _make_engine(driver="realized_vol_carry", lookback=5)
    candles = _make_candles([100.0] * 6)  # no price movement
    val = eng._compute_driver_value(candles)
    assert val == pytest.approx(0.0, abs=1e-10)


def test_realized_vol_high_vol_coin():
    """A coin with large swings has non-zero realized vol."""
    eng = _make_engine(driver="realized_vol_carry", lookback=5)
    candles = _make_candles([100, 110, 90, 120, 80, 110])
    val = eng._compute_driver_value(candles)
    assert val > 0.0


def test_realized_vol_insufficient_candles_returns_none():
    eng = _make_engine(driver="realized_vol_carry", lookback=10)
    candles = _make_candles([100.0] * 5)  # fewer than lookback
    assert eng._compute_driver_value(candles) is None


def test_dollar_volume_value():
    """dollar_volume = mean(close * volume) over lookback bars."""
    eng = _make_engine(driver="dollar_volume", lookback=3)
    candles = _make_candles([100.0, 200.0, 300.0], volumes=[1.0, 1.0, 1.0])
    val = eng._compute_driver_value(candles)
    assert val == pytest.approx((100 + 200 + 300) / 3)


def test_dollar_volume_with_varying_volume():
    eng = _make_engine(driver="dollar_volume", lookback=2)
    candles = _make_candles([100.0, 200.0], volumes=[2.0, 3.0])
    val = eng._compute_driver_value(candles)
    assert val == pytest.approx((100 * 2 + 200 * 3) / 2)


# ── Sign convention tests ─────────────────────────────────────────────────────

def test_sign_minus1_realized_vol_longs_low_vol():
    """sign=-1 + realized_vol: low-vol coins end up in long basket.

    score = -sign * vol = +vol for sign=-1
    rank_basket(score) → long = bottom-q = low-vol coins ✓
    """
    # BTC=0 vol (flat), ETH=high vol (swinging), plus filler coins
    flat_candles   = _make_candles([100.0] * 6)             # vol=0
    swing_candles  = _make_candles([100, 120, 80, 130, 70, 110])  # high vol

    # scores = -(-1)*vol = +vol; flat → score≈0 (lowest) → long_set ✓
    eng = _make_engine(driver="realized_vol_carry", sign=-1, lookback=5,
                       coins=["BTC", "ETH", "SOL", "BNB", "XRP"])

    btc_val = eng._compute_driver_value(flat_candles)    # ≈ 0
    eth_val = eng._compute_driver_value(swing_candles)   # > 0
    assert btc_val is not None
    assert eth_val is not None

    # Apply sign convention manually (mirrors engine internals)
    scores = {
        "BTC": -(-1) * btc_val,   # +0 (low) → long candidate
        "ETH": -(-1) * eth_val,   # +high → short candidate
        "SOL": -(-1) * (btc_val + eth_val) / 3,
        "BNB": -(-1) * (btc_val + eth_val) / 2,
        "XRP": -(-1) * eth_val * 0.5,
    }
    long_set, short_set = rank_basket(scores, q=0.30)
    assert "BTC" in long_set, "low-vol BTC should be longed with sign=-1"
    assert "ETH" in short_set, "high-vol ETH should be shorted with sign=-1"


def test_sign_plus1_dollar_volume_longs_high_dv():
    """sign=+1 + dollar_volume: high-dv coins end up in long basket.

    score = -(+1)*dv = -dv; rank_basket(score) → long=bottom-q=-high-dv → high-dv ✓
    """
    # High-dv coin: large close*volume; low-dv coin: tiny
    high_dv_candles = _make_candles([1000.0] * 5, volumes=[100.0] * 5)  # dv=100000
    low_dv_candles  = _make_candles([1.0] * 5, volumes=[1.0] * 5)      # dv=1

    eng = _make_engine(driver="dollar_volume", sign=1, lookback=5,
                       coins=["BTC", "ETH", "SOL", "BNB", "XRP"])

    btc_val = eng._compute_driver_value(high_dv_candles)  # 100000
    eth_val = eng._compute_driver_value(low_dv_candles)   # 1
    assert btc_val is not None
    assert eth_val is not None

    scores = {
        "BTC": -(1) * btc_val,    # -100000 (lowest) → long ✓
        "ETH": -(1) * eth_val,    # -1 (highest) → short ✓
        "SOL": -(1) * (btc_val * 0.5),
        "BNB": -(1) * (btc_val * 0.3),
        "XRP": -(1) * (eth_val * 10),
    }
    long_set, short_set = rank_basket(scores, q=0.30)
    assert "BTC" in long_set, "high-dv BTC should be longed with sign=+1"
    assert "ETH" in short_set, "low-dv ETH should be shorted with sign=+1"


# ── rank_basket reuse ─────────────────────────────────────────────────────────

def test_rank_basket_dollar_neutral():
    """rank_basket always returns equal-size long/short sets."""
    scores = {"A": 0.01, "B": -0.02, "C": 0.005, "D": 0.008, "E": -0.01}
    long_set, short_set = rank_basket(scores, q=0.30)
    assert len(long_set) == len(short_set)
    assert long_set.isdisjoint(short_set)


# ── Driver validation ─────────────────────────────────────────────────────────

def test_unknown_driver_raises():
    with pytest.raises(ValueError, match="Unknown driver"):
        XsecDriverEngine(
            executor=_FakeExecutor(), client=_FakeClient(),
            name="bad", driver="made_up_driver",
            lookback=10, q=0.3, sign=1, coins=None,
            per_leg_usd=50.0, rebalance_secs=3600,
        )


def test_supported_drivers_set():
    assert "realized_vol_carry" in SUPPORTED_DRIVERS
    assert "dollar_volume" in SUPPORTED_DRIVERS


# ── FastAPI endpoint test ─────────────────────────────────────────────────────

def test_post_xsec_instances_creates_and_persists():
    """POST /xsec/instances → 200, persists StrategyInstance, validates driver."""
    import asyncio
    from unittest.mock import MagicMock, AsyncMock
    from fastapi.testclient import TestClient

    # Build a minimal app that shares the DB setup with main's test config
    from src.api.main import app
    from src.api.database import Base, engine as db_engine
    from sqlalchemy.orm import sessionmaker

    # Create tables in the test engine (SQLite in-memory via DATABASE_URL)
    Base.metadata.create_all(bind=db_engine)

    client = TestClient(app)

    # Stub app.state so the engine doesn't try to hit the network
    app.state.executor = None   # no live executor — engine start is skipped gracefully
    app.state.client = None
    if not hasattr(app.state, "xsec_driver_engines"):
        app.state.xsec_driver_engines = {}

    # self-clean: this suite shares the app DB; drop any row left by a prior run
    # so the create assertion is robust on re-run (not a one-shot pass).
    from src.api import models
    _Session = sessionmaker(bind=db_engine)
    _db = _Session()
    _db.query(models.StrategyInstance).filter(
        models.StrategyInstance.name == "test-vol-carry").delete()
    _db.commit(); _db.close()
    app.state.xsec_driver_engines.pop("test-vol-carry", None)

    payload = {
        "name": "test-vol-carry",
        "driver": "realized_vol_carry",
        "lookback": 12,
        "q": 0.30,
        "sign": -1,
        "coins": ["BTC", "ETH", "SOL", "BNB", "XRP"],
        "per_leg_usd": 25.0,
        "timeframe": "1h",
        "rebalance_secs": 3600,
        "enabled": True,
    }
    resp = client.post("/xsec/instances", json=payload)
    assert resp.status_code == 200, f"expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["strategy_type"] == "xsec_driver"
    assert body["name"] == "test-vol-carry"
    assert body["params"]["driver"] == "realized_vol_carry"
    assert body["params"]["lookback"] == 12
    assert body["status"] == "running"


def test_post_xsec_instances_rejects_unknown_driver():
    """POST /xsec/instances with an invalid driver returns 400."""
    from fastapi.testclient import TestClient
    from src.api.main import app

    client = TestClient(app)
    resp = client.post("/xsec/instances", json={
        "name": "bad-driver-test",
        "driver": "nonexistent_driver",
    })
    assert resp.status_code == 400
    assert "driver" in resp.json()["detail"].lower()


if __name__ == "__main__":
    tests = [
        test_realized_vol_low_vol_coin,
        test_realized_vol_high_vol_coin,
        test_realized_vol_insufficient_candles_returns_none,
        test_dollar_volume_value,
        test_dollar_volume_with_varying_volume,
        test_sign_minus1_realized_vol_longs_low_vol,
        test_sign_plus1_dollar_volume_longs_high_dv,
        test_rank_basket_dollar_neutral,
        test_unknown_driver_raises,
        test_supported_drivers_set,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} pure tests passed (endpoint tests require pytest).")


def test_lifespan_sets_executor_and_client_on_app_state():
    """Regression: POST /xsec/instances starts the engine at runtime via
    app.state.executor + app.state.client. A missing app.state.client (the bug
    that shipped engines as dead DB rows) must fail here. Entering the TestClient
    context runs the real lifespan."""
    from fastapi.testclient import TestClient
    from src.api.main import app

    with TestClient(app):
        assert getattr(app.state, "executor", None) is not None, "lifespan did not set app.state.executor"
        assert getattr(app.state, "client", None) is not None, "lifespan did not set app.state.client"
        assert hasattr(app.state, "loop"), "lifespan did not capture app.state.loop"


# ── Timeframe → fetch-window tests (4h 0-trade bug) ─────────────────────────

def test_fetch_window_scales_with_timeframe(monkeypatch):
    """4h instance must request lookback bars of 4h, not 1h (live bug: window
    covered lookback/4 bars -> insufficient data -> skipped every tick)."""
    import time as _t
    from src.engine.xsec_driver_engine import BAR_MS
    captured = {}

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return []

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured.update(json["req"])
        return _Resp()

    for tf in ("1h", "4h"):
        eng = XsecDriverEngine(
            executor=_FakeExecutor(), client=_FakeClient(), name="t",
            driver="dollar_volume", lookback=96, q=0.3, sign=1,
            coins=["BTC"], per_leg_usd=25.0, rebalance_secs=3600, timeframe=tf,
        )
        monkeypatch.setattr("src.engine.xsec_driver_engine.requests.post", _fake_post)
        eng._fetch_coin_scores()
        span_ms = captured["endTime"] - captured["startTime"]
        assert span_ms == (96 + 2) * BAR_MS[tf], f"{tf}: wrong fetch window"
        assert captured["interval"] == tf


def test_unknown_timeframe_rejected():
    with pytest.raises(ValueError, match="timeframe"):
        XsecDriverEngine(
            executor=_FakeExecutor(), client=_FakeClient(), name="t",
            driver="dollar_volume", lookback=5, q=0.3, sign=1,
            coins=["BTC"], per_leg_usd=25.0, rebalance_secs=3600, timeframe="7h",
        )


# ── New driver value tests (st_reversal, amihud_illiq) ───────────────────────

def test_st_reversal_driver_value():
    """Trailing cumulative return; falling coin scores negative, rising positive."""
    eng = _make_engine(driver="st_reversal", sign=-1, lookback=5)
    falling = _make_candles([100, 98, 96, 94, 92])
    rising = _make_candles([100, 102, 104, 106, 108])
    v_fall = eng._compute_driver_value(falling)
    v_rise = eng._compute_driver_value(rising)
    assert v_fall < 0 < v_rise
    # sign=-1 -> score = +value -> rank_basket longs the bottom = the loser
    scores = {"FALL": -(-1) * v_fall, "RISE": -(-1) * v_rise,
              "A": 0.001, "B": 0.002, "C": 0.003}
    lo, sh = rank_basket(scores, 0.3)
    assert "FALL" in lo and "RISE" in sh


def test_amihud_illiq_driver_value():
    """|ret| per dollar traded: thin coin scores higher than deep coin."""
    eng = _make_engine(driver="amihud_illiq", sign=1, lookback=5)
    closes = [100, 101, 100, 101, 100]
    thin = _make_candles(closes, volumes=[1] * 5)
    deep = _make_candles(closes, volumes=[1000] * 5)
    assert eng._compute_driver_value(thin) > eng._compute_driver_value(deep)


def test_amihud_zero_volume_bars_skipped():
    eng = _make_engine(driver="amihud_illiq", sign=1, lookback=5)
    all_zero = _make_candles([100, 101, 102, 103, 104], volumes=[0] * 5)
    assert eng._compute_driver_value(all_zero) is None
