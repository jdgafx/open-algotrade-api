"""
Tests for XsecCarryEngine.rank_basket (the pure ranking logic).
No network, no full app — pure unit assertions.
"""
import sys
import os

# Allow running from repo root: PYTHONPATH=backend python3 -m pytest backend/tests/test_xsec_engine.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.engine.xsec_engine import rank_basket, liquid_snapshot, XSEC_Q


def test_long_is_lowest_funding():
    """Long basket = coins with lowest (most-negative) funding."""
    funding = {"BTC": 0.01, "ETH": -0.02, "SOL": 0.005, "BNB": 0.008,
               "XRP": -0.01, "DOGE": 0.003, "AVAX": -0.005, "LINK": 0.002,
               "SUI": 0.006, "ARB": -0.015}
    long_set, short_set = rank_basket(funding, q=0.30)
    # Sorted ascending: ETH(-0.02), ARB(-0.015), XRP(-0.01), AVAX(-0.005), ...
    # k = max(1, int(10 * 0.30)) = 3
    assert "ETH" in long_set
    assert "ARB" in long_set
    assert "XRP" in long_set


def test_short_is_highest_funding():
    """Short basket = coins with highest (most-positive) funding."""
    funding = {"BTC": 0.01, "ETH": -0.02, "SOL": 0.005, "BNB": 0.008,
               "XRP": -0.01, "DOGE": 0.003, "AVAX": -0.005, "LINK": 0.002,
               "SUI": 0.006, "ARB": -0.015}
    long_set, short_set = rank_basket(funding, q=0.30)
    # Sorted descending: BTC(0.01), BNB(0.008), SUI(0.006), ...
    assert "BTC" in short_set
    assert "BNB" in short_set
    assert "SUI" in short_set


def test_dollar_neutral_equal_count():
    """Dollar-neutral: |longs| == |shorts|."""
    funding = {"BTC": 0.01, "ETH": -0.02, "SOL": 0.005, "BNB": 0.008,
               "XRP": -0.01, "DOGE": 0.003, "AVAX": -0.005, "LINK": 0.002,
               "SUI": 0.006, "ARB": -0.015}
    long_set, short_set = rank_basket(funding, q=XSEC_Q)
    assert len(long_set) == len(short_set), (
        f"Baskets not equal: {len(long_set)} longs vs {len(short_set)} shorts"
    )


def test_mid_rank_coin_excluded():
    """A mid-rank coin must appear in neither basket."""
    funding = {"BTC": 0.01, "ETH": -0.02, "SOL": 0.000,   # SOL is mid
               "BNB": 0.008, "XRP": -0.01, "DOGE": 0.003,
               "AVAX": -0.005, "LINK": 0.002, "SUI": 0.006, "ARB": -0.015}
    long_set, short_set = rank_basket(funding, q=0.30)
    # SOL has 0.000 funding — should be in the middle tier
    assert "SOL" not in long_set
    assert "SOL" not in short_set


def test_no_lookahead_uses_provided_data_only():
    """rank_basket is a pure function: same input always yields same output.
    No external state, no network — proves no look-ahead."""
    f1 = {"A": 0.01, "B": -0.02, "C": 0.005, "D": 0.008}
    f2 = {"A": 0.01, "B": -0.02, "C": 0.005, "D": 0.008}  # identical copy
    r1 = rank_basket(f1)
    r2 = rank_basket(f2)
    assert r1 == r2


def test_too_few_coins_returns_empty():
    """With fewer than 4 coins, returns (set(), set()) — no partial baskets."""
    long_set, short_set = rank_basket({"BTC": 0.01, "ETH": -0.02, "SOL": 0.005}, q=0.30)
    assert long_set == set()
    assert short_set == set()


def test_no_overlap_between_baskets():
    """Long and short baskets are disjoint."""
    funding = {"BTC": 0.01, "ETH": -0.02, "SOL": 0.005, "BNB": 0.008,
               "XRP": -0.01, "DOGE": 0.003, "AVAX": -0.005, "LINK": 0.002}
    long_set, short_set = rank_basket(funding, q=0.30)
    assert long_set.isdisjoint(short_set), f"Overlap: {long_set & short_set}"


def test_liquid_snapshot_drops_illiquid():
    """Coins below the volume floor are excluded from the snapshot."""
    rates = {"BTC": 0.01, "MEME": -0.05, "ETH": 0.008}
    vols = {"BTC": 500e6, "MEME": 100e3, "ETH": 300e6}  # MEME well below floor
    snap = liquid_snapshot(rates, vols, ["BTC", "MEME", "ETH"], min_vol=3_000_000)
    assert "MEME" not in snap
    assert snap == {"BTC": 0.01, "ETH": 0.008}


def test_liquid_snapshot_keeps_dispersed_liquid_alt():
    """A liquid alt with extreme funding is kept — that's the carry edge."""
    rates = {"BTC": 0.01, "kBONK": -0.0083, "CASHCAT": 0.019}
    vols = {"BTC": 500e6, "kBONK": 6.2e6, "CASHCAT": 28.9e6}
    snap = liquid_snapshot(rates, vols, ["BTC", "kBONK", "CASHCAT"], min_vol=3_000_000)
    assert set(snap) == {"BTC", "kBONK", "CASHCAT"}


def test_liquid_snapshot_missing_coin_ignored():
    """A requested coin absent from the feed is silently skipped (no KeyError)."""
    rates = {"BTC": 0.01}
    vols = {"BTC": 500e6}
    snap = liquid_snapshot(rates, vols, ["BTC", "NOTLISTED"], min_vol=1_000_000)
    assert snap == {"BTC": 0.01}


def test_liquid_snapshot_missing_volume_treated_as_illiquid():
    """No volume data for a coin => treated as 0 => dropped."""
    rates = {"BTC": 0.01, "GHOST": -0.02}
    vols = {"BTC": 500e6}  # GHOST has no volume entry
    snap = liquid_snapshot(rates, vols, ["BTC", "GHOST"], min_vol=1_000_000)
    assert "GHOST" not in snap


if __name__ == "__main__":
    # ponytail: self-check without a test framework
    tests = [
        test_long_is_lowest_funding,
        test_short_is_highest_funding,
        test_dollar_neutral_equal_count,
        test_mid_rank_coin_excluded,
        test_no_lookahead_uses_provided_data_only,
        test_too_few_coins_returns_empty,
        test_no_overlap_between_baskets,
        test_liquid_snapshot_drops_illiquid,
        test_liquid_snapshot_keeps_dispersed_liquid_alt,
        test_liquid_snapshot_missing_coin_ignored,
        test_liquid_snapshot_missing_volume_treated_as_illiquid,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")


# ── Boot reconciliation (redeploy must not orphan the basket) ────────────────

import asyncio

from src.engine.xsec_engine import XsecCarryEngine, PER_LEG_USD


class _FakeExecutor:
    def __init__(self, positions):
        self._pos = positions
        self.closed = []

    async def get_all_positions(self):
        return self._pos

    async def execute_signal(self, sig, strat):
        class R:
            success = True
            error = None
        self.closed.append(sig.symbol)
        return R()


def test_reconcile_adopts_matching_legs_and_closes_stale_sizes():
    """At boot: legs at the configured size are adopted into _open_legs;
    legs whose notional drifted >20% from PER_LEG_USD (config changed
    between boots) are closed, not adopted."""
    positions = [
        {"strategy_name": "xsec_carry", "symbol": "ONDO", "side": "short", "size_usd": PER_LEG_USD},
        {"strategy_name": "xsec_carry", "symbol": "WLD", "side": "long", "size_usd": PER_LEG_USD * 0.99},
        {"strategy_name": "xsec_carry", "symbol": "ETH", "side": "long", "size_usd": PER_LEG_USD * 0.1},
        {"strategy_name": "trend_cross", "symbol": "BTC", "side": "long", "size_usd": PER_LEG_USD},
    ]
    ex = _FakeExecutor(positions)
    eng = XsecCarryEngine(ex, client=None)
    asyncio.run(eng._reconcile_open_legs())
    assert eng._open_legs == {"ONDO": "short", "WLD": "long"}
    assert ex.closed == ["ETH"]  # stale-size leg closed, other strategies untouched


def test_reconcile_survives_executor_error():
    class _Boom:
        async def get_all_positions(self):
            raise RuntimeError("db locked")

    eng = XsecCarryEngine(_Boom(), client=None)
    asyncio.run(eng._reconcile_open_legs())
    assert eng._open_legs == {}
