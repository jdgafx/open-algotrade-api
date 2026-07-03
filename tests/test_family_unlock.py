"""Family-pooled unlock for the >$500 compounding governor (2026-07-03).

The 100-trade bar is unchanged — it is just measurable family-wide, so
instances of the same strategy_type pool their completed-trade evidence.
Guardrails: family must be net positive, and the instance must carry its
own positive sample so a losing sibling cannot ride the family's winners.
"""
from src.api.main import _family_unlock_ok


def test_pooled_family_unlocks():
    # conspop scenario: own 32 trades +$1.78, family (btc+sol+...) 100 trades +$8.7
    assert _family_unlock_ok(32, 1.78, 100, 8.7, fam_min=100)


def test_family_below_bar_stays_capped():
    assert not _family_unlock_ok(32, 1.78, 99, 8.7, fam_min=100)


def test_bleeding_family_never_unlocks():
    assert not _family_unlock_ok(50, 5.0, 150, -0.01, fam_min=100)


def test_losing_sibling_cannot_ride_family():
    # family proven, but this instance is net negative -> capped
    assert not _family_unlock_ok(20, -0.5, 150, 40.0, fam_min=100)


def test_thin_own_sample_cannot_ride_family():
    # 4-trade instance may not ride 150 family trades (own_min default 10)
    assert not _family_unlock_ok(4, 2.0, 150, 40.0, fam_min=100)


def test_own_100_trades_path_unchanged():
    # per-instance rule still reachable purely through the family pool when
    # the instance IS the family (solo instance, 100 own trades, positive)
    assert _family_unlock_ok(100, 3.0, 100, 3.0, fam_min=100)
