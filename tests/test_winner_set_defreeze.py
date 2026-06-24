"""U6 (R10 / F3) — de-freeze the static winner set.

_defrosted_winner_set keeps a static winner protected UNLESS it shows a sustained
recent live loss, in which case it is demotable; dynamic (live-proven) winners
are always promoted in. Thin recent data keeps a name protected (safe degradation).
"""

from src.api.main import _defrosted_winner_set, DEFREEZE_MIN_RECENT_TRADES

_STATICS = {"flip-flop-btc", "vwap-btc", "liqdip-btc"}


def _recent(mapping):
    """recent_pnl_fn from a {name: (recent_pnl, recent_trades)} mapping; unknown
    names report no data."""
    return lambda name: mapping.get(name, (0.0, 0))


def test_sustained_loser_is_demoted():
    """A static winner with enough recent trades AND negative recent PnL drops
    out of protection — the frozen-bleeder demotion U6 enables."""
    fn = _recent({"flip-flop-btc": (-50.0, DEFREEZE_MIN_RECENT_TRADES)})
    result = _defrosted_winner_set(_STATICS, set(), fn)
    assert "flip-flop-btc" not in result
    assert "vwap-btc" in result and "liqdip-btc" in result   # others (no data) stay protected


def test_thin_recent_data_keeps_protection():
    """Below the recent-trade floor (fresh strategy or post-redeploy window) a
    losing reading does NOT demote — safe degradation, mirrors F1."""
    fn = _recent({"flip-flop-btc": (-50.0, DEFREEZE_MIN_RECENT_TRADES - 1)})
    assert "flip-flop-btc" in _defrosted_winner_set(_STATICS, set(), fn)


def test_recent_winner_stays_protected():
    fn = _recent({"flip-flop-btc": (+30.0, DEFREEZE_MIN_RECENT_TRADES + 5)})
    assert "flip-flop-btc" in _defrosted_winner_set(_STATICS, set(), fn)


def test_dynamic_winner_is_promoted_in():
    """A new live-proven strategy (dynamic winner) joins the protected set even
    though it is not a static name — the promotion-INTO-capital side of de-freeze."""
    result = _defrosted_winner_set(_STATICS, {"newcomer-sol"}, _recent({}))
    assert "newcomer-sol" in result


def test_unassessable_name_stays_protected():
    """If recent_pnl_fn raises, the name is kept protected (never demote on an
    error — fail safe)."""
    def _boom(_name):
        raise RuntimeError("executor unavailable")
    assert _defrosted_winner_set(_STATICS, set(), _boom) == _STATICS
