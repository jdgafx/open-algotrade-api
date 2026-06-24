"""U6 (R5) — confidence-governed per-winner allocation cap.

_winner_cap_usd scales a winner's allocation ceiling with its EARNED live
confidence: observation-small at thin evidence, reaching WINNER_MAX_PCT only as
confidence approaches 1 (~30 live trades). Replaces the old flat 0.75-of-balance.
"""

import pytest

from src.api.main import _winner_cap_usd, WINNER_OBS_PCT, WINNER_MAX_PCT


def test_zero_confidence_caps_at_observation_floor():
    # A brand-new winner (no earned confidence) gets the observation-small share.
    assert _winner_cap_usd(10000.0, 0.0) == pytest.approx(10000.0 * WINNER_OBS_PCT)


def test_full_confidence_caps_at_ceiling():
    # ~30 live trades of edge → the full ceiling.
    assert _winner_cap_usd(10000.0, 1.0) == pytest.approx(10000.0 * WINNER_MAX_PCT)


def test_thin_evidence_gets_far_less_than_the_old_flat_cap():
    """A 6-trade winner (confidence ~0.2) is held well below the old flat 0.75 —
    the over-allocation of thin evidence that U6 closes."""
    cap = _winner_cap_usd(10000.0, 0.2)
    expected_pct = WINNER_OBS_PCT + (WINNER_MAX_PCT - WINNER_OBS_PCT) * 0.2
    assert cap == pytest.approx(10000.0 * expected_pct)
    assert cap < 10000.0 * WINNER_MAX_PCT          # much less than the old flat cap
    assert cap > 10000.0 * WINNER_OBS_PCT          # but above the floor


def test_confidence_is_monotonic():
    balance = 10000.0
    caps = [_winner_cap_usd(balance, c) for c in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert caps == sorted(caps)                    # more confidence -> more capital
    assert caps[0] < caps[-1]


def test_confidence_is_clamped():
    # out-of-range confidence cannot push allocation past the ceiling or below floor.
    assert _winner_cap_usd(10000.0, 5.0) == pytest.approx(10000.0 * WINNER_MAX_PCT)
    assert _winner_cap_usd(10000.0, -1.0) == pytest.approx(10000.0 * WINNER_OBS_PCT)
