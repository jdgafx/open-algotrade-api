"""Tests for HLPSentimentGate (Gate 4.6)."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def make_gate():
    from src.services.hlp_gate import HLPSentimentGate
    return HLPSentimentGate()


def _make_position(coin: str, size: float, entry: float, upnl: float) -> dict:
    return {"position": {"coin": coin, "szi": str(size), "entryPx": str(entry), "unrealizedPnl": str(upnl)}}


def _run(coro):
    return asyncio.run(coro)


class TestHLPSentimentGate:
    def test_no_block_when_cache_empty(self):
        gate = make_gate()
        # Force cache time to infinity so no refresh attempt, cache stays empty
        gate._cache_time = float("inf")
        blocked, reason = _run(gate.should_block("BTC", is_long=True))
        assert not blocked
        assert reason == ""

    def test_blocks_when_same_direction_and_large_loss(self):
        gate = make_gate()
        gate._cache = {"BTC": {"direction": "long", "upnl_pct": -0.05, "notional": 1_000_000}}
        gate._cache_time = float("inf")
        blocked, reason = _run(gate.should_block("BTC", is_long=True))
        assert blocked
        assert "HLP long BTC" in reason

    def test_no_block_when_opposite_direction(self):
        gate = make_gate()
        gate._cache = {"BTC": {"direction": "long", "upnl_pct": -0.05, "notional": 1_000_000}}
        gate._cache_time = float("inf")
        blocked, _ = _run(gate.should_block("BTC", is_long=False))
        assert not blocked

    def test_no_block_when_same_direction_but_small_loss(self):
        gate = make_gate()
        gate._cache = {"BTC": {"direction": "long", "upnl_pct": -0.005, "notional": 1_000_000}}
        gate._cache_time = float("inf")
        blocked, _ = _run(gate.should_block("BTC", is_long=True))
        assert not blocked

    def test_no_block_when_same_direction_and_profitable(self):
        gate = make_gate()
        gate._cache = {"BTC": {"direction": "long", "upnl_pct": 0.03, "notional": 1_000_000}}
        gate._cache_time = float("inf")
        blocked, _ = _run(gate.should_block("BTC", is_long=True))
        assert not blocked

    def test_symbol_normalization(self):
        gate = make_gate()
        gate._cache = {"BTC": {"direction": "short", "upnl_pct": -0.04, "notional": 500_000}}
        gate._cache_time = float("inf")
        blocked_plain, _ = _run(gate.should_block("BTC", is_long=False))
        gate2 = make_gate()
        gate2._cache = gate._cache
        gate2._cache_time = float("inf")
        blocked_usd, _ = _run(gate2.should_block("BTC-USD", is_long=False))
        assert blocked_plain == blocked_usd == True

    def test_blocks_short_direction_against_hlp_short(self):
        gate = make_gate()
        gate._cache = {"ETH": {"direction": "short", "upnl_pct": -0.03, "notional": 2_000_000}}
        gate._cache_time = float("inf")
        blocked, reason = _run(gate.should_block("ETH", is_long=False))
        assert blocked
        assert "ETH" in reason

    def test_get_signal_returns_fade_long(self):
        gate = make_gate()
        gate._cache = {"SOL": {"direction": "long", "upnl_pct": -0.06, "notional": 300_000}}
        assert gate.get_signal("SOL") == "fade_long"

    def test_get_signal_returns_none_when_profitable(self):
        gate = make_gate()
        gate._cache = {"SOL": {"direction": "long", "upnl_pct": 0.01, "notional": 300_000}}
        assert gate.get_signal("SOL") is None

    def test_get_signal_returns_none_for_unknown_symbol(self):
        gate = make_gate()
        gate._cache = {}
        assert gate.get_signal("XYZ") is None

    def test_refresh_parses_positions(self):
        from src.services.hlp_gate import HLPSentimentGate
        mock_response = {
            "assetPositions": [
                _make_position("BTC", 10.0, 80000.0, -20000.0),  # long, -2.5%
                _make_position("ETH", -500.0, 2000.0, 5000.0),    # short, +0.5%
            ]
        }

        async def _run_refresh():
            gate = HLPSentimentGate()
            mock_resp = MagicMock()  # sync mock so .json() returns dict, not coroutine
            mock_resp.json.return_value = mock_response
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(return_value=mock_resp)
            with patch("src.services.hlp_gate.httpx.AsyncClient", return_value=mock_client):
                await gate._refresh()
            return gate

        gate = asyncio.run(_run_refresh())
        assert "BTC" in gate._cache
        assert gate._cache["BTC"]["direction"] == "long"
        assert abs(gate._cache["BTC"]["upnl_pct"] - (-0.025)) < 0.001
        assert "ETH" in gate._cache
        assert gate._cache["ETH"]["direction"] == "short"
