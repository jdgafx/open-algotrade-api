"""Tests for src/services/trade_memory.py — Supermemory integration."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_store_trade_skips_when_no_api_key(monkeypatch):
    """store_trade exits early without hitting the network when key is absent."""
    monkeypatch.delenv("SUPERMEMORY_API_KEY", raising=False)
    # If no key, function must return without raising
    from src.services.trade_memory import store_trade
    await store_trade("conspop-btc", "BTC", "LONG", 12.5, "trending", {}, 65000.0)


@pytest.mark.asyncio
async def test_recall_trades_returns_empty_when_no_api_key(monkeypatch):
    """recall_trades returns empty string when key is absent."""
    monkeypatch.delenv("SUPERMEMORY_API_KEY", raising=False)
    from src.services.trade_memory import recall_trades
    result = await recall_trades("conspop-btc", "BTC", "trending")
    assert result == ""


@pytest.mark.asyncio
async def test_store_trade_posts_correct_payload(monkeypatch):
    """store_trade builds a content string with strategy, symbol, PnL, and regime."""
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "sm_test_key")

    posted = {}

    async def fake_post(url, headers, json, **kwargs):
        posted["url"] = url
        posted["json"] = json
        resp = MagicMock()
        resp.status_code = 201
        return resp

    with patch("src.services.trade_memory.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=fake_post)
        mock_client_cls.return_value = mock_client

        from src.services.trade_memory import store_trade
        await store_trade(
            strategy="conspop-btc",
            symbol="BTC",
            signal="LONG",
            pnl=45.92,
            regime="trending",
            params={"cooldown_seconds": 120, "max_trades_per_hour": 6},
            price=65000.0,
        )

    assert "/v3/documents" in posted["url"]
    content = posted["json"]["content"]
    assert "conspop-btc" in content
    assert "WIN" in content
    assert "45.92" in content
    assert "trading-bot" in posted["json"]["tags"]


@pytest.mark.asyncio
async def test_recall_trades_returns_content(monkeypatch):
    """recall_trades parses Supermemory v3 search response correctly."""
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "sm_test_key")

    fake_results = {
        "results": [
            {"content": "[TRADE] 2026-05-15 | conspop-btc | LONG BTC | PnL: $45.92 (WIN)"},
            {"content": "[TRADE] 2026-05-14 | conspop-btc | SHORT BTC | PnL: $-5.10 (LOSS)"},
        ]
    }

    async def fake_post(url, headers, json, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.json = MagicMock(return_value=fake_results)
        return resp

    with patch("src.services.trade_memory.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=fake_post)
        mock_client_cls.return_value = mock_client

        from src.services.trade_memory import recall_trades
        result = await recall_trades("conspop-btc", "BTC", "trending")

    assert "WIN" in result
    assert "LOSS" in result
    assert "conspop-btc" in result
