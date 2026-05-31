"""
Tests for the FastAPI application endpoints.

Tests cover:
- Health check with registry verification
- Trading mode endpoints
- Strategy CRUD (create, read, update, delete)
- Strategy registry listing
- Vault status and operations
- User endpoints
- Dashboard stats and performance
- Portfolio summary and allocation
- Aggregate strategy performance
- Strategy trades and signals
- Market overview
- Paper trading stats
- Batch deploy
- Error cases (404s, 400s, validation)
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from src.api import models


# ──────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "open-algotrade-api"
        assert data["version"] == "2.0.0"
        assert "trading_mode" in data
        assert "registry_strategies" in data
        assert "registry_ok" in data
        assert "orchestrator_ok" in data

    def test_health_has_registry_count(self, client):
        resp = client.get("/health")
        data = resp.json()
        # Registry should find strategies even without orchestrator
        assert isinstance(data["registry_strategies"], int)
        assert data["registry_strategies"] >= 0


# ──────────────────────────────────────────────
# Trading Mode
# ──────────────────────────────────────────────


class TestTradingMode:
    def test_get_trading_mode(self, client):
        resp = client.get("/trading-mode")
        assert resp.status_code == 200
        data = resp.json()
        assert "mode" in data
        assert "available_modes" in data
        assert set(data["available_modes"]) == {"paper", "testnet", "mainnet"}


# ──────────────────────────────────────────────
# Strategy Registry
# ──────────────────────────────────────────────


class TestStrategyRegistry:
    def test_list_registry(self, client):
        resp = client.get("/strategies/registry")
        assert resp.status_code == 200
        data = resp.json()
        assert "available_strategies" in data
        assert "total" in data
        assert isinstance(data["total"], int)
        assert data["total"] > 0

    def test_registry_items_have_required_fields(self, client):
        resp = client.get("/strategies/registry")
        data = resp.json()
        for s in data["available_strategies"]:
            assert "strategy_type" in s
            assert "tier" in s
            assert "description" in s
            assert "default_symbol" in s
            assert "default_timeframe" in s
            assert "default_params" in s


# ──────────────────────────────────────────────
# Strategy CRUD
# ──────────────────────────────────────────────


class TestStrategyCRUD:
    def test_list_strategies_empty(self, client):
        resp = client.get("/strategies")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_create_strategy(self, client):
        payload = {
            "name": "test-rsi-btc",
            "strategy_type": "rsi",
            "symbol": "BTC",
            "timeframe": "1h",
            "leverage": 2,
            "size_usd": 500.0,
            "target_pct": 5.0,
            "max_loss_pct": -10.0,
        }
        resp = client.post("/strategies", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "test-rsi-btc"
        assert data["strategy_type"] == "rsi"
        assert data["symbol"] == "BTC"
        assert data["status"] == "stopped"
        assert data["size_usd"] == 500.0

    def test_create_strategy_duplicate_name(self, client):
        payload = {
            "name": "dup-test",
            "strategy_type": "rsi",
        }
        resp1 = client.post("/strategies", json=payload)
        assert resp1.status_code == 200

        resp2 = client.post("/strategies", json=payload)
        assert resp2.status_code == 400
        assert "already exists" in resp2.json()["detail"]

    def test_create_strategy_invalid_type(self, client):
        payload = {
            "name": "bad-type",
            "strategy_type": "nonexistent_strategy_xyz",
        }
        resp = client.post("/strategies", json=payload)
        assert resp.status_code == 400
        assert "Unknown strategy type" in resp.json()["detail"]

    def test_get_strategy_by_name(self, client):
        client.post("/strategies", json={
            "name": "get-test",
            "strategy_type": "sma_crossover",
            "symbol": "ETH",
        })
        resp = client.get("/strategies/get-test")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "get-test"
        assert data["strategy_type"] == "sma_crossover"
        assert data["symbol"] == "ETH"

    def test_get_strategy_not_found(self, client):
        resp = client.get("/strategies/does-not-exist")
        assert resp.status_code == 404

    def test_update_strategy(self, client):
        client.post("/strategies", json={
            "name": "update-test",
            "strategy_type": "rsi",
            "size_usd": 100.0,
        })
        resp = client.patch("/strategies/update-test", json={
            "size_usd": 250.0,
            "leverage": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["size_usd"] == 250.0
        assert data["leverage"] == 5

    def test_update_strategy_not_found(self, client):
        resp = client.patch("/strategies/no-such", json={"size_usd": 100.0})
        assert resp.status_code == 404

    def test_patch_hot_applies_to_running_strategy(self, client, make_strategy_config):
        """Wiring tracer: PATCHing a *running* strategy must reach the live in-memory
        config (not just the DB row) and the response reports what was hot-applied.
        Closes the DB-only write-back gap."""
        from unittest.mock import MagicMock

        from src.engine.orchestrator import StrategyOrchestrator

        client.post("/strategies", json={
            "name": "hot-test", "strategy_type": "rsi", "size_usd": 100.0,
        })

        orch = StrategyOrchestrator(client=MagicMock(), executor=MagicMock())
        orch.executor._positions = {}
        cfg = make_strategy_config(name="hot-test", size_usd=100.0)
        strat = MagicMock()
        strat.config = cfg
        orch._strategies["hot-test"] = strat
        client.app.state.orchestrator = orch

        resp = client.patch("/strategies/hot-test", json={"size_usd": 250.0})

        assert resp.status_code == 200
        data = resp.json()
        assert data["size_usd"] == 250.0                      # DB row updated (existing behavior)
        assert cfg.size_usd == 250.0                          # LIVE in-memory config updated (new)
        assert "size_usd" in data["live_update"]["applied"]   # API reports the hot-apply

    def test_patch_defers_exit_threshold_while_position_open(self, client, make_strategy_config):
        """Safety: tightening max_loss_pct via PATCH while a position is open must DEFER
        the live change (DB still updated) so should_exit can't force-close the open leg."""
        from unittest.mock import MagicMock

        from src.engine.orchestrator import StrategyOrchestrator

        client.post("/strategies", json={
            "name": "open-pos", "strategy_type": "rsi", "max_loss_pct": -8.0,
        })

        orch = StrategyOrchestrator(client=MagicMock(), executor=MagicMock())
        cfg = make_strategy_config(name="open-pos", max_loss_pct=-8.0)
        strat = MagicMock()
        strat.config = cfg
        orch._strategies["open-pos"] = strat
        pos = MagicMock()
        pos.strategy_name = "open-pos"
        orch.executor._positions = {"open-pos:BTC": pos}
        client.app.state.orchestrator = orch

        resp = client.patch("/strategies/open-pos", json={"max_loss_pct": -3.0})

        assert resp.status_code == 200
        data = resp.json()
        assert data["max_loss_pct"] == -3.0                        # DB row updated
        assert cfg.max_loss_pct == -8.0                            # LIVE config unchanged (deferred)
        assert "max_loss_pct" in data["live_update"]["deferred"]

    def test_patch_reports_restart_required_fields(self, client, make_strategy_config):
        """Loop-local fields (interval_seconds/symbol/timeframe) are persisted but
        reported as restart_required rather than silently hot-applied."""
        from unittest.mock import MagicMock

        from src.engine.orchestrator import StrategyOrchestrator

        client.post("/strategies", json={
            "name": "restarty", "strategy_type": "rsi", "interval_seconds": 30,
        })

        orch = StrategyOrchestrator(client=MagicMock(), executor=MagicMock())
        orch.executor._positions = {}
        cfg = make_strategy_config(name="restarty", interval_seconds=30)
        strat = MagicMock()
        strat.config = cfg
        orch._strategies["restarty"] = strat
        client.app.state.orchestrator = orch

        resp = client.patch("/strategies/restarty", json={"interval_seconds": 60})

        assert resp.status_code == 200
        data = resp.json()
        assert data["interval_seconds"] == 60                                  # DB updated
        assert cfg.interval_seconds == 30                                      # live unchanged
        assert "interval_seconds" in data["live_update"]["restart_required"]

    def test_delete_strategy(self, client):
        client.post("/strategies", json={
            "name": "delete-test",
            "strategy_type": "rsi",
        })
        resp = client.delete("/strategies/delete-test")
        assert resp.status_code == 200

        # Verify it's gone
        resp2 = client.get("/strategies/delete-test")
        assert resp2.status_code == 404

    def test_delete_strategy_not_found(self, client):
        resp = client.delete("/strategies/nonexistent")
        assert resp.status_code == 404

    def test_list_strategies_after_create(self, client):
        client.post("/strategies", json={
            "name": "list-test-a",
            "strategy_type": "rsi",
        })
        client.post("/strategies", json={
            "name": "list-test-b",
            "strategy_type": "macd",
        })
        resp = client.get("/strategies")
        assert resp.status_code == 200
        names = [s["name"] for s in resp.json()]
        assert "list-test-a" in names
        assert "list-test-b" in names


# ──────────────────────────────────────────────
# Vault
# ──────────────────────────────────────────────


class TestVaultEndpoints:
    def test_vault_status(self, client):
        resp = client.get("/vault/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_equity" in data
        assert "total_shares" in data
        assert "nav_per_share" in data

    def test_vault_history(self, client):
        resp = client.get("/vault/history?days=7")
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        assert "total_points" in data
        assert data["total_points"] == 7
        assert len(data["data"]) == 7
        for point in data["data"]:
            assert "timestamp" in point
            assert "equity" in point
            assert "nav_per_share" in point


# ──────────────────────────────────────────────
# Users
# ──────────────────────────────────────────────


class TestUserEndpoints:
    def test_create_user(self, client):
        resp = client.post("/users", json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "password123",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "testuser"
        assert data["email"] == "test@example.com"
        assert data["balance"] == 0.0
        assert data["shares"] == 0.0

    def test_create_user_duplicate(self, client):
        client.post("/users", json={
            "username": "dupuser",
            "email": "dup@example.com",
            "password": "password123",
        })
        resp = client.post("/users", json={
            "username": "dupuser",
            "email": "dup2@example.com",
            "password": "password123",
        })
        assert resp.status_code == 400

    def test_get_user(self, client):
        create_resp = client.post("/users", json={
            "username": "getuser",
            "email": "get@example.com",
            "password": "password123",
        })
        user_id = create_resp.json()["id"]
        resp = client.get(f"/users/{user_id}")
        assert resp.status_code == 200
        assert resp.json()["username"] == "getuser"

    def test_get_user_not_found(self, client):
        resp = client.get("/users/99999")
        assert resp.status_code == 404


# ──────────────────────────────────────────────
# Deposits & Withdrawals
# ──────────────────────────────────────────────


class TestDepositsWithdrawals:
    def _create_user(self, client):
        resp = client.post("/users", json={
            "username": "vaultuser",
            "email": "vault@example.com",
            "password": "password123",
        })
        return resp.json()["id"]

    def test_deposit(self, client):
        user_id = self._create_user(client)
        resp = client.post("/deposit", json={
            "user_id": user_id,
            "amount": 1000.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["balance"] == 1000.0
        assert data["shares"] > 0

    def test_withdraw(self, client):
        user_id = self._create_user(client)
        # Deposit first
        client.post("/deposit", json={
            "user_id": user_id,
            "amount": 1000.0,
        })
        user = client.get(f"/users/{user_id}").json()
        shares = user["shares"]

        resp = client.post("/withdraw", json={
            "user_id": user_id,
            "shares_to_redeem": shares / 2,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["shares_burned"] == shares / 2
        assert data["amount"] > 0

    def test_withdraw_insufficient_shares(self, client):
        user_id = self._create_user(client)
        resp = client.post("/withdraw", json={
            "user_id": user_id,
            "shares_to_redeem": 99999.0,
        })
        assert resp.status_code == 400
        assert "Insufficient shares" in resp.json()["detail"]

    def test_get_deposits(self, client):
        user_id = self._create_user(client)
        client.post("/deposit", json={
            "user_id": user_id,
            "amount": 500.0,
        })
        resp = client.get(f"/deposits/{user_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["amount"] == 500.0


# ──────────────────────────────────────────────
# Portfolio (user-level)
# ──────────────────────────────────────────────


class TestPortfolioUser:
    def test_get_portfolio(self, client):
        resp = client.post("/users", json={
            "username": "portuser",
            "email": "port@example.com",
            "password": "pass",
        })
        user_id = resp.json()["id"]
        client.post("/deposit", json={"user_id": user_id, "amount": 5000.0})

        resp = client.get(f"/portfolio/{user_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "portuser"
        assert data["portfolio_value"] > 0
        assert data["nav_per_share"] > 0

    def test_get_portfolio_not_found(self, client):
        resp = client.get("/portfolio/99999")
        assert resp.status_code == 404


# ──────────────────────────────────────────────
# Dashboard
# ──────────────────────────────────────────────


class TestDashboard:
    def test_dashboard_stats(self, client):
        resp = client.get("/dashboard/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_strategies" in data
        assert "running_strategies" in data
        assert "total_pnl" in data
        assert "total_trades" in data
        assert "win_rate" in data
        assert "active_positions" in data

    def test_dashboard_performance(self, client):
        resp = client.get("/dashboard/performance")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_pnl" in data
        assert "win_rate" in data
        assert "total_trades" in data
        assert "max_drawdown" in data
        assert "equity_curve" in data
        assert "strategy_breakdown" in data
        assert isinstance(data["equity_curve"], list)
        assert isinstance(data["strategy_breakdown"], list)


# ──────────────────────────────────────────────
# Aggregate Strategy Performance (new endpoint)
# ──────────────────────────────────────────────


class TestAggregatePerformance:
    def test_performance_empty(self, client):
        resp = client.get("/strategies/performance")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_strategies"] == 0
        assert data["total_pnl"] == 0
        assert data["total_trades"] == 0
        assert data["overall_win_rate"] == 0
        assert data["strategies"] == []

    def test_performance_with_strategies(self, client):
        client.post("/strategies", json={
            "name": "perf-a",
            "strategy_type": "rsi",
            "symbol": "BTC",
        })
        client.post("/strategies", json={
            "name": "perf-b",
            "strategy_type": "macd",
            "symbol": "ETH",
        })

        resp = client.get("/strategies/performance")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_strategies"] == 2
        assert len(data["strategies"]) == 2
        names = [s["name"] for s in data["strategies"]]
        assert "perf-a" in names
        assert "perf-b" in names

        for s in data["strategies"]:
            assert "strategy_type" in s
            assert "symbol" in s
            assert "total_pnl" in s
            assert "total_trades" in s
            assert "win_rate" in s
            assert "avg_trade_pnl" in s
            assert "profit_factor" in s


# ──────────────────────────────────────────────
# Strategy Trades (new endpoint)
# ──────────────────────────────────────────────


class TestStrategyTrades:
    def test_trades_not_found(self, client):
        resp = client.get("/strategies/nonexistent/trades")
        assert resp.status_code == 404

    def test_trades_empty(self, client):
        client.post("/strategies", json={
            "name": "trades-test",
            "strategy_type": "rsi",
        })
        resp = client.get("/strategies/trades-test/trades")
        assert resp.status_code == 200
        data = resp.json()
        assert data["strategy_name"] == "trades-test"
        assert data["total"] == 0
        assert data["trades"] == []


# ──────────────────────────────────────────────
# Strategy Signals (new endpoint)
# ──────────────────────────────────────────────


class TestStrategySignals:
    def test_signals_not_found(self, client):
        resp = client.get("/strategies/nonexistent/signals")
        assert resp.status_code == 404

    def test_signals_empty(self, client):
        client.post("/strategies", json={
            "name": "signals-test",
            "strategy_type": "rsi",
        })
        resp = client.get("/strategies/signals-test/signals")
        assert resp.status_code == 200
        data = resp.json()
        assert data["strategy_name"] == "signals-test"
        assert data["total"] == 0
        assert data["signals"] == []


# ──────────────────────────────────────────────
# Portfolio Summary (new endpoint)
# ──────────────────────────────────────────────


class TestPortfolioSummary:
    def test_portfolio_summary(self, client):
        resp = client.get("/portfolio/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "trading_mode" in data
        assert "total_equity" in data
        assert "initial_equity" in data
        assert "total_pnl" in data
        assert "total_return_pct" in data
        assert "max_drawdown_pct" in data
        assert "total_strategies" in data
        assert "running_strategies" in data
        assert "total_trades" in data
        assert "win_rate" in data
        assert "active_positions" in data
        assert "total_exposure_usd" in data

    def test_portfolio_summary_with_strategies(self, client):
        client.post("/strategies", json={
            "name": "port-sum-a",
            "strategy_type": "rsi",
            "size_usd": 200.0,
        })
        client.post("/strategies", json={
            "name": "port-sum-b",
            "strategy_type": "macd",
            "size_usd": 300.0,
        })
        resp = client.get("/portfolio/summary")
        data = resp.json()
        assert data["total_strategies"] == 2


# ──────────────────────────────────────────────
# Portfolio Allocation (new endpoint)
# ──────────────────────────────────────────────


class TestPortfolioAllocation:
    def test_allocation_empty(self, client):
        resp = client.get("/portfolio/allocation")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_allocated_usd"] == 0
        assert data["strategies"] == []

    def test_allocation_with_strategies(self, client):
        client.post("/strategies", json={
            "name": "alloc-a",
            "strategy_type": "rsi",
            "size_usd": 400.0,
        })
        client.post("/strategies", json={
            "name": "alloc-b",
            "strategy_type": "macd",
            "size_usd": 600.0,
        })
        resp = client.get("/portfolio/allocation")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_allocated_usd"] == 1000.0
        assert len(data["strategies"]) == 2

        for item in data["strategies"]:
            assert "strategy_name" in item
            assert "strategy_type" in item
            assert "symbol" in item
            assert "size_usd" in item
            assert "allocation_pct" in item
            assert "current_pnl" in item

        # Verify allocation percentages sum to 100
        total_pct = sum(s["allocation_pct"] for s in data["strategies"])
        assert abs(total_pct - 100.0) < 0.5


# ──────────────────────────────────────────────
# Market Overview (new endpoint)
# ──────────────────────────────────────────────


class TestMarketOverview:
    def test_market_overview_default_symbols(self, client):
        resp = client.get("/market/overview")
        assert resp.status_code == 200
        data = resp.json()
        assert "timestamp" in data
        assert "symbols" in data
        symbols = [s["symbol"] for s in data["symbols"]]
        assert "BTC" in symbols
        assert "ETH" in symbols
        assert "SOL" in symbols

    def test_market_overview_custom_symbols(self, client):
        resp = client.get("/market/overview?symbols=BTC,DOGE")
        assert resp.status_code == 200
        data = resp.json()
        symbols = [s["symbol"] for s in data["symbols"]]
        assert "BTC" in symbols
        assert "DOGE" in symbols
        assert len(symbols) == 2

    def test_market_overview_item_fields(self, client):
        resp = client.get("/market/overview?symbols=BTC")
        data = resp.json()
        item = data["symbols"][0]
        assert "symbol" in item
        assert "regime" in item
        assert "regime_confidence" in item
        assert "volatility_regime" in item
        assert "is_volatile" in item
        assert "recommended_strategies" in item


# ──────────────────────────────────────────────
# Batch Deploy
# ──────────────────────────────────────────────


class TestBatchDeploy:
    def test_batch_deploy(self, client):
        payload = {
            "strategies": [
                {"name": "batch-1", "strategy_type": "rsi", "symbol": "BTC"},
                {"name": "batch-2", "strategy_type": "macd", "symbol": "ETH"},
            ]
        }
        resp = client.post("/strategies/deploy-batch", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["results"]) == 2

    def test_batch_deploy_invalid_type(self, client):
        payload = {
            "strategies": [
                {"name": "batch-bad", "strategy_type": "invalid_xyz_123"},
            ]
        }
        resp = client.post("/strategies/deploy-batch", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        # Should report the failure in results
        assert data["failed"] >= 1


# ──────────────────────────────────────────────
# Trades & Positions (legacy)
# ──────────────────────────────────────────────


class TestTradesPositions:
    def test_get_trades_empty(self, client):
        resp = client.get("/trades")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_positions(self, client):
        resp = client.get("/positions")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


# ──────────────────────────────────────────────
# Legacy Strategy Endpoints
# ──────────────────────────────────────────────


class TestLegacyStrategy:
    def test_legacy_status(self, client):
        resp = client.get("/strategy/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "symbol" in data

    def test_legacy_start_stop(self, client):
        start_payload = {
            "symbol": "BTC",
            "timeframe": "1h",
            "lookback_period": 55,
            "atr_period": 20,
            "atr_multiplier": 2.0,
            "leverage": 5,
        }
        resp = client.post("/strategy/start", json=start_payload)
        assert resp.status_code == 200
        assert resp.json()["status"] == "running"

        resp = client.post("/strategy/stop")
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"


# ──────────────────────────────────────────────
# Paper Trading Stats
# ──────────────────────────────────────────────


class TestPaperTradingAPI:
    def test_paper_stats(self, client):
        resp = client.get("/paper/stats")
        # Returns 400 when no executor is configured, 200 with data
        assert resp.status_code in (200, 400, 503)

    def test_paper_positions(self, client):
        resp = client.get("/paper/positions")
        assert resp.status_code in (200, 400, 503)

    def test_paper_trades(self, client):
        resp = client.get("/paper/trades")
        assert resp.status_code in (200, 400, 503)


# ──────────────────────────────────────────────
# Auth Endpoints
# ──────────────────────────────────────────────


class TestAuth:
    def test_register_and_sign_in(self, client):
        # Register
        resp = client.post("/auth/register", json={
            "username": "authuser",
            "email": "auth@example.com",
            "password": "securepassword",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "authuser"
        assert data["email"] == "auth@example.com"
        assert "access_token" in data

        # Sign in
        resp = client.post("/auth/sign-in", json={
            "email": "auth@example.com",
            "password": "securepassword",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "authuser"
        assert "access_token" in data

    def test_sign_in_invalid_credentials(self, client):
        resp = client.post("/auth/sign-in", json={
            "email": "nobody@example.com",
            "password": "wrong",
        })
        assert resp.status_code == 401

    def test_register_duplicate_email(self, client):
        client.post("/auth/register", json={
            "username": "first",
            "email": "same@example.com",
            "password": "pass123",
        })
        resp = client.post("/auth/register", json={
            "username": "second",
            "email": "same@example.com",
            "password": "pass123",
        })
        assert resp.status_code == 400
        assert "Email already registered" in resp.json()["detail"]
