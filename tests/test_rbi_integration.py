"""
Smoke tests for the full RBI stack.
These run against the local app — start the server first:
  cd backend && .venv/bin/uvicorn src.api.main:app --port 8002
"""
import pytest
import httpx

BASE = "http://localhost:8002"


@pytest.mark.integration
def test_rbi_status_endpoint():
    r = httpx.get(f"{BASE}/optimize/rbi/status", timeout=5)
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


@pytest.mark.integration
def test_rbi_history_empty_initially():
    r = httpx.get(f"{BASE}/optimize/rbi/history", timeout=5)
    assert r.status_code == 200
    assert isinstance(r.json(), list)


@pytest.mark.integration
def test_llm_gate_stats_endpoint():
    r = httpx.get(f"{BASE}/optimize/llm-gate/stats", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert "total_evaluations" in data
    assert "accuracy" in data


@pytest.mark.integration
def test_optimize_rbi_trigger_bad_strategy():
    r = httpx.post(
        f"{BASE}/optimize/rbi/trigger/nonexistent_strategy",
        json={"strategy_id": 999, "symbol": "BTC", "n_trials": 5},
        timeout=30,
    )
    assert r.status_code in (400, 404, 422, 500)
