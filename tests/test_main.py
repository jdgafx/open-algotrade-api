"""Smoke tests for FastAPI app configuration and router registration."""
from fastapi.testclient import TestClient


def test_app_has_rbi_optimize_routes():
    from src.api.main import app
    routes = [r.path for r in app.routes]
    assert any("/optimize/rbi" in p for p in routes), f"RBI optimize routes missing from {routes}"


def test_app_has_llm_gate_route():
    from src.api.main import app
    routes = [r.path for r in app.routes]
    assert any("/optimize/llm-gate" in p for p in routes), f"LLM gate route missing from {routes}"
