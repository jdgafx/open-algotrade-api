def test_rbi_routes_import():
    from src.api.routes.rbi_optimize import router, llm_router
    assert router is not None
    assert llm_router is not None
