def test_rbi_routes_import():
    from src.api.routes.rbi_optimize import router, llm_router
    assert router is not None
    assert llm_router is not None


def test_trigger_lookback_default_not_below_engine_default():
    """The trigger route default must not starve sparse-trigger strategies.
    14d gave 0/30 configs >=5 in-sample trades -> no_candidates; 90d gave 26/30.
    Keep the route default aligned with OptimizationEngine.optimize()'s own 90d
    so the two can't silently drift apart again."""
    from src.api.routes.rbi_optimize import TriggerRequest
    from src.engine.optimizer import OptimizationEngine
    import inspect

    route_default = TriggerRequest.model_fields["lookback_days"].default
    engine_default = inspect.signature(OptimizationEngine.optimize).parameters["lookback_days"].default
    assert route_default >= engine_default, (
        f"trigger lookback default {route_default} < engine default {engine_default} "
        "-> sparse strategies will return no_candidates"
    )
