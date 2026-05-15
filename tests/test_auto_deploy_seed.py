"""Tests for _auto_deploy_winners winner-list hygiene."""
import ast
from pathlib import Path


def _winners_list_strategy_types() -> set[str]:
    """Parse src/api/main.py and extract strategy_type values from the
    _auto_deploy_winners 'winners' list without importing the module."""
    src = Path(__file__).resolve().parent.parent / "src" / "api" / "main.py"
    tree = ast.parse(src.read_text())

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "winners":
                    if isinstance(node.value, ast.List):
                        types: set[str] = set()
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Dict):
                                for k, v in zip(elt.keys, elt.values):
                                    if (
                                        isinstance(k, ast.Constant)
                                        and k.value == "strategy_type"
                                        and isinstance(v, ast.Constant)
                                    ):
                                        types.add(v.value)
                        return types
    raise AssertionError("Could not find winners list in main.py")


REGISTRY_DISABLED = {"quarter_theory", "elliott_wave", "elliott_pivot"}


def test_seed_excludes_registry_disabled_types():
    """_auto_deploy_winners must NOT seed strategy types that are commented
    out of strategies/registry.py.

    Regression for 2026-05-15: quarter-btc, ellpiv-btc, elliott-btc were
    being seeded by _auto_deploy_winners even though their underlying
    strategy types are deliberately disabled in the registry. This caused
    them to enter status='error' on every fresh DB and never poll.
    """
    seeded = _winners_list_strategy_types()
    overlap = seeded & REGISTRY_DISABLED
    assert not overlap, (
        f"Seed list includes registry-disabled types {overlap}. "
        f"These will fail to start with 'Unknown strategy type' on every "
        f"fresh DB. Remove them from _auto_deploy_winners or re-enable in "
        f"strategies/registry.py."
    )
