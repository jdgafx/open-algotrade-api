"""Guards the vendored rigor package (backend/src/engine/rigor/), plan U2.

Two jobs:
  1. PARITY — the vendored validation.py/objective.py/_constants.py must not drift
     from the canonical meta-repo source (src/autotuner/backtest/ + config.py). The
     only allowed difference is the config import line. Skips when the meta-repo
     source isn't present (a standalone backend checkout / the Railway container).
  2. FUNCTIONAL — the vendored copy actually works inside the backend: DSR rejects
     overfit / accepts a genuine edge, CPCV summarizes, and backtest_fitness honors
     the pnl_net contract.
"""

import ast
import pathlib

import pytest

# backend/tests/test_rigor.py -> backend/ is parents[1], meta-repo root is parents[2]
_BACKEND = pathlib.Path(__file__).resolve().parents[1]
_META_ROOT = pathlib.Path(__file__).resolve().parents[2]

_MODULE_PAIRS = [
    ("src/engine/rigor/validation.py", "src/autotuner/backtest/validation.py"),
    ("src/engine/rigor/objective.py", "src/autotuner/backtest/objective.py"),
]

_CONST_NAMES = [
    "MIN_TRADES_FOR_PROMOTION", "CPCV_N_SPLITS", "CPCV_N_TEST_SPLITS",
    "CPCV_EMBARGO_FRAC", "CPCV_GAP_BARS", "DSR_MIN",
    "FITNESS_LAMBDA_DD", "FITNESS_LAMBDA_TO",
]


def _normalized_module_source(path: pathlib.Path) -> str:
    """Unparsed module source with import statements stripped — so the one
    legitimate config-import difference is tolerated, but EVERYTHING else
    (function bodies, module-level constants like _EULER_GAMMA, the docstring)
    must match byte-for-byte after normalization. Comments are absent from the
    AST, so the vendored-header comments don't matter."""
    tree = ast.parse(path.read_text())
    tree.body = [
        node for node in tree.body
        if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    return ast.unparse(tree)


def _named_constants(path: pathlib.Path) -> dict:
    tree = ast.parse(path.read_text())
    out = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in _CONST_NAMES):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                pass
    return out


@pytest.mark.parametrize("vendored_rel, source_rel", _MODULE_PAIRS)
def test_vendored_functions_match_meta_repo_source(vendored_rel, source_rel):
    source = _META_ROOT / source_rel
    if not source.exists():
        pytest.skip(f"meta-repo source {source} absent (standalone backend checkout)")
    assert _normalized_module_source(_BACKEND / vendored_rel) == _normalized_module_source(source), (
        f"VENDORED COPY DRIFTED from canonical source.\n"
        f"  vendored: {_BACKEND / vendored_rel}\n  source:   {source}\n"
        f"  Fix: edit the meta-repo source, then re-copy it here "
        f"(only the config import line may differ)."
    )


def test_vendored_constants_match_meta_repo_config():
    source = _META_ROOT / "src/autotuner/config.py"
    if not source.exists():
        pytest.skip(f"meta-repo config {source} absent (standalone backend checkout)")
    vendored = _named_constants(_BACKEND / "src/engine/rigor/_constants.py")
    canonical = _named_constants(source)
    for name in _CONST_NAMES:
        assert name in vendored, f"vendored _constants.py missing {name}"
        assert name in canonical, f"meta-repo config.py missing {name}"
        assert vendored[name] == canonical[name], (
            f"constant {name} drifted: vendored {vendored[name]} != source {canonical[name]}"
        )


# --- functional: the vendored copy is correct inside the backend ---

def test_dsr_rejects_overfit_accepts_genuine_edge():
    import numpy as np

    from src.engine.rigor import deflated_sharpe_ratio

    # Genuine edge among few trials -> DSR high (>= the 0.85 gate).
    rng = np.random.default_rng(0)
    good = rng.normal(0.02, 0.05, size=500)
    sr_good = float(good.mean() / good.std())
    assert deflated_sharpe_ratio(sr_good, [sr_good, 0.01, -0.02, 0.0], good)["DSR"] > 0.85

    # Luckiest of many noise trials -> DSR low.
    noise = rng.normal(0.001, 0.05, size=200)
    sr_noise = float(noise.mean() / noise.std())
    sr_trials = rng.normal(0.0, 0.1, size=200)
    assert deflated_sharpe_ratio(sr_noise, sr_trials, noise)["DSR"] < 0.85


def test_cpcv_stability_frac_positive():
    from src.engine.rigor import cpcv_stability

    assert cpcv_stability([1.0, 2.0, -1.0, 0.5])["frac_positive"] == 0.75
    assert cpcv_stability([-1.0, -2.0])["frac_positive"] == 0.0
    assert cpcv_stability([])["frac_positive"] == 0.0


def test_backtest_fitness_pnl_net_contract():
    from src.engine.rigor import backtest_fitness

    losing = [{"pnl_net": -1.0} for _ in range(40)]
    assert backtest_fitness(losing, span_days=40)["score"] == 0.0       # net-negative rejected
    winning = [{"pnl_net": 1.0} for _ in range(40)]
    assert backtest_fitness(winning, span_days=40)["score"] > 0


def test_purged_cpcv_yields_15_paths():
    from src.engine.rigor import purged_combinatorial_kfold

    paths = list(purged_combinatorial_kfold(600))   # C(6,2) = 15
    assert len(paths) == 15
    train_idx, test_idx = paths[0]
    assert len(test_idx) > 0 and len(train_idx) > 0
