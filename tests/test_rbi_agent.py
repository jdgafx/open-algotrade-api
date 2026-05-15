"""RBI agent regression tests.

Tests that don't require running the actual Anthropic API — just verify
config and code structure.
"""
import re
from pathlib import Path


def test_anthropic_model_id_is_current():
    """The RBI agent must use a current (non-deprecated) Anthropic model.

    Regression for 2026-05-15: agent was calling claude-sonnet-4-20250514
    which Anthropic deprecated; every POST /rbi/jobs returned a 400 Bad
    Request from api.anthropic.com. The user requested Opus inference
    explicitly, so the live model is claude-opus-4-7.

    This test prevents the regression of pinning a dated-suffix model
    name that subsequently gets deprecated.
    """
    src = (
        Path(__file__).resolve().parent.parent
        / "src" / "services" / "rbi_agent.py"
    )
    text = src.read_text()

    deprecated_pattern = re.compile(r'claude-(?:sonnet|opus|haiku)-\d+-\d{8}')
    deprecated_hits = deprecated_pattern.findall(text)
    assert not deprecated_hits, (
        f"rbi_agent.py contains deprecated dated-suffix model name(s): "
        f"{deprecated_hits}. Use the rolling alias (claude-opus-4-7, "
        f"claude-sonnet-4-6, claude-haiku-4-5) instead."
    )

    # Must reference a current model
    current_models = re.compile(r'claude-(?:opus-4-7|sonnet-4-6|haiku-4-5)')
    assert current_models.search(text), (
        f"rbi_agent.py must reference one of the current Claude 4.X models. "
        f"Latest: claude-opus-4-7 (best), claude-sonnet-4-6 (standard), "
        f"claude-haiku-4-5 (cheap)."
    )
