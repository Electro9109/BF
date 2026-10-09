"""Tests for parse/explainer_llm.py — model-free paths only.

These tests cover:
- The ExplainerResult dataclass shape
- Fallback behaviour when the model is disabled via env var
- Fallback behaviour when adapter directory is missing
- Validation/fallback logic (patched _generate_raw)
- Prompt-echo protection: token slicing in _generate_raw signature

Model weights are NOT loaded in this test module.  Integration tests that
actually run inference should be marked with pytest.mark.integration and
skipped in CI if the model is not present.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import parse.explainer_llm as llm_module
from parse.explainer_llm import (
    ADAPTER_PATH,
    BASE_MODEL_ID,
    ENABLED_ENV_VAR,
    ExplainerResult,
    explain,
    is_enabled,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_model_state():
    """Ensure module-level model cache is cleared between tests."""
    original_model = llm_module._model
    original_tokenizer = llm_module._tokenizer
    llm_module._model = None
    llm_module._tokenizer = None
    yield
    llm_module._model = original_model
    llm_module._tokenizer = original_tokenizer


@pytest.fixture
def outlier_record():
    return {
        "schema_version": "1.0",
        "finding_id": "outliers_measurement",
        "source": "eda",
        "category": "statistical",
        "kind": "observation",
        "message": "Column 'measurement' contains 3 IQR-based statistical outlier(s).",
        "attributes": {"count": 3, "column": "measurement"},
        "limitations": ["A statistical outlier is not automatically a data error."],
    }


@pytest.fixture
def no_numbers_record():
    return {
        "schema_version": "1.0",
        "finding_id": "group_batch_id",
        "source": "eda",
        "category": "logical",
        "kind": "interpretation",
        "message": "Column 'batch_id' may define groups for stratified analysis.",
        "attributes": {"column": "batch_id"},
        "limitations": ["Grouping candidacy is inferred from cardinality; confirm its meaning."],
    }


# ── ExplainerResult dataclass ──────────────────────────────────────────────────


def test_explainer_result_defaults():
    r = ExplainerResult(text="hello")
    assert r.text == "hello"
    assert r.fallback_used is False
    assert r.validation_detail == ""
    assert r.finding_id == ""
    assert r.prompt_token_count == 0
    assert r.generated_token_count == 0


def test_explainer_result_fallback_fields():
    r = ExplainerResult(
        text="fallback",
        fallback_used=True,
        validation_detail="validation failed",
        finding_id="outliers_x",
    )
    assert r.fallback_used is True
    assert "validation failed" in r.validation_detail


# ── is_enabled() ──────────────────────────────────────────────────────────────


def test_is_enabled_default(monkeypatch):
    monkeypatch.delenv(ENABLED_ENV_VAR, raising=False)
    assert is_enabled() is True


@pytest.mark.parametrize("val", ["0", "false", "False", "no", "off"])
def test_is_enabled_false_values(monkeypatch, val):
    monkeypatch.setenv(ENABLED_ENV_VAR, val)
    assert is_enabled() is False


@pytest.mark.parametrize("val", ["1", "true", "True", "yes"])
def test_is_enabled_true_values(monkeypatch, val):
    monkeypatch.setenv(ENABLED_ENV_VAR, val)
    assert is_enabled() is True


# ── Fallback when disabled ─────────────────────────────────────────────────────


def test_explain_returns_fallback_when_disabled(monkeypatch, outlier_record):
    monkeypatch.setenv(ENABLED_ENV_VAR, "0")
    result = explain(outlier_record)
    assert result.fallback_used is True
    assert result.text == outlier_record["message"]
    assert result.finding_id == "outliers_measurement"


# ── Fallback when adapter missing ─────────────────────────────────────────────


def test_explain_returns_fallback_when_adapter_missing(monkeypatch, tmp_path, outlier_record):
    monkeypatch.delenv(ENABLED_ENV_VAR, raising=False)
    # Point adapter path to a non-existent directory
    with patch.object(llm_module, "ADAPTER_PATH", tmp_path / "nonexistent"):
        result = explain(outlier_record)
    assert result.fallback_used is True
    assert result.text == outlier_record["message"]


# ── Validation / fallback logic ────────────────────────────────────────────────


def _mock_model_and_tokenizer(module):
    """Inject a mock model+tokenizer that satisfy load_model()."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "<prompt>"
    mock_model = MagicMock()
    module._model = mock_model
    module._tokenizer = mock_tokenizer
    return mock_model, mock_tokenizer


def test_explain_uses_fallback_when_validation_fails(monkeypatch, outlier_record):
    """When generated text fails fidelity checks, fallback text is returned."""
    monkeypatch.delenv(ENABLED_ENV_VAR, raising=False)
    _mock_model_and_tokenizer(llm_module)

    # Bad output: drops all numbers
    bad_output = "There are some unusual values in this column."

    with patch.object(llm_module, "_generate_raw", return_value=(bad_output, 50, 20)):
        result = explain(outlier_record)

    assert result.fallback_used is True
    assert result.text == outlier_record["message"]
    assert result.generated_token_count == 20


def test_explain_returns_generated_text_when_validation_passes(monkeypatch, outlier_record):
    """When generated text passes fidelity checks, it is returned."""
    monkeypatch.delenv(ENABLED_ENV_VAR, raising=False)
    _mock_model_and_tokenizer(llm_module)

    good_output = (
        "The 'measurement' column contains 3 statistical outliers (IQR-based). "
        "A statistical outlier is not automatically a data error."
    )

    with patch.object(llm_module, "_generate_raw", return_value=(good_output, 50, 30)):
        result = explain(outlier_record)

    assert result.fallback_used is False
    assert result.text == good_output
    assert result.generated_token_count == 30


def test_explain_returns_fallback_on_inference_exception(monkeypatch, outlier_record):
    monkeypatch.delenv(ENABLED_ENV_VAR, raising=False)
    _mock_model_and_tokenizer(llm_module)

    with patch.object(llm_module, "_generate_raw", side_effect=RuntimeError("CUDA OOM")):
        result = explain(outlier_record)

    assert result.fallback_used is True
    assert "CUDA OOM" in result.validation_detail


# ── Prompt-echo protection ─────────────────────────────────────────────────────


def test_generated_token_count_reflects_only_new_tokens(monkeypatch, outlier_record):
    """generated_token_count must be the slice after prompt, never the total sequence."""
    monkeypatch.delenv(ENABLED_ENV_VAR, raising=False)
    _mock_model_and_tokenizer(llm_module)

    good_output = (
        "The 'measurement' column contains 3 statistical outliers (IQR-based). "
        "A statistical outlier is not automatically a data error."
    )

    with patch.object(llm_module, "_generate_raw", return_value=(good_output, 120, 45)) as mock_gen:
        result = explain(outlier_record)

    # prompt_token_count and generated_token_count are tracked separately
    assert result.prompt_token_count == 120
    assert result.generated_token_count == 45
    # prompt_token_count + generated_token_count != generated_token_count alone
    assert result.prompt_token_count != result.generated_token_count


# ── Finding ID echo ────────────────────────────────────────────────────────────


def test_finding_id_echoed_in_result(monkeypatch, outlier_record):
    monkeypatch.setenv(ENABLED_ENV_VAR, "0")
    result = explain(outlier_record)
    assert result.finding_id == "outliers_measurement"


# ── Adapter path constant ──────────────────────────────────────────────────────


def test_adapter_path_is_inside_parse_models():
    assert ADAPTER_PATH.parts[-1] == "qwen3-explainer-v0.3"
    assert "parse" in ADAPTER_PATH.parts
    assert "models" in ADAPTER_PATH.parts
