"""Tests for the failure-safe Data Explorer -> Qwen Explainer bridge."""

from parse.explainer_llm import ExplainerResult
from parse.explainer_ui import explain_record


def test_explain_record_passes_recommendation_to_model(monkeypatch):
    import parse.explainer_llm

    seen = {}

    def fake_explain(record, recommendation=None):
        seen["record"] = record
        seen["recommendation"] = recommendation
        return ExplainerResult(
            text="The column contains three missing values.",
            fallback_used=False,
            finding_id=record["finding_id"],
        )

    monkeypatch.setattr(parse.explainer_llm, "explain", fake_explain)
    record = {
        "finding_id": "missing_temperature",
        "message": "Column 'temperature' has 3 missing value(s).",
    }

    result = explain_record(record, recommendation="Consider reviewing the missing values.")

    assert result.fallback_used is False
    assert seen["record"] is record
    assert seen["recommendation"] == "Consider reviewing the missing values."


def test_explain_record_returns_source_message_when_model_raises(monkeypatch):
    import parse.explainer_llm

    def broken_explain(record, recommendation=None):
        raise RuntimeError("weights unavailable")

    monkeypatch.setattr(parse.explainer_llm, "explain", broken_explain)
    record = {
        "finding_id": "missing_temperature",
        "message": "Column 'temperature' has 3 missing value(s).",
    }

    result = explain_record(record)

    assert result.text == record["message"]
    assert result.fallback_used is True
    assert "weights unavailable" in result.validation_detail
