import pytest
from pathlib import Path

from config.paths import SINTER_PRODUCTIVITY_CSV, STEEL_ENERGY_CSV
from parse.core.translator_contract import StructuredModuleResult
from parse.domain_workflows import (
    analyze_sintering_productivity,
    analyze_steel_energy_consumption,
)
from parse.translator_pipeline import ResultsTranslator, deterministic_fallback_translation
from parse.translator_prompt import build_translator_messages, format_user_prompt


def test_sintering_workflow_produces_valid_contract():
    res = analyze_sintering_productivity()
    assert isinstance(res, StructuredModuleResult)
    assert res.department == "sintering"
    assert res.target == "Sinter Productivity, T/m2-hr"
    assert res.statistics["sample_count"] == 449
    assert "r2" in res.evaluation_metrics
    assert len(res.findings) >= 2
    assert len(res.recommendations) >= 1
    assert len(res.limitations) >= 1


def test_energy_workflow_produces_valid_contract():
    res = analyze_steel_energy_consumption()
    assert isinstance(res, StructuredModuleResult)
    assert res.department == "energy"
    assert res.target == "Usage_kWh"
    assert res.statistics["sample_count"] == 35040
    assert "r2" in res.evaluation_metrics
    assert len(res.anomalies) >= 1
    assert len(res.recommendations) >= 1


def test_prompt_formatter_preserves_numerical_tokens():
    res = StructuredModuleResult(
        department="sintering",
        module_type="regression_prediction",
        operation="productivity_prediction",
        dataset_id="sinter_test.csv",
        summary="Test summary",
        target="Sinter Productivity, T/m2-hr",
        units="T/m2-hr",
        evaluation_metrics={"r2": 0.293, "rmse": 0.0645},
        recommendations=["Maintain crushing index above 75%."],
    )
    user_prompt = format_user_prompt(res)
    assert "0.293" in user_prompt
    assert "0.0645" in user_prompt
    assert "Sinter Productivity, T/m2-hr" in user_prompt
    assert "Maintain crushing index above 75%." in user_prompt

    messages = build_translator_messages(res)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "Role: PARSE Results Translator" in messages[0]["content"]


def test_translator_pipeline_deterministic_fallback():
    translator = ResultsTranslator(enabled=False)
    res = analyze_sintering_productivity()
    out = translator.translate(res)

    assert out.is_fallback is True
    assert out.status == "fallback_disabled"
    assert "449" in out.text
    assert "T/m2-hr" in out.text
    assert "RandomForestRegressor" in out.text


def test_translator_malformed_input_handling():
    translator = ResultsTranslator(enabled=False)
    empty_res = StructuredModuleResult(
        department="general_plant",
        module_type="descriptive_analysis",
        dataset_id="empty.csv",
    )
    out = translator.translate(empty_res)
    assert out.status == "malformed_input"
    assert "empty and cannot be translated" in out.text
