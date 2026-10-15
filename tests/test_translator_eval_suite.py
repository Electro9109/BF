import pytest

from parse.core.translator_contract import StructuredModuleResult
from parse.translator_pipeline import ResultsTranslator, deterministic_fallback_translation
from parse.translator_prompt import format_user_prompt


def test_eval_case_1_descriptive_statistics():
    """Case 1: Descriptive statistics with meaningful numerical values."""
    res = StructuredModuleResult(
        department="sintering",
        module_type="descriptive_analysis",
        operation="granulometry_summary",
        dataset_id="sinter_granulometry.csv",
        summary="Analysis of coke and flux sizing distribution.",
        statistics={
            "sample_size": 449,
            "flux_crushing_mean": 82.52,
            "coke_crushing_mean": 77.06,
        },
    )
    prompt = format_user_prompt(res)
    assert "82.52" in prompt
    assert "77.06" in prompt
    assert "449" in prompt

    text = deterministic_fallback_translation(res)
    assert "82.52" in text
    assert "449" in text


def test_eval_case_2_missing_value_analysis_and_recommendation():
    """Case 2: Missing-value analysis and a recommendation for handling missing data."""
    res = StructuredModuleResult(
        department="general_plant",
        module_type="quality_check",
        operation="missing_value_audit",
        dataset_id="plant_sensors.csv",
        summary="Detected missing sensor records in moisture monitoring.",
        findings=[{
            "finding_id": "missing_moisture",
            "observation": "Moisture sensor has 18 missing values (4.0% of batch).",
        }],
        recommendations=["Impute missing values using forward-fill from pre-ignition analyzer."],
    )
    text = deterministic_fallback_translation(res)
    assert "18 missing values" in text
    assert "Impute missing values" in text


def test_eval_case_3_correlation_non_causal():
    """Case 3: Correlation results that must not be interpreted causally."""
    res = StructuredModuleResult(
        department="sintering",
        module_type="descriptive_analysis",
        operation="correlation_screening",
        dataset_id="sinter_chem.csv",
        summary="Screened correlations between basicity and tumbler strength.",
        findings=[{
            "finding_id": "corr_basicity_dti",
            "observation": "Observed Pearson correlation r=0.48 between Basicity and DTI.",
        }],
        limitations=["Correlation does not establish causation or physical necessity."],
    )
    text = deterministic_fallback_translation(res)
    assert "r=0.48" in text
    assert "Correlation does not establish causation" in text


def test_eval_case_4_anomaly_findings():
    """Case 4: Outlier or anomaly findings."""
    res = StructuredModuleResult(
        department="energy",
        module_type="anomaly_detection",
        operation="power_factor_audit",
        dataset_id="substation_log.csv",
        summary="Detected transient power factor dips.",
        anomalies=[{
            "anomaly_id": "pf_dip",
            "description": "Power factor dipped below 0.60 for 14 intervals.",
            "count": 14,
        }],
    )
    text = deterministic_fallback_translation(res)
    assert "Power factor dipped below 0.60" in text
    assert "14" in text


def test_eval_case_5_prediction_with_metrics():
    """Case 5: A prediction accompanied by evaluation metrics."""
    res = StructuredModuleResult(
        department="sintering",
        module_type="regression_prediction",
        operation="productivity_prediction",
        dataset_id="sinter_prod.csv",
        summary="Trained Random Forest regressor on 449 samples.",
        target="Sinter Productivity",
        units="T/m2-hr",
        evaluation_metrics={"mae": 0.0502, "rmse": 0.0645, "r2": 0.293},
    )
    text = deterministic_fallback_translation(res)
    assert "0.0502" in text
    assert "0.293" in text
    assert "T/m2-hr" in text


def test_eval_case_6_analyser_recommendation_preserved():
    """Case 6: An analyser recommendation that must be explained without being altered."""
    res = StructuredModuleResult(
        department="energy",
        module_type="recommendation_engine",
        operation="peak_shaving_recommendation",
        dataset_id="load_curve.csv",
        summary="Identified opportunities for load shifting.",
        recommendations=["Shift heavy batch milling from Maximum_Load hours to Light_Load hours."],
    )
    text = deterministic_fallback_translation(res)
    assert "Shift heavy batch milling from Maximum_Load hours to Light_Load hours." in text


def test_eval_case_7_insufficient_or_malformed_input():
    """Case 7: Insufficient or malformed input."""
    translator = ResultsTranslator(enabled=False)
    empty_res = StructuredModuleResult(
        department="general_plant",
        module_type="descriptive_analysis",
        dataset_id="bad_data.csv",
    )
    out = translator.translate(empty_res)
    assert out.status == "malformed_input"
    assert "empty and cannot be translated" in out.text


def test_eval_case_8_contradictory_or_incomplete_evidence():
    """Case 8: Contradictory or incomplete evidence."""
    res = StructuredModuleResult(
        department="sintering",
        module_type="descriptive_analysis",
        operation="quality_discrepancy",
        dataset_id="sinter_discrepancy.csv",
        summary="Comparison between automated and manual FeO measurements.",
        findings=[{
            "finding_id": "discrepancy_feo",
            "observation": "Manual titrations report FeO=8.4% while XRF reports FeO=9.1% (gap = 0.7 percentage points).",
        }],
        validation_warnings=["Instrument calibration drift detected; measurements conflict."],
        limitations=["Reconcile lab titration before adjusting fuel rate."],
    )
    text = deterministic_fallback_translation(res)
    assert "gap = 0.7 percentage points" in text
    assert "Reconcile lab titration" in text


def test_eval_case_9_missing_confidence_information():
    """Case 9: An example with missing confidence or uncertainty information."""
    res = StructuredModuleResult(
        department="blast_furnace",
        module_type="regression_prediction",
        operation="uncalibrated_prediction",
        dataset_id="custom_burden.csv",
        summary="Predicted Ts for experimental burden mix.",
        target="Ts",
        units="°C",
        predictions={"Ts": 1280.0},
        confidence=None,  # missing confidence
        limitations=["No confidence estimate available; distance from training distribution is unknown."],
    )
    text = deterministic_fallback_translation(res)
    assert "1280" in text
    assert "No confidence estimate available" in text


def test_eval_case_10_department_specific_context_preserved():
    """Case 10: Multi-department context preservation across blast furnace, sintering, and energy."""
    departments = ["blast_furnace", "sintering", "energy"]
    for dept in departments:
        res = StructuredModuleResult(
            department=dept,
            module_type="descriptive_analysis",
            operation=f"{dept}_audit",
            dataset_id=f"{dept}_records.csv",
            summary=f"Operational audit for {dept}.",
        )
        text = deterministic_fallback_translation(res)
        assert dept.replace("_", " ").title() in text
