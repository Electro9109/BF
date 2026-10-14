"""Regression tests for the primary plain-language explanation section."""

from pathlib import Path
from types import SimpleNamespace

from app_web import (
    _build_dataset_summary,
    _explainer_display_content,
    _matching_explainer_result,
)

APP_WEB_PATH = Path(__file__).resolve().parent.parent / "app_web.py"


def test_explainer_is_presented_before_supporting_analysis():
    source = APP_WEB_PATH.read_text(encoding="utf-8")
    summary_position = source.index('st.markdown("### What the analysis tells us")')
    individual_position = source.index('with st.expander("Explain an individual finding", expanded=False)')
    detailed_analysis_position = source.index('st.markdown("### Dataset Analysis")')
    column_profiles_position = source.index('with st.expander("Column profiles", expanded=False)')

    assert summary_position < individual_position < detailed_analysis_position
    assert summary_position < column_profiles_position
    assert 'st.button("Generate explanation", key="generate_explainer", type="primary")' in source
    assert 'st.selectbox(\n                    "Finding to explain"' in source
    assert 'st.text_area(\n                    "Recommendation (optional)"' in source
    assert 'with st.expander("Findings", expanded=False)' in source
    assert 'with st.expander("View source finding and statistics", expanded=False)' in source
    assert '"Attributes", "Relationships", "Quality", "Semantic Candidates"' in source
    assert 'st.markdown("### Cleaner")' in source
    assert '"Download EDA result (JSON)"' in source
    assert '"Download cleaning result (JSON)"' in source


def test_dataset_summary_covers_multiple_source_findings_and_limitations():
    findings = [
        SimpleNamespace(
            finding_id="missing:age",
            observation="Column age has 2 missing values.",
            limitations=("The missingness mechanism is unknown.",),
        ),
        SimpleNamespace(
            finding_id="association:group:outcome",
            observation="Group and outcome show an association (Cramer's V 0.42).",
            limitations=("Association does not establish causation.",),
        ),
    ]
    result = SimpleNamespace(
        findings=findings,
        limitations=("Candidate meanings require contextual review.",),
        dataset_profile=SimpleNamespace(row_count=20, column_count=3),
    )

    summary = _build_dataset_summary(result)

    assert summary["finding_count"] == 2
    assert [item["finding_id"] for item in summary["observations"]] == [
        "missing:age",
        "association:group:outcome",
    ]
    assert [item["observation"] for item in summary["observations"]] == [
        finding.observation for finding in findings
    ]
    assert "20 rows and 3 columns" in summary["scope"]
    assert "not every possible property" in summary["scope"]
    assert summary["limitations"] == [
        "The missingness mechanism is unknown.",
        "Association does not establish causation.",
        "Candidate meanings require contextual review.",
    ]
    assert "recommendation" not in summary


def test_dataset_summary_keeps_empty_findings_empty():
    result = SimpleNamespace(
        findings=[],
        limitations=["No limitations were reported."],
        dataset_profile=SimpleNamespace(row_count=0, column_count=2),
    )

    summary = _build_dataset_summary(result)

    assert summary["finding_count"] == 0
    assert summary["observations"] == []


def test_dataset_summary_limits_primary_observations_and_reports_omitted_count():
    findings = [
        SimpleNamespace(
            finding_id=f"finding:{index}",
            observation=f"Source observation {index} with value {index}.",
            limitations=(),
        )
        for index in range(7)
    ]
    result = SimpleNamespace(
        findings=findings,
        limitations=[],
        dataset_profile=SimpleNamespace(row_count=10, column_count=2),
    )

    summary = _build_dataset_summary(result)

    assert [item["finding_id"] for item in summary["observations"]] == [
        "finding:0",
        "finding:1",
        "finding:2",
        "finding:3",
        "finding:4",
    ]
    assert summary["omitted_count"] == 2


def test_previous_explanation_is_not_reused_for_different_inputs():
    result = {
        "finding_id": "missing:age",
        "recommendation": "Review the missing values.",
        "content": "Validated explanation",
    }

    assert _matching_explainer_result(
        result, "missing:age", "Review the missing values."
    ) is result
    assert _matching_explainer_result(
        result, "skew:age", "Review the missing values."
    ) is None
    assert _matching_explainer_result(
        result, "missing:age", None
    ) is None


def test_unvalidated_result_displays_original_finding():
    rejected = {
        "content": "Unvalidated model output",
        "fallback": False,
        "validation_passed": False,
    }
    validated = {
        "content": "Validated explanation",
        "fallback": False,
        "validation_passed": True,
    }

    assert _explainer_display_content(rejected, "Original finding") == "Original finding"
    assert _explainer_display_content(validated, "Original finding") == "Validated explanation"


def test_reprofiling_and_dataset_adoption_clear_explanation_state():
    source = APP_WEB_PATH.read_text(encoding="utf-8")
    assert source.count("st.session_state.explainer_result = None") >= 3
    assert source.count('st.session_state.explainer_finding_key = ""') >= 3
    assert source.count('st.session_state.explainer_recommendation = ""') >= 3
