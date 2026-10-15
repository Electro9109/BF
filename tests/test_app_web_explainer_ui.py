"""Regression tests for the primary plain-language explanation section."""

from pathlib import Path
from types import SimpleNamespace

from app_web import (
    _build_dataset_summary,
    _explainer_display_content,
    _matching_explainer_result,
)
import pandas as pd
from parse.analysis import generate_next_actions
from parse.core.contracts import SourceRef
from parse.eda import NextAction
from parse.eda_ui import analyze_loaded_dataset

APP_WEB_PATH = Path(__file__).resolve().parent.parent / "app_web.py"


def test_explainer_is_presented_before_supporting_analysis():
    source = APP_WEB_PATH.read_text(encoding="utf-8")
    summary_position = source.index('st.markdown("### What the analysis tells us")')
    individual_position = source.index('st.markdown("### Explain a finding")')
    translated_position = source.index("html.escape(display_content)")
    source_finding_position = source.index(
        'with st.expander("View source finding and statistics", expanded=False)'
    )
    detailed_analysis_position = source.index('st.markdown("### Dataset Analysis")')
    column_profiles_position = source.index('with st.expander("Column profiles", expanded=False)')
    analysis_bundle_state = source.index(
        'analysis_bundle = st.session_state.get("analysis_bundle")'
    )
    validation_use = source.index(
        "validation_analyses = [analysis_bundle.eda]"
    )

    assert summary_position < individual_position < detailed_analysis_position
    assert individual_position < translated_position < source_finding_position
    assert summary_position < column_profiles_position
    assert analysis_bundle_state < validation_use
    assert 'st.button("Generate explanation", key="generate_explainer", type="primary")' in source
    assert 'st.selectbox(\n                    "Finding to explain"' in source
    assert 'st.text_area(\n                    "Recommendation (optional)"' in source
    assert 'with st.expander("Findings", expanded=False)' in source
    assert "Supporting values:" in source
    assert "generate_next_actions(eda_result)" in source
    assert 'with st.expander("Other workflow checks", expanded=False)' in source
    assert '"Attributes", "Relationships", "Quality", "Semantic Candidates"' in source
    assert 'st.markdown("### Cleaner")' in source
    assert '"Download EDA result (JSON)"' in source
    assert '"Download cleaning result (JSON)"' in source


def test_dataset_summary_covers_multiple_source_findings_and_limitations():
    findings = [
        SimpleNamespace(
            finding_id="missing:age",
            category="quality",
            observation="Column age has 2 missing values.",
            result={"missing_count": 2, "missing_rate": 0.1},
            limitations=("The missingness mechanism is unknown.",),
        ),
        SimpleNamespace(
            finding_id="association:group:outcome",
            category="relationship",
            observation="Group and outcome show an association (Cramer's V 0.42).",
            result={"cramers_v": 0.42},
            limitations=("Association does not establish causation.",),
        ),
    ]
    result = SimpleNamespace(
        findings=findings,
        limitations=("Candidate meanings require contextual review.",),
        dataset_profile=SimpleNamespace(row_count=20, column_count=3),
    )

    actions = [
        NextAction("analyse", True, "At least one numeric variable is available."),
        NextAction(
            "build_prediction",
            False,
            "No sufficiently supported target/predictor set was detected.",
        ),
    ]
    summary = _build_dataset_summary(result, actions)

    assert summary["finding_count"] == 2
    assert [item["finding_id"] for item in summary["observations"]] == [
        "missing:age",
        "association:group:outcome",
    ]
    assert [item["observation"] for item in summary["observations"]] == [
        finding.observation for finding in findings
    ]
    assert summary["observations"][0]["supporting_values"] == {
        "missing_count": 2,
        "missing_rate": 0.1,
    }
    assert summary["observations"][1]["supporting_values"] == {"cramers_v": 0.42}
    assert summary["recommendations"] == [
        {
            "action": "analyse",
            "applicable": True,
            "reason": "At least one numeric variable is available.",
        }
    ]
    assert summary["other_actions"] == [
        {
            "action": "build_prediction",
            "applicable": False,
            "reason": "No sufficiently supported target/predictor set was detected.",
        }
    ]
    assert "20 rows and 3 columns" in summary["scope"]
    assert "not every possible property" in summary["scope"]
    assert summary["limitations"] == [
        "The missingness mechanism is unknown.",
        "Association does not establish causation.",
        "Candidate meanings require contextual review.",
    ]

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
            category="quality",
            observation=f"Source observation {index} with value {index}.",
            result={},
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


def test_dataset_summary_uses_existing_numeric_evidence_and_workflow_actions():
    frame = pd.DataFrame(
        {"group": ["A", "A", "B", "B", "B"], "value": [1.0, 2.0, 8.0, None, 12.0]}
    )
    bundle = analyze_loaded_dataset(
        frame,
        SourceRef("summary-test", "csv", "summary.csv"),
        None,
    )

    summary = _build_dataset_summary(
        bundle.eda,
        generate_next_actions(bundle.eda),
    )

    missingness = next(
        item for item in summary["observations"] if item["finding_id"] == "missing:value"
    )
    difference = next(
        item
        for item in summary["observations"]
        if item["finding_id"] == "group_difference:group:value"
    )
    assert missingness["supporting_values"] == {"missing_count": 1, "missing_rate": 0.2}
    assert difference["supporting_values"]["mean_spread"] == 8.5
    assert [action["action"] for action in summary["recommendations"]] == [
        "clean_prepare",
        "analyse",
        "explore_retrieve",
        "generate_report",
    ]


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


def test_new_ingestion_and_reanalysis_clear_explanation_state():
    source = APP_WEB_PATH.read_text(encoding="utf-8")
    assert "st.session_state.explainer_result = None" in source
    assert 'st.session_state.explainer_finding_key = ""' in source
    assert 'st.session_state.explainer_recommendation = ""' in source
    assert '"explainer_result"' in source
    assert "ingestion_analysis_hash = None" in source
