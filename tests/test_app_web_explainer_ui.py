"""Regression tests for the primary plain-language explanation section."""

from pathlib import Path

from app_web import _explainer_display_content, _matching_explainer_result

APP_WEB_PATH = Path(__file__).resolve().parent.parent / "app_web.py"


def test_explainer_is_presented_before_supporting_analysis():
    source = APP_WEB_PATH.read_text(encoding="utf-8")
    explanation_position = source.index('st.markdown("### Plain-language explanation")')
    detailed_analysis_position = source.index('st.markdown("### Dataset Analysis")')
    column_profiles_position = source.index('with st.expander("Column profiles", expanded=False)')

    assert explanation_position < detailed_analysis_position
    assert explanation_position < column_profiles_position
    assert 'st.button("Generate explanation", key="generate_explainer", type="primary")' in source
    assert 'with st.expander("Findings", expanded=False)' in source
    assert 'with st.expander("View source finding and statistics", expanded=False)' in source
    assert '"Attributes", "Relationships", "Quality", "Semantic Candidates"' in source
    assert 'st.markdown("### Cleaner")' in source
    assert '"Download EDA result (JSON)"' in source
    assert '"Download cleaning result (JSON)"' in source


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
