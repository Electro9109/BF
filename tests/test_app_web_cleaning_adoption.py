"""Cleaning remains a reviewed artifact until explicit source registration."""

from pathlib import Path
import pandas as pd

from parse.analysis import AnalysisRequest
from parse.semantic_analysis import AnalysisPipeline
from parse.cleaning import HumanDecision
from parse.cleaning_api import apply_cleaning, create_cleaning_context, propose_cleaning
from parse.core.contracts import SourceRef
from parse.eda_ui import analyze_loaded_dataset

APP_WEB_PATH = Path(__file__).resolve().parent.parent / "app_web.py"


def test_adopted_cleaned_frame_no_longer_reports_fixed_duplicate():
    frame = pd.DataFrame({"a": [1, 2, 1], "b": ["x", "y", "x"]})
    source = SourceRef("dup-test", "csv", label="dup.csv")
    bundle = AnalysisPipeline().analyze(AnalysisRequest(frame, source))
    context = create_cleaning_context(bundle, purpose="descriptive_analysis")
    issues_before, proposals = propose_cleaning(context, frame)

    assert any(issue.issue_id == "duplicate_rows" for issue in issues_before)
    dupe_prop = next(proposal for proposal in proposals if proposal.action == "remove_duplicates")

    cleaning_result = apply_cleaning(
        frame,
        [
            HumanDecision(
                decision_id="d1",
                proposal_id=dupe_prop.proposal_id,
                action="APPROVE",
                modifications={},
                rationale="Approved in test",
                reviewer="Test",
                created_at="2026-09-14",
            )
        ],
        context,
    )
    adopted_frame = cleaning_result.cleaned
    adopted_source = SourceRef(
        source.source_id,
        source.source_type,
        locator=source.locator,
        label=f"{source.label} (cleaned, adopted)",
    )
    adopted_bundle = analyze_loaded_dataset(adopted_frame, adopted_source, None)
    adopted_context = create_cleaning_context(adopted_bundle, purpose="descriptive_analysis")
    issues_after, _ = propose_cleaning(adopted_context, adopted_frame)

    assert not any(issue.issue_id == "duplicate_rows" for issue in issues_after)


def test_app_web_cleaning_is_reviewed_without_replacing_the_staged_original():
    source = APP_WEB_PATH.read_text(encoding="utf-8")
    assert "Apply review and keep original" in source
    assert "Use cleaned data for further analysis" not in source
    assert 'st.session_state.eda_working_dataset = "cleaned_adopted"' not in source
    assert "_reset_cleaning_result_on_purpose_change" in source
    assert "on_change=_reset_cleaning_result_on_purpose_change" in source
    assert "st.session_state.cleaning_result = None" in source
    apply_start = source.index("st.session_state.cleaning_result = apply_cleaning")
    apply_section = source[apply_start:apply_start + 700]
    assert "st.session_state.eda_frame =" not in apply_section


def test_registration_uses_selected_reviewed_frame_without_mutating_analysis_frame():
    source = APP_WEB_PATH.read_text(encoding="utf-8")
    assert 'options.append("Reviewed working copy")' in source
    assert 'review.cleaned' in source
    assert 'else st.session_state.eda_frame' in source
    assert "review.validation.to_dict()" in source
    assert "review.unresolved_issue_ids" in source
