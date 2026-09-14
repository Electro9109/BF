"""Task 10: adopted cleaned frame must drive re-detection (no Streamlit)."""

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


def test_app_web_cleaning_state_flow_guards():
    source = APP_WEB_PATH.read_text(encoding="utf-8")
    assert "Use cleaned data for further analysis" in source
    assert "eda_working_dataset" in source
    assert "cleaned_adopted" in source
    assert "_reset_cleaning_result_on_purpose_change" in source
    assert "on_change=_reset_cleaning_result_on_purpose_change" in source
    assert "st.session_state.cleaning_result = None" in source
    # Adoption is explicit: apply path stores result but must not swap the working frame.
    apply_block_start = source.index("st.session_state.cleaning_result = apply_cleaning")
    adopt_block_start = source.index("Use cleaned data for further analysis")
    apply_section = source[apply_block_start:adopt_block_start]
    assert "st.session_state.eda_frame =" not in apply_section
