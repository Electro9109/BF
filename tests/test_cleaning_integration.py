import pytest
import pandas as pd
from parse.core.contracts import SourceRef
from parse.analysis import AnalysisRequest, AnalysisOrchestrator
from parse.semantic_analysis import AnalysisPipeline
from parse.cleaning_api import create_cleaning_context, propose_cleaning, apply_cleaning
from parse.cleaning import HumanDecision


@pytest.fixture
def sample_frame():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 1], # duplicate id 1
        "value": [10.5, None, 15.2, 100.0, 10.5], # missing value, duplicate row, outlier 100.0
        "category": ["A", "B", "A", "C", "A"], # exactly duplicate row 0 and 4
    })


@pytest.fixture
def analysis_bundle(sample_frame):
    source = SourceRef("test_dataset", "csv")
    request = AnalysisRequest(sample_frame, source)
    return AnalysisPipeline().analyze(request)


def test_cleaning_context_creation_from_bundle(analysis_bundle):
    context = create_cleaning_context(analysis_bundle, purpose="prediction")
    assert context.purpose == "prediction"
    assert "value" in context.attribute_profiles
    assert context.attribute_profiles["value"].missing_count == 1
    
    # Check that outliers are ported over
    has_outlier = False
    for anomaly in context.attribute_profiles["value"].anomalies:
        fid = anomaly.get("finding_id", "")
        if "outlier" in fid or "unusual" in fid:
            has_outlier = True
    assert has_outlier


def test_cleaning_consumes_analysis_missingness_not_re_run(analysis_bundle, sample_frame):
    context = create_cleaning_context(analysis_bundle, purpose="descriptive_analysis")
    issues, proposals = propose_cleaning(context, sample_frame)
    
    missing_issues = [i for i in issues if i.kind == "missing_values"]
    assert len(missing_issues) == 1
    assert missing_issues[0].field == "value"
    assert missing_issues[0].method == "analysis_profile" # specific to context-aware detection
    
    missing_proposals = [p for p in proposals if p.action == "impute_missing"]
    assert len(missing_proposals) == 1
    assert missing_proposals[0].field == "value"


def test_cleaning_duplicate_kinds_distinguished(analysis_bundle, sample_frame):
    context = create_cleaning_context(analysis_bundle)
    issues, proposals = propose_cleaning(context, sample_frame)
    
    # Exact duplicate row
    exact_dupes = [i for i in issues if i.duplicate_kind == "exact"]
    assert len(exact_dupes) == 1
    assert exact_dupes[0].issue_id == "duplicate_rows"
    
    # We may not have structural analysis set up to find 'id' as a candidate key 
    # in this simple dataset without more config, but we can verify exact dupes work.


def test_human_decision_REJECT_prevents_transform(analysis_bundle, sample_frame):
    context = create_cleaning_context(analysis_bundle, purpose="prediction")
    issues, proposals = propose_cleaning(context, sample_frame)
    
    impute_prop = next(p for p in proposals if p.action == "impute_missing")
    
    decisions = [
        HumanDecision(
            decision_id="d1",
            proposal_id=impute_prop.proposal_id,
            action="REJECT",
            modifications={},
            rationale="Do not impute",
            reviewer="Test",
            created_at="2026-09-11"
        )
    ]
    
    result = apply_cleaning(sample_frame, decisions, context)
    assert not result.cleaned["value"].notna().all() # should still have a null
    assert len(result.changes) == 0


def test_before_after_validation_produced(analysis_bundle, sample_frame):
    context = create_cleaning_context(analysis_bundle, purpose="prediction")
    issues, proposals = propose_cleaning(context, sample_frame)
    
    dupe_prop = next(p for p in proposals if p.action == "remove_duplicates")
    
    decisions = [
        HumanDecision("d1", dupe_prop.proposal_id, "APPROVE", {}, "Ok", "Test", "2026-09-11")
    ]
    
    result = apply_cleaning(sample_frame, decisions, context)
    
    assert result.validation is not None
    assert result.validation.before_snapshot.row_count == 5
    assert result.validation.after_snapshot.row_count == 4
    
    assert result.downstream_impact is not None
    assert result.downstream_impact.sample_size_before == 5
    assert result.downstream_impact.sample_size_after == 4


def test_purpose_unknown_stays_conservative(analysis_bundle, sample_frame):
    context = create_cleaning_context(analysis_bundle, purpose="unknown")
    issues, proposals = propose_cleaning(context, sample_frame)
    
    # "unknown" purpose shouldn't propose imputation
    missing_proposals = [p for p in proposals if p.action == "impute_missing"]
    assert len(missing_proposals) == 0
