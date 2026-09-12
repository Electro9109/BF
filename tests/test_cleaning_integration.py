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
    assert missing_proposals[0].confidence is not None
    assert "confidence_basis" in missing_proposals[0].parameters
    assert missing_proposals[0].parameters["confidence_basis"]["missingness_pattern_checked"] is False


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
    assert result.validation.newly_introduced_issues == []
    assert any("Re-detection covers duplicates" in lim for lim in result.validation.limitations)
    
    assert result.downstream_impact is not None
    assert result.downstream_impact.sample_size_before == 5
    assert result.downstream_impact.sample_size_after == 4
    assert "value" in result.downstream_impact.distribution_shifts
    assert "category" in result.downstream_impact.distribution_shifts
    assert "id" in result.downstream_impact.distribution_shifts


def test_imputation_downstream_impact_shift(analysis_bundle, sample_frame):
    context = create_cleaning_context(analysis_bundle, purpose="prediction")
    issues, proposals = propose_cleaning(context, sample_frame)

    impute_prop = next(p for p in proposals if p.action == "impute_missing")

    decisions = [
        HumanDecision("d1", impute_prop.proposal_id, "APPROVE", {}, "Impute missing", "Test", "2026-09-11")
    ]

    result = apply_cleaning(sample_frame, decisions, context)

    assert result.downstream_impact is not None
    shifts = result.downstream_impact.distribution_shifts
    assert "value" in shifts
    val_shift = shifts["value"]
    assert val_shift["kind"] == "numeric"
    assert val_shift["sample_size_before"] == 4
    assert val_shift["sample_size_after"] == 5
    assert not val_shift["unchanged"]
    # All columns present in both frames must be present in shifts
    assert "id" in shifts
    assert "category" in shifts
    assert shifts["id"]["unchanged"] is True


def test_purpose_unknown_stays_conservative(analysis_bundle, sample_frame):
    context = create_cleaning_context(analysis_bundle, purpose="unknown")
    issues, proposals = propose_cleaning(context, sample_frame)
    
    # "unknown" purpose shouldn't propose imputation
    missing_proposals = [p for p in proposals if p.action == "impute_missing"]
    assert len(missing_proposals) == 0


def test_validation_empty_cleaned_frame_degrades_gracefully():
    from parse.cleaning import DataCleaner
    source = SourceRef("empty_ds", "csv")
    empty_df = pd.DataFrame(columns=["a", "b"])
    cleaner = DataCleaner(source)
    result = cleaner.clean(empty_df)

    assert result.validation is not None
    assert result.validation.newly_introduced_issues == []
    assert any("cleaned dataset has 0 rows" in note for note in result.validation.limitations)

