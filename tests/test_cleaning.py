import pandas as pd

from parse.cleaning import DataCleaner
from parse.core import SourceRef
from parse.eda import DataUnderstanding


def test_cleaner_proposes_changes_without_mutating_input():
    frame = pd.DataFrame({"id": [1, 2, 2], "value": [10.0, None, None], "kind": ["a", "b", "b"]})
    original = frame.copy(deep=True)
    source = SourceRef("dataset-a", "csv", "a.csv")

    result = DataCleaner(source).clean(frame)

    pd.testing.assert_frame_equal(frame, original)
    assert result.data_modified is False
    assert {proposal.proposal_id for proposal in result.proposals} == {"remove_duplicate_rows", "impute_value"}
    for proposal in result.proposals:
        assert proposal.confidence is not None
        assert 0.0 <= proposal.confidence <= 1.0
        assert "confidence_basis" in proposal.parameters
        assert isinstance(proposal.parameters["confidence_basis"], dict)
        assert len(proposal.parameters["confidence_basis"]) > 0
    assert result.before.to_dict() == result.after.to_dict()
    assert result.unresolved_issue_ids


def test_cleaner_applies_only_approved_proposals_and_records_lineage():
    frame = pd.DataFrame({"id": [1, 2, 2], "value": [10.0, None, None]})
    source = SourceRef("dataset-a", "csv", "a.csv")
    eda = DataUnderstanding(source).profile(frame)

    result = DataCleaner(source).clean(frame, approved=["impute_value"], eda_result=eda)

    assert result.cleaned["value"].isna().sum() == 0
    assert len(result.cleaned) == 3
    assert len(result.changes) == 2
    assert all(change.proposal_id == "impute_value" for change in result.changes)
    assert result.after.missing_by_column["value"] == 0
    assert "duplicate_rows" in result.unresolved_issue_ids
    assert result.provenance.operation == "data_quality_cleaning"


def test_cleaner_does_not_propose_mixed_values_or_outlier_deletion():
    frame = pd.DataFrame({"value": ["1", "unknown", "1000"]})
    issues, proposals = DataCleaner().detect(frame, DataUnderstanding().profile(frame))

    assert any(issue.kind == "mixed_values" for issue in issues)
    assert not any(proposal.action in {"convert_type", "remove_outliers"} for proposal in proposals)


def test_validation_detects_newly_introduced_outlier():
    # Construct a dataset where duplicate rows mask an outlier under IQR rule:
    # 20 duplicate rows with v=10, and single rows with v=11, 12, 13, and 100.
    # In original frame: 20 rows of 10 pull Q1 and Q3 to 10 (IQR=0), so 100 is not flagged as outlier.
    # When duplicate rows are removed, cleaned frame has [10, 11, 12, 13, 100].
    # Q1=11, Q3=13, IQR=2. Upper fence is 13 + 1.5*2 = 16. Value 100 is an outlier!
    rows = [{"v": 10.0, "label": "dup"}] * 20 + [
        {"v": 11.0, "label": "a"},
        {"v": 12.0, "label": "b"},
        {"v": 13.0, "label": "c"},
        {"v": 100.0, "label": "outlier"},
    ]
    frame = pd.DataFrame(rows)
    source = SourceRef("dataset-outlier", "csv", "outlier.csv")

    result = DataCleaner(source).clean(frame, approved=["remove_duplicate_rows"])

    assert result.validation is not None
    new_issues = result.validation.newly_introduced_issues
    assert len(new_issues) >= 1
    assert any("statistical_outliers in 'v'" in issue_msg for issue_msg in new_issues)
    assert any("Re-detection covers duplicates" in lim for lim in result.validation.limitations)


def test_validation_detects_newly_introduced_duplicate_row():
    # Construct a dataset where imputing a missing value makes two rows identical exact duplicates:
    # Row 0: {"x": 10.0, "y": "A"}
    # Row 1: {"x": None, "y": "A"}
    # Row 2: {"x": 20.0, "y": "B"}
    # In original frame, Row 0 and Row 1 are NOT duplicates.
    # If x is imputed with 10.0, Row 0 and Row 1 become exact duplicates!
    # To ensure median is 10.0, we can provide rows where median(non_null) == 10.0.
    frame = pd.DataFrame({
        "x": [10.0, None, 10.0],
        "y": ["A", "A", "B"],
    })
    source = SourceRef("dataset-dup", "csv", "dup.csv")

    result = DataCleaner(source).clean(frame, approved=["impute_x"])

    assert result.validation is not None
    new_issues = result.validation.newly_introduced_issues
    assert len(new_issues) >= 1
    assert any("duplicate_rows in dataset" in issue_msg for issue_msg in new_issues)


def test_outlier_proposal_contains_dual_method_basis():
    from parse.cleaning_api import create_cleaning_context, propose_cleaning
    from parse.analysis import AnalysisRequest
    from parse.semantic_analysis import AnalysisPipeline

    frame = pd.DataFrame({
        "num": [10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 1000.0]
    })
    source = SourceRef("dataset-dual", "csv")
    bundle = AnalysisPipeline().analyze(AnalysisRequest(frame, source))
    context = create_cleaning_context(bundle)
    issues, proposals = propose_cleaning(context, frame)

    outlier_props = [p for p in proposals if p.action == "flag_for_review"]
    assert len(outlier_props) == 1
    basis = outlier_props[0].parameters["confidence_basis"]
    assert basis["dual_method"] == "dual_IQR_MAD"
    assert basis["agreement_count"] == 1
    assert basis["agreement_rate"] == 1.0
    assert basis["raw_mad"] == 10.0
    assert "per_row_mad_distances" in basis
    assert "per_row_agreement" in basis


# --- Task 6: Unified Detection Regression Tests ---


def test_discrepancy_1_candidate_identifier_with_bare_eda():
    # Proves Discrepancy 1 resolution: candidate-identifier conflict detection fires
    # when only bare eda_result is passed (no CleaningContext).
    from parse.analysis import AnalysisOrchestrator, AnalysisRequest

    frame = pd.DataFrame({
        "id": [1, 2, 3, 1], # 'id' is a candidate identifier but has a conflict in non-exact duplicate rows
        "val": [10, 20, 30, 40], # row 0 and 3 differ in 'val', so not an exact duplicate row
    })
    source = SourceRef("bare_eda_test", "csv")
    eda = AnalysisOrchestrator().analyze(AnalysisRequest(frame, source))
    assert "id" in eda.structural_profile.candidate_index_columns

    cleaner = DataCleaner(source)
    issues, proposals = cleaner.detect(frame, eda_result=eda)

    candidate_issues = [i for i in issues if i.kind == "duplicate_identifier"]
    assert len(candidate_issues) == 1
    assert candidate_issues[0].field == "id"
    assert candidate_issues[0].duplicate_kind == "conflicting_identifier"


def test_discrepancy_2_purpose_gating_behavior():
    # Proves Discrepancy 2 resolution:
    # 1. Context-less callers (purpose=None) receive imputation proposals.
    # 2. Context callers with purpose="unknown" have imputation suppressed.
    # 3. Context callers with purpose="prediction" receive imputation proposals.
    from parse.cleaning_api import create_cleaning_context, propose_cleaning
    from parse.analysis import AnalysisRequest
    from parse.semantic_analysis import AnalysisPipeline

    frame = pd.DataFrame({
        "col": [1.0, 2.0, None, 4.0],
    })
    source = SourceRef("purpose_test", "csv")

    # Context-less caller:
    cleaner = DataCleaner(source)
    issues_bare, props_bare = cleaner.detect(frame)
    assert any(p.action == "impute_missing" for p in props_bare)

    # Context with purpose="unknown":
    bundle = AnalysisPipeline().analyze(AnalysisRequest(frame, source))
    ctx_unknown = create_cleaning_context(bundle, purpose="unknown")
    issues_unk, props_unk = propose_cleaning(ctx_unknown, frame)
    assert any(i.kind == "missing_values" for i in issues_unk)
    assert not any(p.action == "impute_missing" for p in props_unk)

    # Context with purpose="prediction":
    ctx_pred = create_cleaning_context(bundle, purpose="prediction")
    issues_pred, props_pred = propose_cleaning(ctx_pred, frame)
    assert any(p.action == "impute_missing" for p in props_pred)


def test_discrepancy_3_mixed_values_detected_via_context_path():
    # Proves Discrepancy 3 resolution:
    # Context-path callers (full CleaningContext) now correctly detect mixed-type values
    # via the ported numeric_fraction heuristic.
    from parse.cleaning_api import create_cleaning_context, propose_cleaning
    from parse.analysis import AnalysisRequest
    from parse.semantic_analysis import AnalysisPipeline

    frame = pd.DataFrame({
        "mixed_col": ["1", "unknown", "1000", "2"],
    })
    source = SourceRef("mixed_test", "csv")
    bundle = AnalysisPipeline().analyze(AnalysisRequest(frame, source))
    context = create_cleaning_context(bundle, purpose="descriptive_analysis")

    issues, proposals = propose_cleaning(context, frame)
    mixed_issues = [i for i in issues if i.kind == "mixed_values"]
    assert len(mixed_issues) == 1
    assert mixed_issues[0].field == "mixed_col"
    assert mixed_issues[0].method == "numeric_parse_fraction"


def test_discrepancy_4_multiple_outlier_findings_per_column():
    # Proves Discrepancy 4 resolution:
    # Multiple matching outlier findings for a single column produce multiple issues and proposals.
    from parse.analysis import Finding

    frame = pd.DataFrame({
        "val": [10.0, 11.0, 12.0, 10.0, 11.0, 100.0],
    })
    source = SourceRef("multi_outlier", "csv")

    class MockEDA:
        findings = [
            Finding(
                finding_id="outliers_val",
                category="distribution",
                subject="val",
                observation="Primary IQR outliers detected",
                method="IQR",
                evidence=(),
            ),
            Finding(
                finding_id="unusual:val",
                category="anomaly",
                subject="val",
                observation="Secondary extreme distribution anomaly detected",
                method="MAD",
                evidence=(),
            ),
        ]

    cleaner = DataCleaner(source)
    issues, proposals = cleaner.detect(frame, eda_result=MockEDA())

    outlier_issues = [i for i in issues if i.kind == "statistical_outliers"]
    outlier_proposals = [p for p in proposals if p.action == "flag_for_review"]

    assert len(outlier_issues) == 2
    assert len(outlier_proposals) == 2
    assert {i.issue_id for i in outlier_issues} == {"outliers_val", "unusual:val_1"}
    assert {p.proposal_id for p in outlier_proposals} == {"investigate_val_outliers", "investigate_val_outliers_1"}

