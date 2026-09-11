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