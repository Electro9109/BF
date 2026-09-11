import pandas as pd
import pytest

from parse.eda import DataUnderstanding
from parse.core import SourceRef


def test_eda_profiles_structure_quality_and_actions_without_mutating_data():
    frame = pd.DataFrame(
        {
            "record_id": [1, 2, 3, 4],
            "temperature": [100.0, 110.0, 110.0, None],
            "group": ["A", "A", "B", "B"],
            "notes": ["ok", "ok", "repeat", "missing"],
        }
    )
    original = frame.copy(deep=True)
    result = DataUnderstanding(SourceRef("dataset-a", "csv", "a.csv")).profile(frame)

    assert result.row_count == 4
    assert result.column_count == 4
    assert result.data_modified is False
    pd.testing.assert_frame_equal(frame, original)
    assert result.source.source_id == "dataset-a"
    assert next(column for column in result.columns if column.name == "temperature").inferred_type == "numeric"
    assert any(finding.category == "quality" for finding in result.findings)
    assert any(finding.kind == "interpretation" and "record_id" in finding.message for finding in result.findings)
    assert any(action.action == "clean_prepare" and action.applicable for action in result.next_actions)
    assert any(action.action == "analyse" and action.applicable for action in result.next_actions)
    assert result.to_dict()["provenance"]["operation"] == "structured_data_understanding"


def test_eda_handles_categorical_text_and_relationships_cautiously():
    frame = pd.DataFrame(
        {
            "category": ["x", "x", "y", "y"],
            "signal_a": [1.0, 2.0, 3.0, 4.0],
            "signal_b": [2.0, 4.0, 6.0, 8.0],
        }
    )
    result = DataUnderstanding().profile(frame)

    category = next(column for column in result.columns if column.name == "category")
    assert category.inferred_type == "categorical"
    relationship = next(finding for finding in result.findings if finding.category == "relationship")
    assert relationship.attributes["correlation"] > 0.99
    assert "causation" in relationship.limitations[0]
    assert any(action.action == "explore_retrieve" and action.applicable for action in result.next_actions)
    assert any(action.action == "generate_report" and action.applicable for action in result.next_actions)


def test_eda_reports_distribution_temporal_and_group_findings():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "group": ["a", "a", "b", "b"],
            "measurement": [1.0, 1.0, 1.0, 100.0],
        }
    )

    result = DataUnderstanding().profile(frame)
    categories = {finding.category for finding in result.findings}

    assert "statistical" in categories
    assert any("appears temporal" in finding.message for finding in result.findings)
    assert any("differs across" in finding.message for finding in result.findings)
    measurement = next(column for column in result.columns if column.name == "measurement")
    assert measurement.summary["outlier_count"] == 1
    assert measurement.summary["q1"] == 1.0


def test_eda_marks_uncertain_numeric_text_types():
    frame = pd.DataFrame({"value": ["1.0", "2.0", "unknown"]})

    result = DataUnderstanding().profile(frame)
    value = result.columns[0]

    assert value.inferred_type == "categorical"
    assert value.type_confidence < 1.0


def test_eda_reports_mixed_numeric_text_and_empty_input():
    mixed = DataUnderstanding().profile(pd.DataFrame({"value": ["1", "unknown", "3"]}))
    empty = DataUnderstanding().profile(pd.DataFrame(columns=["value"]))

    assert any(finding.category == "quality" and "mixes numeric-like" in finding.message for finding in mixed.findings)
    assert any(issue.code == "empty_dataset" for issue in empty.issues)


def test_eda_does_not_recommend_prediction_without_candidate_target():
    frame = pd.DataFrame({"name": ["a", "b"], "value": [1, 2]})
    result = DataUnderstanding().profile(frame)

    prediction = next(action for action in result.next_actions if action.action == "build_prediction")
    assert prediction.applicable is False
    assert "target" in prediction.reason


def test_eda_reads_csv_without_modifying_the_source(tmp_path):
    source = tmp_path / "dataset.csv"
    pd.DataFrame({"id": [1, 2], "target": [10.0, 11.0]}).to_csv(source, index=False)

    result = DataUnderstanding.from_file(source)

    assert result.source.source_type == "csv"
    assert result.source.label == "dataset.csv"
    assert result.row_count == 2
    assert any(column.candidate_target for column in result.columns)


def test_eda_rejects_unsupported_file_formats(tmp_path):
    source = tmp_path / "dataset.json"
    source.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported dataset format"):
        DataUnderstanding.from_file(source)
