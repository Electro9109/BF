import pandas as pd

from parse.analysis import (
    AnalysisOrchestrator,
    AnalysisRequest,
    DatasetAnalysisAnalyzer,
)
from parse.semantic_analysis import (
    AnalysisMethodSelector,
    AnalysisPipeline,
    ConfirmationRecord,
    HumanContext,
    RelevanceRequest,
)
from parse.eda_ui import analyze_uploaded_dataset
from parse.core import Analyzer, SourceRef


def test_analysis_is_domain_agnostic_read_only_and_evidence_backed():
    frame = pd.DataFrame({
        "record_id": [1, 2, 3, 4, 5],
        "value": [1.0, 2.0, 3.0, 4.0, 100.0],
        "group": ["a", "a", "b", "b", "b"],
        "when": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-05", "2024-01-06"]),
        "empty": [None] * 5,
    })
    original = frame.copy(deep=True)
    source = SourceRef("dataset-1", "csv", "dataset.csv")
    result = AnalysisOrchestrator().analyze(AnalysisRequest(frame, source))

    assert result.dataset_profile.row_count == 5
    assert result.structural_profile.candidate_keys == ("record_id", "value", "when")
    assert result.structural_profile.empty_columns == ("empty",)
    value = next(attribute for attribute in result.attributes if attribute.name == "value")
    assert value.distribution.mean == 22.0
    assert value.distribution.outlier_count == 1
    assert value.distribution.mad == 1.0
    assert next(attribute for attribute in result.attributes if attribute.name == "when").temporal_properties["gap_count"] == 1
    assert any(finding.category == "quality" for finding in result.findings)
    unusual = next(finding for finding in result.findings if finding.finding_id == "unusual:value")
    assert unusual.knowledge_state == "statistical"
    assert "not evidence of error" in unusual.limitations[0]
    assert unusual.evidence[0].ref_id == "dataset-1"
    pd.testing.assert_frame_equal(frame, original)
    assert result.data_modified is False


def test_analysis_selects_numeric_and_categorical_relationship_methods():
    frame = pd.DataFrame({
        "linear": [1, 2, 3, 4, 5],
        "scaled": [2, 4, 6, 8, 10],
        "category_a": ["x", "x", "x", "y", "y"],
        "category_b": ["a", "a", "a", "b", "b"],
    })
    result = AnalysisOrchestrator().analyze(
        AnalysisRequest(frame, SourceRef("dataset-2", "user_input")))

    numeric = next(f for f in result.findings if f.finding_id == "relationship:linear:scaled")
    assert numeric.result["selected_method"] == "pearson"
    assert numeric.result["selection_rationale"]
    assert numeric.result["sample_size"] == 5
    categorical = next(f for f in result.findings if f.finding_id == "association:category_a:category_b")
    assert categorical.method == "contingency_table_cramers_v"
    assert categorical.result["cramers_v"] > 0.3
    assert "causation" in categorical.limitations[0]


def test_analyzer_adapter_implements_generic_parse_capability():
    analyzer = DatasetAnalysisAnalyzer()
    assert isinstance(analyzer, Analyzer)
    result = analyzer.analyze(
        [pd.DataFrame({"x": [1, 2, 3]})],
        source=SourceRef("dataset-3", "user_input"),
    )
    assert result.method == "parse_eda_v1"
    assert result.outputs["dataset_profile"]["row_count"] == 3
    assert result.input_refs[0].relation == "observed_from"


def test_semantic_candidates_are_uncertain_and_human_confirmation_is_explicit():
    frame = pd.DataFrame({"record_id": [1, 2, 3], "temperature_kg": [10.0, 11.0, 12.0], "comment": [
        "first free-form observation", "second free-form observation", "third free-form observation",
    ]})
    request = AnalysisRequest(frame, SourceRef("dataset-4", "user_input"))
    bundle = AnalysisPipeline().analyze(request)

    unit = next(candidate for candidate in bundle.semantic.candidates if candidate.candidate_id == "temperature_kg:unit")
    assert unit.knowledge_state == "inferred"
    assert unit.confidence < 1
    assert unit.evidence[0].relation == "derived_from"
    assert any("Meaning of attribute 'comment'" in unknown for unknown in bundle.semantic.unknowns)

    context = HumanContext()
    context.apply(ConfirmationRecord("review-1", unit.candidate_id, "confirm", "temperature_kg", "temperature", "reviewer"))
    assert context.state_for(unit.candidate_id) == "confirmed"
    assert context.confirmations[0].created_at
    context.apply(ConfirmationRecord("review-2", unit.candidate_id, "conflict", "temperature_kg", None, "reviewer"))
    assert context.state_for(unit.candidate_id) == "conflicting"


def test_relevance_is_task_specific_and_summary_is_structured_evidence():
    frame = pd.DataFrame({"target": [1, 2, 3, 4], "support": [2, 4, 6, 8], "other": [4, 1, 4, 1]})
    bundle = AnalysisPipeline().analyze(
        AnalysisRequest(frame, SourceRef("dataset-5", "user_input")),
        RelevanceRequest("predict target", selected_attributes=("target",)),
    )
    target = next(item for item in bundle.relevance.candidates if item.attribute == "target")
    assert target.category == "potentially_relevant"
    assert target.knowledge_state == "inferred"
    assert "predict target" in bundle.summary
    assert "does not establish domain meaning" in bundle.summary


def test_method_selector_explains_insufficient_and_selected_methods():
    selector = AnalysisMethodSelector()
    assert selector.select_numeric_relationship(2, .99, .99).method == "insufficient_data"
    selected = selector.select_numeric_relationship(20, .2, .9)
    assert selected.method == "spearman"
    assert selected.rationale


def test_upload_boundary_exposes_complete_analysis_bundle():
    content = b"target,group\n1,a\n2,b\n3,b\n"
    bundle = analyze_uploaded_dataset("sample.csv", content, "understand target")

    assert bundle.eda.request.source.source_type == "csv"
    assert bundle.semantic.candidates
    assert bundle.relevance.request.objective == "understand target"