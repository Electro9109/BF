import ast
from pathlib import Path

import pytest

from parse.core import (
    AnalysisResult,
    Applicability,
    EvidenceRef,
    EvaluationResult,
    Issue,
    KnowledgeItem,
    Provenance,
    RetrievedEvidence,
    SourceRef,
    SynthesisResult,
)
from parse.core import Analyzer, Evaluator, Processor, Retriever, Synthesizer
from parse.core import run_workflow


ROOT = Path(__file__).resolve().parent.parent


def test_core_contracts_preserve_lineage_and_round_trip_knowledge():
    source = SourceRef("doc-1", "document", "sinter.txt#overview")
    provenance = Provenance(sources=[source], operation="parse_document")
    issue = Issue("missing_value", "warning", "A value was not supplied", field="FeO")
    item = KnowledgeItem(
        item_id="chunk-1",
        kind="observation",
        attributes={"value": 42, "unit": "C"},
        provenance=provenance,
        issues=[issue],
        incomplete=True,
    )

    restored = KnowledgeItem.from_dict(item.to_dict())

    assert restored.kind == "observation"
    assert restored.attributes == item.attributes
    assert restored.provenance.sources[0].locator == "sinter.txt#overview"
    assert restored.issues[0].code == "missing_value"
    assert restored.incomplete is True


def test_core_distinguishes_interpretation_from_observation():
    provenance = Provenance(operation="condition_parser")
    observation = KnowledgeItem("raw-1", "observation", {"text": "CO=40%"}, provenance)
    interpretation = KnowledgeItem(
        "parsed-1",
        "interpretation",
        {"CO_pct": 40.0},
        Provenance(operation="condition_parser", parent_ids=[observation.item_id]),
    )

    assert observation.kind != interpretation.kind
    assert interpretation.provenance.parent_ids == ["raw-1"]


def test_results_require_evidence_and_keep_applicability_separate():
    provenance = Provenance(operation="model_prediction")
    input_ref = EvidenceRef("experiment-1", "observed_from")
    analysis = AnalysisResult(
        result_id="analysis-1",
        outputs={"temperature": 1300.0},
        input_refs=[input_ref],
        method="example-model",
        provenance=provenance,
        applicability=Applicability(
            indicators={"nearest_distance": 1.2},
            limitations=["Small training dataset"],
        ),
    )
    synthesis = SynthesisResult(
        result_id="synthesis-1",
        content="The result is supported by the supplied evidence.",
        evidence_refs=[input_ref],
        analysis_refs=[EvidenceRef("analysis-1", "derived_from")],
        method="template-synthesis",
        provenance=Provenance(operation="synthesis", parent_ids=["analysis-1"]),
    )
    evaluation = EvaluationResult(
        target_ref=EvidenceRef("analysis-1", "derived_from"),
        criteria=["applicability"],
        metrics={"reviewed": False},
    )

    assert analysis.applicability.indicators["nearest_distance"] == 1.2
    assert synthesis.analysis_refs[0].ref_id == "analysis-1"
    assert evaluation.target_ref.ref_id == "analysis-1"


def test_core_rejects_invalid_contract_values():
    with pytest.raises(ValueError, match="source_type"):
        SourceRef("source", "unknown")
    with pytest.raises(ValueError, match="severity"):
        Issue("problem", "critical", "Unsupported severity")
    with pytest.raises(ValueError, match="finite"):
        RetrievedEvidence("item", float("nan"), EvidenceRef("item", "retrieved_from"))


def test_core_has_no_forbidden_implementation_dependencies():
    forbidden = {
        "sinter",
        "bf",
        "faiss",
        "qwen",
        "sentence_transformers",
        "streamlit",
        "transformers",
    }
    core_path = ROOT / "parse" / "core"

    for path in core_path.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name.split(".")[0].lower() for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [node.module.split(".")[0].lower()] if node.module else []
            else:
                continue
            assert forbidden.isdisjoint(imported), f"Forbidden dependency in {path}: {imported}"


def test_operation_protocols_are_implementation_neutral():
    class ExampleProcessor:
        def process(self, source):
            return None

    class ExampleRetriever:
        def retrieve(self, query, candidates=None):
            return None

    class ExampleAnalyzer:
        def analyze(self, inputs, **context):
            return None

    class ExampleSynthesizer:
        def synthesize(self, evidence, analyses=(), **context):
            return None

    class ExampleEvaluator:
        def evaluate(self, target, criteria, **context):
            return None

    assert isinstance(ExampleProcessor(), Processor)
    assert isinstance(ExampleRetriever(), Retriever)
    assert isinstance(ExampleAnalyzer(), Analyzer)
    assert isinstance(ExampleSynthesizer(), Synthesizer)
    assert isinstance(ExampleEvaluator(), Evaluator)


def test_core_workflow_moves_traceable_outputs_through_all_operations():
    source_ref = SourceRef("source-1", "user_input")
    provenance = Provenance(sources=[source_ref], operation="test")
    item = KnowledgeItem("item-1", "observation", {"value": 1}, provenance)
    input_ref = EvidenceRef("item-1", "observed_from")

    class ExampleProcessor:
        def process(self, source):
            return __import__("parse.core", fromlist=["ProcessingResult"]).ProcessingResult([item])

    class ExampleRetriever:
        def retrieve(self, query, candidates=None):
            from parse.core import RetrievedEvidence, RetrievalResult
            return RetrievalResult(
                query=query,
                items=[RetrievedEvidence("item-1", 1.0, input_ref)],
            )

    class ExampleAnalyzer:
        def analyze(self, inputs, **context):
            return AnalysisResult(
                "analysis-1",
                {"value": 1},
                [input_ref],
                "test-analysis",
                provenance,
            )

    class ExampleSynthesizer:
        def synthesize(self, evidence, analyses=(), **context):
            return SynthesisResult(
                "synthesis-1",
                "Evidence-backed result",
                [input_ref],
                [EvidenceRef("analysis-1", "derived_from")],
                "test-synthesis",
                provenance,
            )

    class ExampleEvaluator:
        def evaluate(self, target, criteria, **context):
            return EvaluationResult(
                EvidenceRef("synthesis-1", "derived_from"),
                list(criteria),
                {"status": "unreviewed"},
            )

    result = run_workflow(
        source="source",
        query="test",
        processor=ExampleProcessor(),
        retriever=ExampleRetriever(),
        analyzer=ExampleAnalyzer(),
        synthesizer=ExampleSynthesizer(),
        evaluator=ExampleEvaluator(),
        criteria=["grounding"],
    )

    assert result.processing.items[0].item_id == "item-1"
    assert result.retrieval.items[0].item_id == "item-1"
    assert result.analyses[0].result_id == "analysis-1"
    assert result.synthesis.result_id == "synthesis-1"
    assert result.evaluation.criteria == ["grounding"]
