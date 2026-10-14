import ast
import json
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
from parse.core import ResourceManifest, create_resource_manifest, run_workflow


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


def test_resource_manifest_creates_valid_json_and_round_trips(tmp_path):
    source = SourceRef("source-7", "document", "docs/example.txt", "Example note")
    provenance = Provenance(sources=[source], operation="manual_registration")
    manifest_path = tmp_path / "resource_manifest.json"

    manifest = create_resource_manifest(
        {
            "resource_class": "private_operational_data",
            "content_type": "document",
            "title": "Sinter process document",
            "source": source.to_dict(),
            "contexts": ["sinter", "reducibility"],
            "provenance": provenance.to_dict(),
            "processing_state": "registered",
            "metadata": {"owner": "ops-team", "notes": "Example only"},
        },
        manifest_path,
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    restored = ResourceManifest.from_json(manifest_path)

    assert isinstance(manifest, ResourceManifest)
    assert manifest.resource_id == restored.resource_id
    assert payload["resource_class"] == "private_operational_data"
    assert payload["source"]["source_id"] == "source-7"
    assert payload["contexts"] == ["sinter", "reducibility"]
    assert payload["metadata"]["owner"] == "ops-team"
    assert restored.title == "Sinter process document"


def test_resource_manifest_reuses_stable_id_for_equivalent_resources(tmp_path):
    source = SourceRef("source-8", "document", "docs/another.txt", "Another note")
    metadata = {
        "resource_class": "public_research",
        "content_type": "document",
        "title": "Public paper",
        "source": source.to_dict(),
        "contexts": ["reduction", "pellet"],
    }

    first = create_resource_manifest(metadata, tmp_path / "first.json")
    second = create_resource_manifest(metadata, tmp_path / "second.json")

    assert first.resource_id == second.resource_id
    assert first.resource_id.startswith("public_research-")


def test_resource_manifest_rejects_invalid_or_missing_fields(tmp_path):
    with pytest.raises(ValueError, match="resource_class"):
        create_resource_manifest({"content_type": "document", "title": "Missing type"}, tmp_path / "missing_class.json")

    with pytest.raises(ValueError, match="content_type"):
        create_resource_manifest({"resource_class": "public_dataset", "title": "Missing content"}, tmp_path / "missing_content.json")

    with pytest.raises(TypeError, match="contexts"):
        create_resource_manifest(
            {
                "resource_class": "internal_technical_document",
                "content_type": "text",
                "title": "Bad contexts",
                "contexts": "not-a-list",
            },
            tmp_path / "bad_contexts.json",
        )


def test_resource_manifest_refuses_to_overwrite_existing_file(tmp_path):
    manifest_path = tmp_path / "existing.json"
    manifest_path.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        create_resource_manifest(
            {
                "resource_class": "derived_analysis_result",
                "content_type": "derived_artifact",
                "title": "Existing file",
            },
            manifest_path,
        )


def test_resource_manifest_cleans_temp_file_on_write_failure(tmp_path, monkeypatch):
    import parse.core.resource_manifest as resource_manifest_module

    destination = tmp_path / "write_fail.json"

    def raise_replace(_src, _dst):
        raise OSError("disk full")

    monkeypatch.setattr(resource_manifest_module.os, "replace", raise_replace)

    with pytest.raises(OSError, match="disk full"):
        create_resource_manifest(
            {
                "resource_class": "public_dataset",
                "content_type": "csv",
                "title": "Dataset manifest",
                "source": {"source_id": "dataset-9", "source_type": "csv", "locator": "public/data.csv"},
            },
            destination,
        )

    assert not destination.exists()
    assert not (tmp_path / ".write_fail.json.tmp").exists()


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
