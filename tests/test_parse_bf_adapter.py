from data.schemas import Chunk
from parse.adapters.bf import (
    answer_to_synthesis,
    chunk_to_knowledge_item,
    prediction_to_analysis,
    retrieval_matches_to_result,
)
from pipeline.prediction_pipeline import PredictionResult


def test_bf_chunk_becomes_traceable_core_knowledge():
    chunk = Chunk(
        content="Sinter reducibility depends on reduction conditions.",
        embed_text="evidence",
        source="sinter.txt",
        topic="sinter",
        section="overview",
    )

    item = chunk_to_knowledge_item(chunk)

    assert item.kind == "chunk"
    assert item.attributes["topic"] == "sinter"
    assert item.provenance.sources[0].source_type == "document"
    assert item.provenance.sources[0].locator == "overview"


def test_bf_retrieval_matches_become_core_evidence():
    chunk = Chunk("Evidence", "Evidence", "sinter.txt", "sinter", "overview")

    result = retrieval_matches_to_result("explain sinter", [(chunk, 0.8)])

    assert result.items[0].score == 0.8
    assert result.items[0].evidence.relation == "retrieved_from"
    assert result.items[0].metadata["source"] == "sinter.txt"


def test_bf_prediction_and_answer_preserve_lineage():
    prediction = PredictionResult(
        predictions={"Ts": 1300.0, "confidence": "medium", "distance": 1.4},
        inputs={"chemistry": {"Basicity": 1.4}},
        parser_warnings=["CO was inferred as a balance value"],
    )
    analysis = prediction_to_analysis(prediction, result_id="analysis-1")
    retrieval = retrieval_matches_to_result(
        "explain",
        [(Chunk("Evidence", "Evidence", "doc.txt", "sinter", "mechanism"), 0.9)],
    )
    synthesis = answer_to_synthesis(
        "The result is supported by the retrieved evidence.",
        retrieval,
        [analysis],
    )

    assert analysis.outputs["Ts"] == 1300.0
    assert analysis.applicability.indicators["distance"] == 1.4
    assert analysis.issues[0].code == "parser_warning"
    assert synthesis.evidence_refs[0].relation == "retrieved_from"
    assert synthesis.analysis_refs[0].ref_id == "analysis-1"
