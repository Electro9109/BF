"""Adapters that keep the existing BF implementation behind PARSE Core."""

from __future__ import annotations

import hashlib
from typing import Iterable

from data.schemas import Chunk
from parse.core.contracts import (
    AnalysisResult,
    Applicability,
    EvidenceRef,
    Issue,
    KnowledgeItem,
    Provenance,
    RetrievalResult,
    RetrievedEvidence,
    SourceRef,
    SynthesisResult,
)
from pipeline.prediction_pipeline import PredictionResult


def _chunk_id(chunk: Chunk) -> str:
    identity = f"{chunk.source}::{chunk.section}::{chunk.content}"
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]
    return f"bf-chunk-{digest}"


def chunk_to_knowledge_item(chunk: Chunk) -> KnowledgeItem:
    """Represent an existing BF chunk as a traceable Core knowledge item."""
    source = SourceRef(
        source_id=f"bf-document:{chunk.source}",
        source_type="document",
        locator=chunk.section,
        label=chunk.source,
    )
    return KnowledgeItem(
        item_id=_chunk_id(chunk),
        kind="chunk",
        attributes={
            "content": chunk.content,
            "topic": chunk.topic,
            "section": chunk.section,
        },
        provenance=Provenance(
            sources=[source],
            operation="bf_document_chunking",
            parent_ids=[chunk.source],
        ),
    )


def retrieval_matches_to_result(
    query: str,
    matches: Iterable[tuple[Chunk, float]],
) -> RetrievalResult:
    """Convert BF RetrievalEngine matches into Core retrieval evidence."""
    items = []
    for chunk, score in matches:
        item = chunk_to_knowledge_item(chunk)
        items.append(
            RetrievedEvidence(
                item_id=item.item_id,
                score=float(score),
                evidence=EvidenceRef(
                    item.item_id,
                    "retrieved_from",
                    locator=chunk.section,
                ),
                metadata={"source": chunk.source, "topic": chunk.topic},
            )
        )
    fallback = "no_matches" if not items else None
    issues = []
    if fallback:
        issues.append(Issue("no_matches", "warning", "BF retrieval returned no evidence"))
    return RetrievalResult(query=query, items=items, fallback=fallback, issues=issues)


def prediction_to_analysis(
    result: PredictionResult,
    result_id: str = "bf-analysis",
) -> AnalysisResult:
    """Convert a BF prediction result into a Core analysis result."""
    source = SourceRef("bf-prediction-input", "user_input")
    input_ref = EvidenceRef("bf-prediction-input", "observed_from")
    issues = [
        Issue("parser_warning", "warning", warning)
        for warning in result.parser_warnings
    ]
    predictions = dict(result.predictions)
    indicators = {
        key: predictions[key]
        for key in ("confidence", "distance")
        if key in predictions
    }
    applicability = Applicability(
        indicators=indicators,
        limitations=["Model performance is limited by the available Sinter dataset."],
    )
    return AnalysisResult(
        result_id=result_id,
        outputs=predictions,
        input_refs=[input_ref],
        method="bf-sinter-predictor",
        provenance=Provenance(
            sources=[source],
            operation="bf_prediction",
            method="bf-sinter-predictor",
            parent_ids=[result_id],
        ),
        applicability=applicability,
        issues=issues,
    )


def answer_to_synthesis(
    answer: str,
    retrieval: RetrievalResult,
    analyses: Iterable[AnalysisResult] = (),
    result_id: str = "bf-synthesis",
) -> SynthesisResult:
    """Convert a grounded BF answer into a Core synthesis result."""
    analysis_list = list(analyses)
    evidence_refs = [item.evidence for item in retrieval.items]
    analysis_refs = [
        EvidenceRef(analysis.result_id, "derived_from")
        for analysis in analysis_list
    ]
    issues = list(retrieval.issues)
    if not answer.strip():
        issues.append(Issue("empty_synthesis", "error", "BF synthesis returned empty content"))
    return SynthesisResult(
        result_id=result_id,
        content=answer,
        evidence_refs=evidence_refs,
        analysis_refs=analysis_refs,
        method="bf-grounded-generation",
        provenance=Provenance(
            operation="bf_synthesis",
            method="bf-grounded-generation",
            parent_ids=[ref.ref_id for ref in evidence_refs + analysis_refs],
        ),
        issues=issues,
        limitations=["Generated text is a derived explanation, not source authority."],
    )
