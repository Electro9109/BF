"""Small orchestration helper for the PARSE Core operation chain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from parse.core.contracts import (
    AnalysisResult,
    EvaluationResult,
    KnowledgeItem,
    ProcessingResult,
    RetrievalResult,
    SynthesisResult,
)
from parse.core.operations import Analyzer, Evaluator, Processor, Retriever, Synthesizer


@dataclass
class WorkflowResult:
    """Traceable outputs from one processing-to-evaluation run."""

    processing: ProcessingResult
    retrieval: RetrievalResult
    analyses: list[AnalysisResult] = field(default_factory=list)
    synthesis: SynthesisResult | None = None
    evaluation: EvaluationResult | None = None


def run_workflow(
    source: Any,
    query: str,
    processor: Processor,
    retriever: Retriever,
    synthesizer: Synthesizer,
    evaluator: Evaluator,
    analyzer: Analyzer | None = None,
    criteria: Sequence[str] = (),
) -> WorkflowResult:
    """Run the minimal PARSE chain with explicit operation collaborators."""
    processing = processor.process(source)
    retrieval = retriever.retrieve(query, processing.items)

    analyses: list[AnalysisResult] = []
    if analyzer is not None:
        analyses.append(analyzer.analyze(processing.items, retrieval=retrieval))

    synthesis = synthesizer.synthesize(retrieval, analyses)
    evaluation = evaluator.evaluate(
        synthesis,
        criteria,
        retrieval=retrieval,
        analyses=analyses,
    )
    return WorkflowResult(
        processing=processing,
        retrieval=retrieval,
        analyses=analyses,
        synthesis=synthesis,
        evaluation=evaluation,
    )
