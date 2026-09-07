"""Dependency-light PARSE Core contracts."""

from parse.core.contracts import (
    AnalysisResult,
    Applicability,
    EvidenceRef,
    EvaluationResult,
    Issue,
    KnowledgeItem,
    ProcessingResult,
    Provenance,
    RetrievalResult,
    RetrievedEvidence,
    SourceRef,
    SynthesisResult,
)
from parse.core.operations import Analyzer, Evaluator, Processor, Retriever, Synthesizer
from parse.core.workflow import WorkflowResult, run_workflow

__all__ = [
    "AnalysisResult",
    "Applicability",
    "EvidenceRef",
    "EvaluationResult",
    "Issue",
    "KnowledgeItem",
    "ProcessingResult",
    "Provenance",
    "RetrievalResult",
    "RetrievedEvidence",
    "SourceRef",
    "SynthesisResult",
    "Analyzer",
    "Evaluator",
    "Processor",
    "Retriever",
    "Synthesizer",
    "WorkflowResult",
    "run_workflow",
]
