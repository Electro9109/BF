"""Capability protocols for the minimal PARSE Core.

These protocols describe boundaries only. Implementations belong to domain or
infrastructure modules and may use any suitable technology.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from parse.core.contracts import (
    AnalysisResult,
    EvaluationResult,
    KnowledgeItem,
    ProcessingResult,
    RetrievalResult,
    SynthesisResult,
)


@runtime_checkable
class Processor(Protocol):
    def process(self, source: Any) -> ProcessingResult:
        """Convert a source into traceable knowledge items."""


@runtime_checkable
class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        candidates: Sequence[KnowledgeItem] | None = None,
    ) -> RetrievalResult:
        """Return ranked evidence references for a query."""


@runtime_checkable
class Analyzer(Protocol):
    def analyze(
        self,
        inputs: Sequence[KnowledgeItem],
        **context: Any,
    ) -> AnalysisResult:
        """Produce a traceable analytical result."""


@runtime_checkable
class Synthesizer(Protocol):
    def synthesize(
        self,
        evidence: RetrievalResult,
        analyses: Sequence[AnalysisResult] = (),
        **context: Any,
    ) -> SynthesisResult:
        """Combine evidence and analysis into a traceable result."""


@runtime_checkable
class Evaluator(Protocol):
    def evaluate(
        self,
        target: Any,
        criteria: Sequence[str],
        **context: Any,
    ) -> EvaluationResult:
        """Evaluate a result against explicit criteria."""
