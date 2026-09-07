"""Minimal, dependency-free PARSE Core contracts.

The Core carries lineage and operation results without knowing a process
vocabulary or an infrastructure implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any, Mapping


_SOURCE_TYPES = {
    "document",
    "csv",
    "spreadsheet",
    "user_input",
    "model",
    "generated_output",
}
_RELATIONS = {
    "observed_from",
    "interpreted_from",
    "derived_from",
    "retrieved_from",
    "compared_with",
}
_KNOWLEDGE_KINDS = {
    "document",
    "chunk",
    "structured_record",
    "observation",
    "interpretation",
}
_ISSUE_SEVERITIES = {"info", "warning", "error"}


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _copy_mapping(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return dict(value)


@dataclass(frozen=True)
class SourceRef:
    """Stable identity and location for an information source."""

    source_id: str
    source_type: str
    locator: str | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.source_id, "source_id")
        if self.source_type not in _SOURCE_TYPES:
            raise ValueError(f"Unsupported source_type: {self.source_type}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceRef":
        return cls(**dict(value))


@dataclass
class Provenance:
    """Lineage for source-derived or derived information."""

    sources: list[SourceRef] = field(default_factory=list)
    operation: str | None = None
    method: str | None = None
    parent_ids: list[str] = field(default_factory=list)
    created_at: str | None = None

    def __post_init__(self) -> None:
        self.sources = list(self.sources)
        self.parent_ids = list(self.parent_ids)
        for source in self.sources:
            if not isinstance(source, SourceRef):
                raise TypeError("sources must contain SourceRef values")
        for parent_id in self.parent_ids:
            _require_text(parent_id, "parent_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sources": [source.to_dict() for source in self.sources],
            "operation": self.operation,
            "method": self.method,
            "parent_ids": list(self.parent_ids),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Provenance":
        data = dict(value)
        data["sources"] = [SourceRef.from_dict(item) for item in data.get("sources", [])]
        return cls(**data)


@dataclass(frozen=True)
class Issue:
    """A validation, interpretation, or operation concern."""

    code: str
    severity: str
    message: str
    field: str | None = None
    source: SourceRef | None = None

    def __post_init__(self) -> None:
        _require_text(self.code, "code")
        _require_text(self.message, "message")
        if self.severity not in _ISSUE_SEVERITIES:
            raise ValueError(f"Unsupported issue severity: {self.severity}")
        if self.source is not None and not isinstance(self.source, SourceRef):
            raise TypeError("source must be a SourceRef")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source"] = self.source.to_dict() if self.source else None
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Issue":
        data = dict(value)
        if data.get("source") is not None:
            data["source"] = SourceRef.from_dict(data["source"])
        return cls(**data)


@dataclass(frozen=True)
class EvidenceRef:
    """Reference from a result to a source or prior result."""

    ref_id: str
    relation: str
    locator: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.ref_id, "ref_id")
        if self.relation not in _RELATIONS:
            raise ValueError(f"Unsupported evidence relation: {self.relation}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceRef":
        return cls(**dict(value))


@dataclass
class KnowledgeItem:
    """Flexible envelope for processed source information."""

    item_id: str
    kind: str
    attributes: dict[str, Any]
    provenance: Provenance
    issues: list[Issue] = field(default_factory=list)
    incomplete: bool = False

    def __post_init__(self) -> None:
        _require_text(self.item_id, "item_id")
        if self.kind not in _KNOWLEDGE_KINDS:
            raise ValueError(f"Unsupported knowledge kind: {self.kind}")
        self.attributes = _copy_mapping(self.attributes, "attributes")
        self.issues = list(self.issues)
        if not isinstance(self.provenance, Provenance):
            raise TypeError("provenance must be a Provenance")
        if any(not isinstance(issue, Issue) for issue in self.issues):
            raise TypeError("issues must contain Issue values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "kind": self.kind,
            "attributes": dict(self.attributes),
            "provenance": self.provenance.to_dict(),
            "issues": [issue.to_dict() for issue in self.issues],
            "incomplete": self.incomplete,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KnowledgeItem":
        data = dict(value)
        data["provenance"] = Provenance.from_dict(data["provenance"])
        data["issues"] = [Issue.from_dict(item) for item in data.get("issues", [])]
        return cls(**data)


@dataclass
class ProcessingResult:
    """Output of a processing operation."""

    items: list[KnowledgeItem] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.items = list(self.items)
        self.issues = list(self.issues)
        if any(not isinstance(item, KnowledgeItem) for item in self.items):
            raise TypeError("items must contain KnowledgeItem values")
        if any(not isinstance(issue, Issue) for issue in self.issues):
            raise TypeError("issues must contain Issue values")


@dataclass(frozen=True)
class RetrievedEvidence:
    """One ranked evidence item returned by retrieval."""

    item_id: str
    score: float
    evidence: EvidenceRef
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.item_id, "item_id")
        if not isinstance(self.score, (int, float)) or not isfinite(self.score):
            raise ValueError("score must be a finite number")
        if not isinstance(self.evidence, EvidenceRef):
            raise TypeError("evidence must be an EvidenceRef")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass
class RetrievalResult:
    """Evidence returned for a query, including fallback information."""

    query: str
    items: list[RetrievedEvidence] = field(default_factory=list)
    fallback: str | None = None
    issues: list[Issue] = field(default_factory=list)

    def __post_init__(self) -> None:
        _require_text(self.query, "query")
        self.items = list(self.items)
        self.issues = list(self.issues)
        if any(not isinstance(item, RetrievedEvidence) for item in self.items):
            raise TypeError("items must contain RetrievedEvidence values")


@dataclass(frozen=True)
class Applicability:
    """Non-calibrated indicators describing analytical applicability."""

    indicators: dict[str, Any] = field(default_factory=dict)
    interpretation: str | None = None
    limitations: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        object.__setattr__(self, "indicators", dict(self.indicators))
        object.__setattr__(self, "limitations", list(self.limitations))


@dataclass
class AnalysisResult:
    """Output of an analysis method or model."""

    result_id: str
    outputs: dict[str, Any]
    input_refs: list[EvidenceRef]
    method: str
    provenance: Provenance
    applicability: Applicability | None = None
    issues: list[Issue] = field(default_factory=list)

    def __post_init__(self) -> None:
        _require_text(self.result_id, "result_id")
        _require_text(self.method, "method")
        self.outputs = _copy_mapping(self.outputs, "outputs")
        self.input_refs = list(self.input_refs)
        self.issues = list(self.issues)
        if not isinstance(self.provenance, Provenance):
            raise TypeError("provenance must be a Provenance")
        if any(not isinstance(ref, EvidenceRef) for ref in self.input_refs):
            raise TypeError("input_refs must contain EvidenceRef values")


@dataclass
class SynthesisResult:
    """Evidence-grounded user-facing synthesis output."""

    result_id: str
    content: str
    evidence_refs: list[EvidenceRef]
    analysis_refs: list[EvidenceRef]
    method: str
    provenance: Provenance
    issues: list[Issue] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _require_text(self.result_id, "result_id")
        _require_text(self.content, "content")
        _require_text(self.method, "method")
        self.evidence_refs = list(self.evidence_refs)
        self.analysis_refs = list(self.analysis_refs)
        self.issues = list(self.issues)
        self.limitations = list(self.limitations)
        if not isinstance(self.provenance, Provenance):
            raise TypeError("provenance must be a Provenance")
        all_refs = self.evidence_refs + self.analysis_refs
        if any(not isinstance(ref, EvidenceRef) for ref in all_refs):
            raise TypeError("result references must contain EvidenceRef values")


@dataclass
class EvaluationResult:
    """Criteria-driven evaluation of another result or operation."""

    target_ref: EvidenceRef
    criteria: list[str]
    metrics: dict[str, Any]
    interpretation: str | None = None
    evidence_refs: list[EvidenceRef] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.target_ref, EvidenceRef):
            raise TypeError("target_ref must be an EvidenceRef")
        self.criteria = list(self.criteria)
        self.metrics = _copy_mapping(self.metrics, "metrics")
        self.evidence_refs = list(self.evidence_refs)
        self.issues = list(self.issues)
        self.limitations = list(self.limitations)
        if any(not isinstance(ref, EvidenceRef) for ref in self.evidence_refs):
            raise TypeError("evidence_refs must contain EvidenceRef values")
