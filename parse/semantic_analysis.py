"""Semantic, relevance, context, and synthesis stages for dataset analysis.

The implementation generates candidates from observable evidence. It never
turns a name heuristic or statistical association into confirmed domain truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import re
from typing import Any, Mapping, Sequence

from parse.analysis import (
    AnalysisMethodSelector,
    AnalysisRequest,
    AnalysisOrchestrator,
    EDAResult,
    Finding,
    _json_value,
)
from parse.core.contracts import EvidenceRef, Provenance, SourceRef


@dataclass(frozen=True)
class SemanticCandidate:
    candidate_id: str
    attribute: str
    candidate_meaning: str
    role: str
    evidence: tuple[EvidenceRef, ...]
    method: str
    assumptions: tuple[str, ...] = ()
    confidence: float | None = None
    knowledge_state: str = "inferred"
    conflicting_candidate_ids: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.knowledge_state not in {"inferred", "confirmed", "unknown", "conflicting"}:
            raise ValueError("semantic candidates must be inferred, confirmed, unknown, or conflicting")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["evidence"] = [item.to_dict() for item in self.evidence]
        return _json_value(value)


@dataclass(frozen=True)
class SemanticResult:
    source: SourceRef
    candidates: tuple[SemanticCandidate, ...]
    findings: tuple[Finding, ...]
    unknowns: tuple[str, ...]
    provenance: Provenance

    def to_dict(self) -> dict[str, Any]:
        return _json_value({
            "source": self.source.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "findings": [finding.to_dict() for finding in self.findings],
            "unknowns": list(self.unknowns),
            "provenance": self.provenance.to_dict(),
        })


@dataclass(frozen=True)
class ConfirmationRecord:
    confirmation_id: str
    candidate_id: str
    action: str
    attribute: str
    meaning: str | None
    reviewer: str
    note: str | None = None
    created_at: str = ""

    def __post_init__(self) -> None:
        if self.action not in {"confirm", "reject", "modify", "unknown", "conflict"}:
            raise ValueError("unsupported confirmation action")
        if not self.reviewer.strip():
            raise ValueError("reviewer must be non-empty")
        if not self.created_at:
            object.__setattr__(self, "created_at", datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass
class HumanContext:
    """Explicit contextual knowledge; raw data and EDA results remain unchanged."""

    confirmations: list[ConfirmationRecord] = field(default_factory=list)

    def apply(self, record: ConfirmationRecord) -> None:
        self.confirmations.append(record)

    def state_for(self, candidate_id: str) -> str | None:
        records = [record for record in self.confirmations if record.candidate_id == candidate_id]
        if not records:
            return None
        return {"confirm": "confirmed", "modify": "confirmed", "reject": "unknown",
                "unknown": "unknown", "conflict": "conflicting"}[records[-1].action]

    def to_dict(self) -> dict[str, Any]:
        return {"confirmations": [record.to_dict() for record in self.confirmations]}


@dataclass(frozen=True)
class RelevanceRequest:
    objective: str
    selected_attributes: tuple[str, ...] = ()
    confirmed_context: HumanContext | None = None
    domain_context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.objective.strip():
            raise ValueError("objective must be non-empty")
        object.__setattr__(self, "selected_attributes", tuple(self.selected_attributes))
        object.__setattr__(self, "domain_context", dict(self.domain_context))


@dataclass(frozen=True)
class RelevanceCandidate:
    attribute: str
    category: str
    score: float
    evidence: tuple[EvidenceRef, ...]
    rationale: str
    knowledge_state: str = "inferred"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["evidence"] = [item.to_dict() for item in self.evidence]
        return _json_value(value)


@dataclass(frozen=True)
class RelevanceResult:
    request: RelevanceRequest
    candidates: tuple[RelevanceCandidate, ...]
    limitations: tuple[str, ...]
    provenance: Provenance

    def to_dict(self) -> dict[str, Any]:
        return _json_value({
            "objective": self.request.objective,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "limitations": list(self.limitations),
            "provenance": self.provenance.to_dict(),
        })


class SemanticAnalyzer:
    """Generate cautious meaning candidates from names, profiles, and context."""

    _UNIT_SUFFIXES = {"kg", "g", "mg", "mm", "cm", "m", "s", "min", "h", "hz", "%"}

    def analyze(self, eda: EDAResult, context: Mapping[str, Any] | None = None) -> SemanticResult:
        supplied = dict(context or eda.request.context)
        candidates: list[SemanticCandidate] = []
        findings: list[Finding] = []
        unknowns: list[str] = []
        for attribute in eda.attributes:
            evidence = (EvidenceRef(eda.request.source.source_id, "derived_from", locator=attribute.name),)
            generated = self._candidates(attribute.name, attribute, evidence, supplied)
            if not generated:
                unknowns.append(f"Meaning of attribute '{attribute.name}' is not established.")
            candidates.extend(generated)
            for candidate in generated:
                findings.append(Finding(
                    finding_id=f"semantic:{candidate.candidate_id}", category="semantic",
                    subject=attribute.name, observation=f"Candidate meaning: {candidate.candidate_meaning}.",
                    method=candidate.method, evidence=candidate.evidence,
                    assumptions=candidate.assumptions, interpretation=candidate.candidate_meaning,
                    knowledge_state=candidate.knowledge_state, limitations=candidate.limitations,
                    result={"role": candidate.role, "confidence": candidate.confidence},
                ))
        return SemanticResult(
            source=eda.request.source, candidates=tuple(candidates), findings=tuple(findings),
            unknowns=tuple(unknowns), provenance=Provenance(
                sources=[eda.request.source], parent_ids=[eda.request.source.source_id],
                operation="semantic_candidate_analysis", method="evidence_heuristics_v1",
                created_at=datetime.now(timezone.utc).isoformat()),
        )

    def _candidates(self, name: str, attribute: Any, evidence: tuple[EvidenceRef, ...], context: Mapping[str, Any]) -> list[SemanticCandidate]:
        candidates: list[SemanticCandidate] = []
        supplied = context.get("attributes", {})
        if isinstance(supplied, Mapping) and name in supplied:
            value = supplied[name]
            meaning = value if isinstance(value, str) else value.get("meaning") if isinstance(value, Mapping) else None
            if meaning:
                candidates.append(self._candidate(name, "context", str(meaning), "context", evidence,
                                                  "user_provided_context", 1.0, ("The supplied context has not been independently verified.",)))
        roles = set(attribute.structural_roles)
        if "candidate_identifier" in roles:
            candidates.append(self._candidate(name, "entity", "record or entity identifier", "entity", evidence,
                                              "structural_role_candidate", .7, ("Uniqueness and naming do not prove entity identity.",)))
        lowered = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        tokens = set(lowered.split("_"))
        if tokens & {"date", "datetime", "timestamp", "time"} or attribute.observed_type == "temporal":
            candidates.append(self._candidate(name, "temporal", "time-related attribute", "temporal", evidence,
                                              "name_and_observed_type", .65, ("Temporal values do not establish the time semantics.",)))
        unit = next((token for token in tokens if token in self._UNIT_SUFFIXES), None)
        if unit:
            candidates.append(self._candidate(name, "unit", f"values may use unit '{unit}'", "unit", evidence,
                                              "attribute_name_pattern", .55, ("Unit tokens in names may be abbreviated or inconsistent.",)))
        if attribute.observed_type == "numeric":
            candidates.append(self._candidate(name, "metric", "numeric property or measurement", "metric", evidence,
                                              "observed_numeric_type", .5, ("Numeric representation alone does not establish measurement meaning.",)))
        elif attribute.observed_type in {"categorical", "boolean"}:
            candidates.append(self._candidate(name, "categorical", "categorical or state property", "property", evidence,
                                              "observed_value_domain", .45, ("Categories require contextual interpretation.",)))
        return candidates

    @staticmethod
    def _candidate(name: str, role: str, meaning: str, kind: str, evidence: tuple[EvidenceRef, ...],
                   method: str, confidence: float, limitations: tuple[str, ...]) -> SemanticCandidate:
        return SemanticCandidate(f"{name}:{role}", name, meaning, kind, evidence, method,
                                 confidence=confidence, limitations=limitations)


class RelevanceAnalyzer:
    """Score relevance only for one explicit operation objective."""

    def analyze(self, eda: EDAResult, semantic: SemanticResult, request: RelevanceRequest) -> RelevanceResult:
        objective_tokens = set(re.findall(r"[a-z0-9]+", request.objective.lower()))
        candidates: list[RelevanceCandidate] = []
        relationships: dict[str, int] = {}
        for finding in eda.findings:
            if finding.category == "relationship":
                for subject in finding.subject if isinstance(finding.subject, tuple) else (finding.subject,):
                    relationships[str(subject)] = relationships.get(str(subject), 0) + 1
        for attribute in eda.attributes:
            tokens = set(re.findall(r"[a-z0-9]+", attribute.name.lower()))
            score = 0.0
            reasons: list[str] = []
            category = "requires_clarification"
            if attribute.name in request.selected_attributes:
                score += .8
                reasons.append("selected by the operation requester")
                category = "selected"
            if tokens & objective_tokens:
                score += .4
                reasons.append("attribute name overlaps the operation objective")
            if relationships.get(attribute.name):
                score += min(.3, .1 * relationships[attribute.name])
                reasons.append("related to another observed attribute")
            if attribute.missing_rate >= .5:
                score -= .2
                reasons.append("high missingness limits current usefulness")
            if score >= .7:
                category = "potentially_relevant"
            elif score <= .1 and not reasons:
                category = "potentially_irrelevant"
                reasons.append("no evidence connected it to this objective")
            candidates.append(RelevanceCandidate(
                attribute=attribute.name, category=category, score=max(0.0, min(1.0, score)),
                evidence=(EvidenceRef(eda.request.source.source_id, "derived_from", locator=attribute.name),),
                rationale="; ".join(reasons), knowledge_state="inferred"))
        return RelevanceResult(request, tuple(candidates),
            ("Relevance is specific to this request and must not be stored as a global attribute property.",
             "Observed association does not establish task usefulness or causation."),
            Provenance(sources=[eda.request.source], operation="task_relevance_analysis",
                       method="objective_and_evidence_v1", parent_ids=[eda.request.source.source_id],
                       created_at=datetime.now(timezone.utc).isoformat()))


@dataclass(frozen=True)
class AnalysisBundle:
    eda: EDAResult
    semantic: SemanticResult
    relevance: RelevanceResult | None
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {"eda": self.eda.to_dict(), "semantic": self.semantic.to_dict(),
                "relevance": self.relevance.to_dict() if self.relevance else None, "summary": self.summary}


class AnalysisPipeline:
    """Run EDA, semantic candidates, optional task relevance, and synthesis."""

    def __init__(self, eda: AnalysisOrchestrator | None = None,
                 semantic: SemanticAnalyzer | None = None,
                 relevance: RelevanceAnalyzer | None = None) -> None:
        self.eda = eda or AnalysisOrchestrator()
        self.semantic = semantic or SemanticAnalyzer()
        self.relevance = relevance or RelevanceAnalyzer()

    def analyze(self, request: AnalysisRequest, relevance: RelevanceRequest | None = None) -> AnalysisBundle:
        eda_result = self.eda.analyze(request)
        semantic_result = self.semantic.analyze(eda_result)
        relevance_result = self.relevance.analyze(eda_result, semantic_result, relevance) if relevance else None
        return AnalysisBundle(eda_result, semantic_result, relevance_result,
                              AnalysisSynthesizer().summarize(eda_result, semantic_result, relevance_result))


class AnalysisSynthesizer:
    """Produce a human-readable summary from structured evidence only."""

    @staticmethod
    def summarize(eda: EDAResult, semantic: SemanticResult,
                  relevance: RelevanceResult | None = None) -> str:
        quality = [f.observation for f in eda.findings if f.category == "quality"]
        relationships = [f.observation for f in eda.findings if f.category == "relationship"]
        lines = [eda.summary(), f"Observed types: {dict(eda.structural_profile.type_counts)}."]
        if quality:
            lines.append(f"Quality observations: {' '.join(quality[:3])}")
        if relationships:
            lines.append(f"Relationship observations: {' '.join(relationships[:3])}")
        lines.append(f"{len(semantic.candidates)} semantic candidate(s) require review; {len(semantic.unknowns)} attribute meaning(s) remain unknown.")
        if relevance:
            relevant = [item.attribute for item in relevance.candidates if item.category == "potentially_relevant"]
            lines.append(f"For objective '{relevance.request.objective}', potentially relevant attributes: {', '.join(relevant) or 'none established'}.")
        lines.append("This summary reports evidence and candidates; it does not establish domain meaning, causality, or permanent relevance.")
        return "\n".join(lines)