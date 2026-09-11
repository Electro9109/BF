"""Cleaning context contract.

Defines the explicit boundary between Dataset Analysis and Data Quality/Cleaning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from parse.analysis import EDAResult as NewEDAResult
from parse.core.contracts import Provenance, SourceRef
from parse.semantic_analysis import AnalysisBundle, HumanContext

CleaningPurpose = Literal[
    "descriptive_analysis",
    "statistical_modelling",
    "prediction",
    "retrieval",
    "reporting",
    "integration",
    "unknown",
]

KnowledgeState = Literal[
    "OBSERVED",
    "INFERRED",
    "USER_CONFIRMED",
    "DOMAIN_DEFINED",
    "UNKNOWN",
    "CONFLICTING",
]

MissingnessKind = Literal[
    "MISSING",
    "BLANK",
    "EXPLICIT_UNKNOWN",
    "INVALID_PLACEHOLDER",
    "NOT_APPLICABLE",
    "STRUCTURALLY_ABSENT",
]


@dataclass(frozen=True)
class AttributeCleaningProfile:
    """Attribute-level profile synthesized from Analysis for Cleaning."""
    name: str
    observed_type: str
    semantic_candidates: tuple[dict[str, Any], ...] = ()
    semantic_status: KnowledgeState = "UNKNOWN"
    missing_count: int = 0
    missing_rate: float = 0.0
    missingness_kind: MissingnessKind | None = None
    unique_count: int = 0
    cardinality_rate: float = 0.0
    distribution_summary: dict[str, Any] = field(default_factory=dict)
    anomalies: tuple[dict[str, Any], ...] = ()
    relationships: tuple[dict[str, Any], ...] = ()
    representation_issues: tuple[dict[str, Any], ...] = ()
    quality_issues: tuple[dict[str, Any], ...] = ()
    provenance: Provenance | None = None
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.provenance:
            result["provenance"] = self.provenance.to_dict()
        return result


@dataclass
class CleaningContext:
    """The explicit contract passed from Analysis to Cleaning."""
    source_ref: SourceRef
    eda_result: NewEDAResult
    semantic_result: Any  # parse.semantic_analysis.SemanticResult
    human_context: HumanContext | None
    attribute_profiles: dict[str, AttributeCleaningProfile]
    purpose: CleaningPurpose = "unknown"
    assumptions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source_ref.to_dict(),
            "purpose": self.purpose,
            "attribute_profiles": {
                k: v.to_dict() for k, v in self.attribute_profiles.items()
            },
            "assumptions": list(self.assumptions),
            "limitations": list(self.limitations),
        }


def create_cleaning_context(
    bundle: AnalysisBundle,
    purpose: CleaningPurpose = "unknown",
    human_context: HumanContext | None = None,
) -> CleaningContext:
    """Factory to synthesize CleaningContext from an AnalysisBundle."""
    
    profiles: dict[str, AttributeCleaningProfile] = {}
    
    # 1. Map EDA attributes
    for attr in bundle.eda.attributes:
        dist_summary = {}
        if attr.distribution:
            dist_summary = {
                "min": attr.distribution.minimum,
                "max": attr.distribution.maximum,
                "mean": attr.distribution.mean,
                "median": attr.distribution.median,
                "outlier_count": attr.distribution.outlier_count,
            }
            
        profiles[attr.name] = AttributeCleaningProfile(
            name=attr.name,
            observed_type=attr.observed_type,
            missing_count=attr.missing_count,
            missing_rate=attr.missing_rate,
            missingness_kind="MISSING" if attr.missing_count > 0 else None, # Simplified for now, can be enriched
            unique_count=attr.unique_count,
            cardinality_rate=attr.cardinality_rate,
            distribution_summary=dist_summary,
            provenance=attr.provenance,
        )

    # 2. Enrich with quality/anomaly findings
    for finding in bundle.eda.findings:
        if isinstance(finding.subject, str) and finding.subject in profiles:
            attr_prof = profiles[finding.subject]
            
            # Convert finding to dict for storage in tuple
            f_dict = finding.to_dict()
            
            if finding.category == "quality":
                # Need to bypass frozen status for the factory
                object.__setattr__(attr_prof, "quality_issues", attr_prof.quality_issues + (f_dict,))
                
            elif finding.category == "distribution" and "unusual" in finding.finding_id:
                object.__setattr__(attr_prof, "anomalies", attr_prof.anomalies + (f_dict,))
                
        elif isinstance(finding.subject, tuple) and finding.category == "relationship":
             for subj in finding.subject:
                 if subj in profiles:
                     attr_prof = profiles[subj]
                     object.__setattr__(attr_prof, "relationships", attr_prof.relationships + (finding.to_dict(),))

    # 3. Enrich with semantic candidates
    for candidate in bundle.semantic.candidates:
        if candidate.attribute in profiles:
            attr_prof = profiles[candidate.attribute]
            object.__setattr__(attr_prof, "semantic_candidates", attr_prof.semantic_candidates + (candidate.to_dict(),))
            
            # Determine semantic status based on human context
            status: KnowledgeState = "INFERRED"
            if human_context:
                hs = human_context.state_for(candidate.candidate_id)
                if hs == "confirmed":
                    status = "USER_CONFIRMED"
                elif hs == "conflicting":
                    status = "CONFLICTING"
                elif hs == "unknown":
                    status = "UNKNOWN"
            object.__setattr__(attr_prof, "semantic_status", status)


    return CleaningContext(
        source_ref=bundle.eda.request.source,
        eda_result=bundle.eda,
        semantic_result=bundle.semantic,
        human_context=human_context,
        attribute_profiles=profiles,
        purpose=purpose,
    )
