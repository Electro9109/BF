"""Standardized Logical Contract for PARSE Module Results and the Results Translator.

This contract defines the structured interface between analytical, predictive,
anomaly-detection, and recommendation modules and the PARSE Results Translator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from parse.core.contracts import (
    AnalysisResult,
    Applicability,
    EvidenceRef,
    Issue,
    Provenance,
    SourceRef,
)

TRANSLATOR_CONTRACT_VERSION = "1.0"

SUPPORTED_MODULE_TYPES = {
    "descriptive_analysis",
    "regression_prediction",
    "classification_prediction",
    "anomaly_detection",
    "quality_check",
    "recommendation_engine",
}

SUPPORTED_DEPARTMENTS = {
    "blast_furnace",
    "sintering",
    "energy",
    "steel_melting_shop",
    "coke_oven",
    "general_plant",
}


@dataclass
class StructuredModuleResult:
    """Standardized result envelope passed to the PARSE Results Translator."""

    schema_version: str = TRANSLATOR_CONTRACT_VERSION
    module_type: str = "descriptive_analysis"
    department: str = "general_plant"
    operation: str = "analysis"
    dataset_id: str = "unspecified"
    summary: str = ""
    target: str | None = None
    units: str | None = None
    statistics: dict[str, Any] = field(default_factory=dict)
    findings: list[dict[str, Any]] = field(default_factory=list)
    predictions: dict[str, Any] = field(default_factory=dict)
    evaluation_metrics: dict[str, Any] = field(default_factory=dict)
    anomalies: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    confidence: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.module_type not in SUPPORTED_MODULE_TYPES:
            raise ValueError(f"Unsupported module_type: {self.module_type}. Must be one of {SUPPORTED_MODULE_TYPES}")
        if self.department not in SUPPORTED_DEPARTMENTS:
            raise ValueError(f"Unsupported department: {self.department}. Must be one of {SUPPORTED_DEPARTMENTS}")
        self.statistics = dict(self.statistics)
        self.findings = list(self.findings)
        self.predictions = dict(self.predictions)
        self.evaluation_metrics = dict(self.evaluation_metrics)
        self.anomalies = list(self.anomalies)
        self.recommendations = list(self.recommendations)
        self.validation_warnings = list(self.validation_warnings)
        self.limitations = list(self.limitations)
        self.evidence = list(self.evidence)
        self.provenance = dict(self.provenance)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StructuredModuleResult":
        return cls(**dict(data))

    def to_core_analysis_result(self, result_id: str | None = None) -> AnalysisResult:
        """Convert into canonical PARSE AnalysisResult for pipeline compatibility."""
        rid = result_id or f"{self.department}:{self.module_type}:{self.operation}"
        source = SourceRef(self.dataset_id, "csv" if ".csv" in self.dataset_id else "user_input")
        in_refs = [EvidenceRef(source.source_id, "observed_from")]
        
        issues = [
            Issue("validation_warning", "warning", warn, source=source)
            for warn in self.validation_warnings
        ]
        
        outputs = {
            "schema_version": self.schema_version,
            "module_type": self.module_type,
            "department": self.department,
            "operation": self.operation,
            "dataset_id": self.dataset_id,
            "summary": self.summary,
            "target": self.target,
            "units": self.units,
            "statistics": self.statistics,
            "findings": self.findings,
            "predictions": self.predictions,
            "evaluation_metrics": self.evaluation_metrics,
            "anomalies": self.anomalies,
            "recommendations": self.recommendations,
            "confidence": self.confidence,
        }
        
        applicability = Applicability(
            indicators={"confidence": self.confidence} if self.confidence else {},
            limitations=self.limitations,
        )
        
        prov = Provenance(
            sources=[source],
            operation=self.operation,
            method=self.module_type,
            parent_ids=[self.dataset_id],
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        
        return AnalysisResult(
            result_id=rid,
            outputs=outputs,
            input_refs=in_refs,
            method=self.module_type,
            provenance=prov,
            applicability=applicability,
            issues=issues,
        )
