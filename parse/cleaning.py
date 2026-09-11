"""Controlled, traceable data-quality detection and cleaning.

The cleaner never changes the supplied frame. Transformations are proposed
first and only approved proposals are applied to a copy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field as dataclass_field
from typing import Any, Iterable

import pandas as pd

from parse.core.contracts import EvidenceRef, Provenance, SourceRef
from parse.eda import EDAResult as OldEDAResult
from parse.cleaning_context import CleaningContext


@dataclass(frozen=True)
class CleaningIssue:
    issue_id: str
    kind: str
    severity: str
    message: str
    field: str | None = None
    row_indices: tuple[Any, ...] = ()
    method: str | None = None
    assumptions: tuple[str, ...] = ()
    knowledge_state: str = "OBSERVED"
    evidence: tuple[EvidenceRef, ...] = ()
    uncertainty: str | None = None
    limitations: tuple[str, ...] = ()
    missingness_kind: str | None = None
    duplicate_kind: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["row_indices"] = list(self.row_indices)
        value["assumptions"] = list(self.assumptions)
        value["evidence"] = [e.to_dict() for e in self.evidence]
        value["limitations"] = list(self.limitations)
        return value


@dataclass
class TransformationProposal:
    proposal_id: str
    issue_id: str
    action: str
    target: str = "dataset"
    status: str = "proposed"
    field: str | None = None
    method: str | None = None
    parameters: dict[str, Any] = dataclass_field(default_factory=dict)
    rationale: str = ""
    affected_records: tuple[Any, ...] = ()
    evidence: tuple[EvidenceRef, ...] = ()
    assumptions: tuple[str, ...] = ()
    expected_effect: str = ""
    reversibility: str = "reversible"
    confidence: float | None = None
    knowledge_state: str = "INFERRED"
    provenance: Provenance | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["affected_records"] = list(self.affected_records)
        value["evidence"] = [e.to_dict() for e in self.evidence]
        value["assumptions"] = list(self.assumptions)
        if self.provenance:
            value["provenance"] = self.provenance.to_dict()
        return value


@dataclass(frozen=True)
class ChangeRecord:
    proposal_id: str
    issue_id: str
    row_index: Any
    field: str | None
    original_value: Any
    transformed_value: Any
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QualitySnapshot:
    row_count: int
    column_count: int
    missing_by_column: dict[str, int]
    duplicate_rows: int
    dtypes: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HumanDecision:
    decision_id: str
    proposal_id: str
    action: str
    modifications: dict[str, Any]
    rationale: str | None
    reviewer: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    intended_issues_addressed: list[str]
    newly_introduced_issues: list[str]
    before_snapshot: QualitySnapshot
    after_snapshot: QualitySnapshot
    comparison: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "intended_issues_addressed": self.intended_issues_addressed,
            "newly_introduced_issues": self.newly_introduced_issues,
            "before_snapshot": self.before_snapshot.to_dict(),
            "after_snapshot": self.after_snapshot.to_dict(),
            "comparison": self.comparison,
        }


@dataclass
class DownstreamImpact:
    changed_attributes: list[str]
    sample_size_before: int
    sample_size_after: int
    distribution_shifts: dict[str, Any]
    potential_information_loss: list[str]
    limitations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CleaningResult:
    source: SourceRef
    original: pd.DataFrame
    cleaned: pd.DataFrame
    issues: list[CleaningIssue]
    proposals: list[TransformationProposal]
    changes: list[ChangeRecord]
    before: QualitySnapshot
    after: QualitySnapshot
    unresolved_issue_ids: list[str]
    provenance: Provenance
    purpose: str = "unknown"
    context_ref: str | None = None
    human_decisions: list[HumanDecision] = dataclass_field(default_factory=list)
    validation: ValidationResult | None = None
    downstream_impact: DownstreamImpact | None = None
    analysis_evidence_refs: list[EvidenceRef] = dataclass_field(default_factory=list)

    @property
    def data_modified(self) -> bool:
        return bool(self.changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "issues": [issue.to_dict() for issue in self.issues],
            "proposals": [proposal.to_dict() for proposal in self.proposals],
            "changes": [change.to_dict() for change in self.changes],
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "unresolved_issue_ids": list(self.unresolved_issue_ids),
            "data_modified": self.data_modified,
            "provenance": self.provenance.to_dict(),
            "purpose": self.purpose,
            "context_ref": self.context_ref,
            "human_decisions": [d.to_dict() for d in self.human_decisions],
            "validation": self.validation.to_dict() if self.validation else None,
            "downstream_impact": self.downstream_impact.to_dict() if self.downstream_impact else None,
            "analysis_evidence_refs": [r.to_dict() for r in self.analysis_evidence_refs],
        }


class DataCleaner:
    """Detect quality concerns and apply explicitly approved proposals."""

    def __init__(self, source: SourceRef | None = None):
        self.source = source or SourceRef("in_memory_dataset", "user_input", label="DataFrame")

    def detect(self, frame: pd.DataFrame, eda_result: OldEDAResult | None = None, context: CleaningContext | None = None) -> tuple[list[CleaningIssue], list[TransformationProposal]]:
        self._validate_frame(frame)
        
        if context is not None:
            return self._detect_with_context(frame, context)
            
        # Backwards compatibility path
        return self._detect_legacy(frame, eda_result)

    def _detect_legacy(self, frame: pd.DataFrame, eda_result: OldEDAResult | None) -> tuple[list[CleaningIssue], list[TransformationProposal]]:
        issues: list[CleaningIssue] = []
        proposals: list[TransformationProposal] = []

        duplicate_rows = tuple(frame.index[frame.duplicated(keep="first")])
        if duplicate_rows:
            issue_id = "duplicate_rows"
            issues.append(CleaningIssue(
                issue_id, "duplicate_rows", "warning",
                f"{len(duplicate_rows)} duplicate row(s) can be reviewed.",
                row_indices=duplicate_rows, method="pandas.duplicated",
                assumptions=("Rows are duplicates across all columns.",),
            ))
            proposals.append(TransformationProposal(
                "remove_duplicate_rows", issue_id, "remove_duplicates",
                method="keep_first", rationale="Remove exact duplicate observations while retaining the first row.",
            ))

        for column in frame.columns:
            series = frame[column]
            missing_rows = tuple(series.index[series.isna()])
            if missing_rows:
                issue_id = f"missing_{column}"
                issues.append(CleaningIssue(
                    issue_id, "missing_values", "warning",
                    f"Column '{column}' has {len(missing_rows)} missing value(s).",
                    field=str(column), row_indices=missing_rows, method="isna",
                    assumptions=("Missingness is not assumed to be random.", "Imputation requires user approval."),
                ))
                non_null = series.dropna()
                if len(non_null):
                    numeric = pd.api.types.is_numeric_dtype(series)
                    method = "median" if numeric else "mode"
                    value = float(non_null.median()) if numeric else non_null.mode().iloc[0]
                    proposals.append(TransformationProposal(
                        f"impute_{column}", issue_id, "impute_missing",
                        field=str(column), method=method, parameters={"value": value},
                        rationale=f"Fill missing '{column}' values with the observed {method}; review whether this is scientifically appropriate.",
                    ))

            numeric_fraction = pd.to_numeric(series.dropna(), errors="coerce").notna().mean() if series.notna().any() else 0
            if series.dtype == object and 0 < numeric_fraction < 1:
                issues.append(CleaningIssue(
                    f"mixed_values_{column}", "mixed_values", "warning",
                    f"Column '{column}' mixes numeric-like and non-numeric values.",
                    field=str(column), method="numeric_parse_fraction",
                    assumptions=("Non-numeric values may be meaningful domain values.", "No conversion is proposed without domain review."),
                ))

            if eda_result is not None:
                outlier = next((finding for finding in eda_result.findings if finding.finding_id == f"outliers_{column}"), None)
                if outlier is not None:
                    issues.append(CleaningIssue(
                        f"outliers_{column}", "statistical_outliers", "info",
                        outlier.message, field=str(column), method="IQR",
                        assumptions=("An outlier is not automatically a data error.", "No values are removed automatically."),
                    ))

        return issues, proposals

    def _detect_with_context(self, frame: pd.DataFrame, context: CleaningContext) -> tuple[list[CleaningIssue], list[TransformationProposal]]:
        issues: list[CleaningIssue] = []
        proposals: list[TransformationProposal] = []
        
        source_id = context.source_ref.source_id
        
        # 1. Duplicates (from structure analysis)
        # Note: Analysis doesn't explicitly return duplicate row indices, so we still check it,
        # but we use structural knowledge for identifiers.
        duplicate_rows = tuple(frame.index[frame.duplicated(keep="first")])
        if duplicate_rows:
            issue_id = "duplicate_rows"
            issues.append(CleaningIssue(
                issue_id, "duplicate_rows", "warning",
                f"{len(duplicate_rows)} duplicate row(s) can be reviewed.",
                row_indices=duplicate_rows, method="pandas.duplicated",
                assumptions=("Rows are duplicates across all columns.",),
                duplicate_kind="exact",
            ))
            proposals.append(TransformationProposal(
                "remove_duplicate_rows", issue_id, "remove_duplicates",
                method="keep_first", rationale="Remove exact duplicate observations while retaining the first row.",
                affected_records=duplicate_rows,
            ))
            
        # Detect conflicting identifiers based on candidate keys from structure
        keys = context.eda_result.structural_profile.candidate_index_columns
        for key in keys:
            if key in frame.columns:
                dupe_keys = frame.index[frame.duplicated(subset=[key], keep=False)]
                if len(dupe_keys) > len(duplicate_rows):
                    issues.append(CleaningIssue(
                        f"conflicting_identifier_{key}", "duplicate_identifier", "warning",
                        f"Candidate identifier '{key}' has repeated values across non-exact duplicate rows.",
                        field=key, row_indices=tuple(dupe_keys), method="structural_duplicate",
                        duplicate_kind="conflicting_identifier",
                        evidence=(EvidenceRef(source_id, "derived_from", locator=key),)
                    ))

        # 2. Consume Attribute Profiles
        for attr_name, profile in context.attribute_profiles.items():
            if attr_name not in frame.columns:
                continue
                
            # Missingness
            if profile.missing_count > 0:
                missing_rows = tuple(frame.index[frame[attr_name].isna()])
                issue_id = f"missing_{attr_name}"
                issues.append(CleaningIssue(
                    issue_id, "missing_values", "warning",
                    f"Column '{attr_name}' has {profile.missing_count} missing value(s).",
                    field=attr_name, row_indices=missing_rows, method="analysis_profile",
                    assumptions=("Missingness is not assumed to be random.", "Imputation requires user approval."),
                    missingness_kind=profile.missingness_kind or "MISSING",
                    evidence=(EvidenceRef(source_id, "derived_from", locator=attr_name),)
                ))
                
                # Propose based on purpose
                if context.purpose != "unknown":
                     non_null = frame[attr_name].dropna()
                     if len(non_null):
                         if profile.observed_type == "numeric":
                             val = float(non_null.median())
                             method = "median"
                         else:
                             val = non_null.mode().iloc[0]
                             method = "mode"
                             
                         proposals.append(TransformationProposal(
                            f"impute_{attr_name}", issue_id, "impute_missing",
                            target=attr_name, field=attr_name, method=method, parameters={"value": val},
                            rationale=f"Fill missing '{attr_name}' values with the observed {method}. Purpose: {context.purpose}.",
                            affected_records=missing_rows,
                        ))

            # Quality Issues (mixed types, empty)
            for q_issue in profile.quality_issues:
                if q_issue.get("finding_id", "").startswith("mixed_"):
                    # Actually, parsing analysis doesn't generate "mixed_" natively yet, but we prepare for it
                    issues.append(CleaningIssue(
                        f"mixed_{attr_name}", "mixed_values", "warning",
                        q_issue.get("observation", "Mixed types detected"),
                        field=attr_name, method="analysis_finding",
                        evidence=(EvidenceRef(source_id, "derived_from", locator=attr_name),)
                    ))
                    
            # Anomalies (Outliers)
            for anomaly in profile.anomalies:
                 issue_id = anomaly.get("finding_id", f"outliers_{attr_name}")
                 issues.append(CleaningIssue(
                    issue_id, "statistical_outliers", "info",
                    anomaly.get("observation", "Outliers detected"),
                    field=attr_name, method=anomaly.get("method", "IQR"),
                    assumptions=tuple(anomaly.get("assumptions", [])),
                    limitations=tuple(anomaly.get("limitations", [])),
                    evidence=(EvidenceRef(source_id, "derived_from", locator=attr_name),)
                 ))
                 
                 proposals.append(TransformationProposal(
                    f"investigate_{attr_name}_outliers", issue_id, "flag_for_review",
                    target=attr_name, field=attr_name, method="flag",
                    rationale="Statistically unusual values should be investigated before exclusion.",
                 ))

        return issues, proposals

    def clean(
        self, 
        frame: pd.DataFrame, 
        approved: Iterable[str] = (), 
        eda_result: OldEDAResult | None = None,
        decisions: list[HumanDecision] | None = None,
        context: CleaningContext | None = None,
    ) -> CleaningResult:
        self._validate_frame(frame)
        original = frame.copy(deep=True)
        
        issues, proposals = self.detect(original, context=context, eda_result=eda_result)
        
        approved_ids = set(approved)
        if decisions:
            approved_ids.update(d.proposal_id for d in decisions if d.action == "APPROVE")
            
        changes: list[ChangeRecord] = []
        cleaned = original.copy(deep=True)

        for proposal in proposals:
            if proposal.proposal_id not in approved_ids:
                continue
            proposal.status = "approved"
            if proposal.action == "remove_duplicates":
                duplicate_indices = original.index[original.duplicated(keep="first")]
                for row_index in duplicate_indices:
                    changes.append(ChangeRecord(proposal.proposal_id, proposal.issue_id, row_index, None, original.loc[row_index].to_dict(), None, proposal.rationale))
                cleaned = cleaned.drop_duplicates(keep="first")
            elif proposal.action == "impute_missing":
                column = proposal.field
                if column and column in cleaned:
                    value = proposal.parameters["value"]
                    for row_index in original.index[original[column].isna()]:
                        changes.append(ChangeRecord(proposal.proposal_id, proposal.issue_id, row_index, column, None, value, proposal.rationale))
                    cleaned[column] = cleaned[column].fillna(value)

        for proposal in proposals:
            if proposal.proposal_id not in approved_ids:
                # If a decision explicitly rejected it
                if decisions and any(d.proposal_id == proposal.proposal_id and d.action == "REJECT" for d in decisions):
                    proposal.status = "rejected"
                elif decisions and any(d.proposal_id == proposal.proposal_id and d.action == "DEFER" for d in decisions):
                    proposal.status = "deferred"
                else:
                    proposal.status = "proposed"
                    
        unresolved = [
            issue.issue_id for issue in issues 
            if not any(proposal.issue_id == issue.issue_id and proposal.status == "approved" for proposal in proposals)
        ]
        
        before_snap = self._snapshot(original)
        after_snap = self._snapshot(cleaned)
        
        val_result = ValidationResult(
            intended_issues_addressed=[p.issue_id for p in proposals if p.status == "approved"],
            newly_introduced_issues=[],
            before_snapshot=before_snap,
            after_snapshot=after_snap,
            comparison={
                "row_count_diff": after_snap.row_count - before_snap.row_count,
            }
        )
        
        impact = DownstreamImpact(
            changed_attributes=[c.field for c in changes if c.field],
            sample_size_before=before_snap.row_count,
            sample_size_after=after_snap.row_count,
            distribution_shifts={},
            potential_information_loss=[],
            limitations=["Downstream impact is approximated."]
        )
        
        prov_method = "controlled_proposals_v2" if context else "controlled_proposals_v1"
        
        return CleaningResult(
            source=self.source, original=original, cleaned=cleaned,
            issues=issues, proposals=proposals, changes=changes,
            before=before_snap, after=after_snap,
            unresolved_issue_ids=unresolved,
            provenance=Provenance([self.source], "data_quality_cleaning", prov_method),
            purpose=context.purpose if context else "unknown",
            context_ref="CleaningContext" if context else None,
            human_decisions=decisions or [],
            validation=val_result,
            downstream_impact=impact,
            analysis_evidence_refs=[EvidenceRef(context.source_ref.source_id, "derived_from")] if context else []
        )

    @staticmethod
    def _snapshot(frame: pd.DataFrame) -> QualitySnapshot:
        return QualitySnapshot(int(len(frame)), int(len(frame.columns)), {str(column): int(frame[column].isna().sum()) for column in frame.columns}, int(frame.duplicated().sum()), {str(column): str(dtype) for column, dtype in frame.dtypes.items()})

    @staticmethod
    def _validate_frame(frame: pd.DataFrame) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("DataCleaner expects a pandas DataFrame")