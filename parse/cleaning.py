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
from parse.cleaning_confidence import (
    score_duplicate_removal,
    score_imputation,
    score_outlier_flag,
)
from parse.analysis import AnalysisOrchestrator, AnalysisRequest
from parse.cleaning_impact import (
    compute_distribution_shifts,
    derive_information_loss_notes,
)


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
    limitations: list[str] = dataclass_field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intended_issues_addressed": self.intended_issues_addressed,
            "newly_introduced_issues": self.newly_introduced_issues,
            "before_snapshot": self.before_snapshot.to_dict(),
            "after_snapshot": self.after_snapshot.to_dict(),
            "comparison": self.comparison,
            "limitations": self.limitations,
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

    def detect(
        self,
        frame: pd.DataFrame,
        eda_result: Any | None = None,
        context: CleaningContext | None = None,
    ) -> tuple[list[CleaningIssue], list[TransformationProposal]]:
        self._validate_frame(frame)
        return self._detect(frame, eda_result=eda_result, context=context)

    def _detect(
        self,
        frame: pd.DataFrame,
        eda_result: Any | None = None,
        context: CleaningContext | None = None,
    ) -> tuple[list[CleaningIssue], list[TransformationProposal]]:
        issues: list[CleaningIssue] = []
        proposals: list[TransformationProposal] = []

        # 1. Resolve shared inputs
        resolved_eda = eda_result or (context.eda_result if context else None)
        resolved_purpose = context.purpose if context else None  # None means "not specified", distinct from "unknown"
        attribute_profiles = context.attribute_profiles if context else None
        source_id = context.source_ref.source_id if context else self.source.source_id

        # 2. Duplicate rows detection (exact rows across all columns)
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
            dupe_conf, dupe_basis = score_duplicate_removal(frame, duplicate_rows)
            proposals.append(TransformationProposal(
                "remove_duplicate_rows", issue_id, "remove_duplicates",
                method="keep_first", rationale="Remove exact duplicate observations while retaining the first row.",
                affected_records=duplicate_rows,
                confidence=dupe_conf,
                parameters={"confidence_basis": dupe_basis},
            ))

        # 3. Discrepancy 1: Candidate-identifier conflict detection
        # Enabled whenever structural_profile is reachable from resolved_eda
        if resolved_eda is not None and hasattr(resolved_eda, "structural_profile"):
            keys = resolved_eda.structural_profile.candidate_index_columns
            for key in keys:
                if key in frame.columns:
                    dupe_keys = frame.index[frame.duplicated(subset=[key], keep=False)]
                    if len(dupe_keys) > len(duplicate_rows):
                        issues.append(CleaningIssue(
                            f"conflicting_identifier_{key}", "duplicate_identifier", "warning",
                            f"Candidate identifier '{key}' has repeated values across non-exact duplicate rows.",
                            field=key, row_indices=tuple(dupe_keys), method="structural_duplicate",
                            duplicate_kind="conflicting_identifier",
                            evidence=(EvidenceRef(source_id, "derived_from", locator=key),),
                        ))

        # 4. Per-column inspection: missingness, mixed values, quality findings
        for column in frame.columns:
            series = frame[column]
            attr_prof = attribute_profiles.get(column) if attribute_profiles else None

            # --- Missing values ---
            missing_rows = tuple(series.index[series.isna()])
            if missing_rows:
                issue_id = f"missing_{column}"
                missing_count = len(missing_rows)
                method = "analysis_profile" if attr_prof else "isna"
                missingness_kind = (attr_prof.missingness_kind or "MISSING") if attr_prof else None
                evidence = (EvidenceRef(source_id, "derived_from", locator=str(column)),) if attr_prof else ()

                issues.append(CleaningIssue(
                    issue_id, "missing_values", "warning",
                    f"Column '{column}' has {missing_count} missing value(s).",
                    field=str(column), row_indices=missing_rows, method=method,
                    assumptions=("Missingness is not assumed to be random.", "Imputation requires user approval."),
                    missingness_kind=missingness_kind,
                    evidence=evidence,
                ))

                # Discrepancy 2: Imputation purpose gating
                # Propose whenever non-null values exist, unless resolved_purpose == "unknown"
                if resolved_purpose != "unknown":
                    non_null = series.dropna()
                    if len(non_null):
                        if attr_prof and attr_prof.observed_type:
                            is_numeric = attr_prof.observed_type == "numeric"
                        else:
                            is_numeric = pd.api.types.is_numeric_dtype(series)

                        imp_method = "median" if is_numeric else "mode"
                        val = float(non_null.median()) if is_numeric else non_null.mode().iloc[0]

                        imp_conf, imp_basis = score_imputation(series, frame=frame, column_name=str(column))
                        rationale = (
                            f"Fill missing '{column}' values with the observed {imp_method}. Purpose: {context.purpose}."
                            if context
                            else f"Fill missing '{column}' values with the observed {imp_method}; review whether this is scientifically appropriate."
                        )
                        proposals.append(TransformationProposal(
                            f"impute_{column}", issue_id, "impute_missing",
                            target=str(column), field=str(column), method=imp_method,
                            parameters={"value": val, "confidence_basis": imp_basis},
                            rationale=rationale,
                            affected_records=missing_rows,
                            confidence=imp_conf,
                        ))

            # --- Mixed values (Discrepancy 3: port heuristic and deduplicate) ---
            mixed_detected = False
            numeric_fraction = pd.to_numeric(series.dropna(), errors="coerce").notna().mean() if series.notna().any() else 0
            if series.dtype == object and 0 < numeric_fraction < 1:
                issues.append(CleaningIssue(
                    f"mixed_values_{column}", "mixed_values", "warning",
                    f"Column '{column}' mixes numeric-like and non-numeric values.",
                    field=str(column), method="numeric_parse_fraction",
                    assumptions=("Non-numeric values may be meaningful domain values.", "No conversion is proposed without domain review."),
                ))
                mixed_detected = True

            if attr_prof:
                for q_issue in attr_prof.quality_issues:
                    if q_issue.get("finding_id", "").startswith("mixed_"):
                        if not mixed_detected:
                            issues.append(CleaningIssue(
                                f"mixed_{column}", "mixed_values", "warning",
                                q_issue.get("observation", "Mixed types detected"),
                                field=str(column), method="analysis_finding",
                                evidence=(EvidenceRef(source_id, "derived_from", locator=str(column)),),
                            ))
                            mixed_detected = True

        # 5. Outliers (Discrepancy 4: iterate all matching findings per column from resolved_eda)
        if resolved_eda is not None and hasattr(resolved_eda, "findings"):
            for column in frame.columns:
                col_str = str(column)
                matching_findings = [
                    f for f in resolved_eda.findings
                    if (
                        getattr(f, "finding_id", "") in {f"outliers_{col_str}", f"unusual:{col_str}"}
                        or (
                            getattr(f, "subject", None) == col_str
                            and (
                                getattr(f, "category", "") in {"distribution", "anomaly"}
                                or getattr(f, "kind", "") == "outlier"
                                or "unusual" in getattr(f, "finding_id", "")
                                or "outlier" in getattr(f, "finding_id", "")
                            )
                        )
                    )
                ]

                for idx, finding in enumerate(matching_findings):
                    finding_id = getattr(finding, "finding_id", f"outliers_{col_str}")
                    issue_id = finding_id if idx == 0 else f"{finding_id}_{idx}"
                    msg = getattr(finding, "message", None) or getattr(
                        finding, "observation", f"Column '{col_str}' contains outlier(s)."
                    )
                    method = getattr(finding, "method", "IQR")
                    assumptions = tuple(getattr(finding, "assumptions", (
                        "An outlier is not automatically a data error.",
                        "No values are removed automatically.",
                    )))
                    limitations = tuple(getattr(finding, "limitations", ()))
                    evidence = tuple(getattr(finding, "evidence", ()))

                    issues.append(CleaningIssue(
                        issue_id, "statistical_outliers", "info",
                        msg, field=col_str, method=method,
                        assumptions=assumptions,
                        limitations=limitations,
                        evidence=evidence,
                    ))

                    outlier_conf, outlier_basis = score_outlier_flag(frame[column])
                    proposal_id = (
                        f"investigate_{col_str}_outliers" if idx == 0 else f"investigate_{col_str}_outliers_{idx}"
                    )
                    proposals.append(TransformationProposal(
                        proposal_id, issue_id, "flag_for_review",
                        target=col_str, field=col_str, method="flag",
                        parameters={"confidence_basis": outlier_basis},
                        rationale="Statistically unusual values should be investigated before exclusion.",
                        confidence=outlier_conf,
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
        
        # Re-detect issues on cleaned frame for ValidationResult
        newly_introduced: list[str] = []
        validation_limitations: list[str] = []

        if len(cleaned) == 0:
            validation_limitations.append("Re-detection for newly introduced issues skipped: cleaned dataset has 0 rows.")
        else:
            try:
                fresh_source = SourceRef(
                    self.source.source_id,
                    self.source.source_type,
                    label=f"{self.source.label} (post-cleaning)",
                )
                fresh_eda = AnalysisOrchestrator().analyze(AnalysisRequest(cleaned, fresh_source))
                cleaned_issues, _ = self.detect(cleaned, eda_result=fresh_eda)

                # Separately reconcile structural/candidate-identifier issues on cleaned
                cleaned_duplicate_rows = tuple(cleaned.index[cleaned.duplicated(keep="first")])
                candidate_keys = fresh_eda.structural_profile.candidate_index_columns
                for key in candidate_keys:
                    if key in cleaned.columns:
                        dupe_keys = cleaned.index[cleaned.duplicated(subset=[key], keep=False)]
                        if len(dupe_keys) > len(cleaned_duplicate_rows):
                            cleaned_issues.append(
                                CleaningIssue(
                                    f"conflicting_identifier_{key}",
                                    "duplicate_identifier",
                                    "warning",
                                    f"Candidate identifier '{key}' has repeated values across non-exact duplicate rows.",
                                    field=key,
                                    row_indices=tuple(dupe_keys),
                                    method="structural_duplicate",
                                    duplicate_kind="conflicting_identifier",
                                    evidence=(EvidenceRef(fresh_source.source_id, "derived_from", locator=key),),
                                )
                            )

                # Diff against original issues keyed by (kind, field)
                original_keys = {(issue.kind, issue.field) for issue in issues}
                seen_new_keys: set[tuple[str, str | None]] = set()
                for c_issue in cleaned_issues:
                    issue_key = (c_issue.kind, c_issue.field)
                    if issue_key not in original_keys and issue_key not in seen_new_keys:
                        seen_new_keys.add(issue_key)
                        field_display = f"'{c_issue.field}'" if c_issue.field else "dataset"
                        newly_introduced.append(
                            f"New issue after cleaning — {c_issue.kind} in {field_display}: {c_issue.message}"
                        )

                validation_limitations.append(
                    "Re-detection covers duplicates, missingness, mixed-type values, statistical outliers, "
                    "and candidate-identifier conflicts on the cleaned frame; it does not re-run semantic "
                    "candidate generation or human confirmation state, which are unaffected by transformation actions."
                )
            except Exception as exc:
                newly_introduced = []
                validation_limitations.append(
                    f"Re-detection for newly introduced issues skipped due to unexpected error: {exc}"
                )

        val_result = ValidationResult(
            intended_issues_addressed=[p.issue_id for p in proposals if p.status == "approved"],
            newly_introduced_issues=newly_introduced,
            before_snapshot=before_snap,
            after_snapshot=after_snap,
            comparison={
                "row_count_diff": after_snap.row_count - before_snap.row_count,
            },
            limitations=validation_limitations,
        )
        
        shifts = compute_distribution_shifts(original, cleaned)
        loss_notes = derive_information_loss_notes(shifts, changes)

        impact = DownstreamImpact(
            changed_attributes=[c.field for c in changes if c.field],
            sample_size_before=before_snap.row_count,
            sample_size_after=after_snap.row_count,
            distribution_shifts=shifts,
            potential_information_loss=loss_notes,
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