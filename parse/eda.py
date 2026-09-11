"""Dependency-light automated understanding for structured tabular data.

This capability profiles a DataFrame without modifying it. Findings are
observations or cautious interpretations, each tied to the supplied source.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd

from parse.core.contracts import EvidenceRef, Issue, Provenance, SourceRef


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    inferred_type: str
    row_count: int
    non_null_count: int
    missing_count: int
    missing_fraction: float
    unique_count: int
    type_confidence: float = 1.0
    candidate_identifier: bool = False
    candidate_group: bool = False
    candidate_target: bool = False
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EDAFinding:
    finding_id: str
    category: str
    kind: str
    message: str
    evidence: tuple[EvidenceRef, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["evidence"] = [item.to_dict() for item in self.evidence]
        return value


@dataclass(frozen=True)
class NextAction:
    action: str
    applicable: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EDAResult:
    source: SourceRef
    row_count: int
    column_count: int
    columns: list[ColumnProfile]
    findings: list[EDAFinding]
    issues: list[Issue]
    next_actions: list[NextAction]
    limitations: list[str]
    provenance: Provenance
    data_modified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": [column.to_dict() for column in self.columns],
            "findings": [finding.to_dict() for finding in self.findings],
            "issues": [issue.to_dict() for issue in self.issues],
            "next_actions": [action.to_dict() for action in self.next_actions],
            "limitations": list(self.limitations),
            "provenance": self.provenance.to_dict(),
            "data_modified": self.data_modified,
        }

    def summary(self) -> str:
        """Return a concise user-facing summary grounded in the profile."""
        quality_count = sum(1 for finding in self.findings if finding.category == "quality")
        interpretations = [finding.message for finding in self.findings if finding.kind == "interpretation"]
        lines = [
            f"Dataset contains {self.row_count} rows and {self.column_count} columns.",
            f"Detected {quality_count} data-quality finding(s).",
        ]
        if interpretations:
            lines.append("Key interpretations:")
            lines.extend(f"- {message}" for message in interpretations[:5])
        applicable = [action.action for action in self.next_actions if action.applicable]
        lines.append(f"Applicable next actions: {', '.join(applicable) or 'none identified'}.")
        return "\n".join(lines)


class DataUnderstanding:
    """Profile structured tabular data without changing the input."""

    def __init__(self, source: SourceRef | None = None):
        self.source = source or SourceRef("in_memory_dataset", "user_input", label="DataFrame")

    @classmethod
    def from_file(cls, path: str | Path, sheet_name: str | int = 0) -> EDAResult:
        """Load a supported structured file and return its profile.

        Only CSV and Excel are supported in this first capability slice.
        Loading is read-only; profiling never writes or transforms the source.
        """
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            frame = pd.read_csv(file_path)
            source_type = "csv"
        elif suffix in {".xlsx", ".xls"}:
            frame = pd.read_excel(file_path, sheet_name=sheet_name)
            source_type = "spreadsheet"
        else:
            raise ValueError(f"Unsupported dataset format: {suffix or '<none>'}")
        source = SourceRef(
            source_id=f"dataset:{file_path.resolve()}",
            source_type=source_type,
            locator=str(file_path.resolve()),
            label=file_path.name,
        )
        return cls(source).profile(frame)

    def profile(self, frame: pd.DataFrame) -> EDAResult:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("DataUnderstanding.profile expects a pandas DataFrame")

        row_count, column_count = frame.shape
        columns = [self._profile_column(frame, name) for name in frame.columns]
        findings: list[EDAFinding] = []
        issues: list[Issue] = []

        if row_count == 0:
            issues.append(Issue("empty_dataset", "error", "The dataset has no rows", source=self.source))
        if column_count == 0:
            issues.append(Issue("empty_schema", "error", "The dataset has no columns", source=self.source))

        duplicate_count = int(frame.duplicated().sum())
        if duplicate_count:
            findings.append(self._finding(
                "duplicate_rows", "quality", "observation",
                f"{duplicate_count} duplicate row(s) were detected.",
                {"count": duplicate_count},
            ))
            issues.append(Issue(
                "duplicate_rows", "warning",
                f"{duplicate_count} duplicate row(s) were detected.",
                source=self.source,
            ))

        for column in columns:
            if column.missing_count:
                findings.append(self._finding(
                    f"missing_{column.name}", "quality", "observation",
                    f"Column '{column.name}' has {column.missing_count} missing value(s).",
                    {"column": column.name, "missing_count": column.missing_count,
                     "missing_fraction": column.missing_fraction},
                    locator=column.name,
                ))
            numeric_parse_fraction = column.summary.get("numeric_parse_fraction")
            if numeric_parse_fraction is not None and 0 < numeric_parse_fraction < 1:
                findings.append(self._finding(
                    f"mixed_values_{column.name}", "quality", "observation",
                    f"Column '{column.name}' mixes numeric-like and non-numeric values ({numeric_parse_fraction:.0%} parse as numbers).",
                    {"column": column.name, "numeric_parse_fraction": numeric_parse_fraction},
                    locator=column.name,
                    limitations=("The non-numeric values require domain review before conversion.",),
                ))
            if column.candidate_identifier:
                findings.append(self._finding(
                    f"identifier_{column.name}", "logical", "interpretation",
                    f"Column '{column.name}' may identify records rather than measure a phenomenon.",
                    {"column": column.name}, locator=column.name,
                    limitations=("This is a heuristic based on name and uniqueness; confirm its meaning.",),
                ))
            if column.candidate_target:
                findings.append(self._finding(
                    f"target_{column.name}", "logical", "interpretation",
                    f"Column '{column.name}' may be a target or outcome variable.",
                    {"column": column.name}, locator=column.name,
                    limitations=("Target candidacy is inferred from name/type and is not a causal claim.",),
                ))
            if column.candidate_group:
                findings.append(self._finding(
                    f"group_{column.name}", "logical", "interpretation",
                    f"Column '{column.name}' may define groups for stratified analysis.",
                    {"column": column.name}, locator=column.name,
                    limitations=("Grouping candidacy is inferred from low cardinality; confirm its meaning.",),
                ))

            if column.inferred_type == "numeric" and column.summary.get("outlier_count", 0):
                findings.append(self._finding(
                    f"outliers_{column.name}", "statistical", "observation",
                    f"Column '{column.name}' contains {column.summary['outlier_count']} IQR-based statistical outlier(s).",
                    {"column": column.name, "count": column.summary["outlier_count"]},
                    locator=column.name,
                    limitations=("A statistical outlier is not automatically a data error.",),
                ))
            if column.inferred_type == "numeric" and column.summary.get("skewness") is not None and abs(column.summary["skewness"]) >= 1:
                findings.append(self._finding(
                    f"skew_{column.name}", "statistical", "observation",
                    f"Column '{column.name}' is strongly skewed (skewness {column.summary['skewness']:.2f}).",
                    {"column": column.name, "skewness": column.summary["skewness"]},
                    locator=column.name,
                ))
            if column.missing_fraction >= 0.5:
                findings.append(self._finding(
                    f"sparse_{column.name}", "quality", "observation",
                    f"Column '{column.name}' is sparse with {column.missing_fraction:.0%} missing values.",
                    {"column": column.name, "missing_fraction": column.missing_fraction},
                    locator=column.name,
                ))

        findings.extend(self._temporal_findings(frame, columns))
        findings.extend(self._relationship_findings(frame, columns))
        next_actions = self._next_actions(frame, columns, findings)
        limitations = [
            "Type, identifier, target, and action recommendations are heuristic interpretations.",
            "Statistical relationships and outliers do not establish causation or data error.",
            "No values, rows, or columns were modified by profiling.",
        ]
        return EDAResult(
            source=self.source,
            row_count=row_count,
            column_count=column_count,
            columns=columns,
            findings=findings,
            issues=issues,
            next_actions=next_actions,
            limitations=limitations,
            provenance=Provenance(
                sources=[self.source],
                operation="structured_data_understanding",
                method="pandas_profile_v1",
            ),
            data_modified=False,
        )

    def _profile_column(self, frame: pd.DataFrame, name: Any) -> ColumnProfile:
        series = frame[name]
        missing_count = int(series.isna().sum())
        non_null = series.dropna()
        inferred_type, type_confidence = self._infer_type(series)
        summary: dict[str, Any] = {}
        if inferred_type == "numeric" and len(non_null):
            numeric = pd.to_numeric(non_null, errors="coerce").dropna()
            summary = {
                "min": float(numeric.min()),
                "max": float(numeric.max()),
                "mean": float(numeric.mean()),
                "median": float(numeric.median()),
                "std": float(numeric.std(ddof=0)),
                "q1": float(numeric.quantile(0.25)),
                "q3": float(numeric.quantile(0.75)),
                "skewness": float(numeric.skew()) if len(numeric) >= 3 else None,
            }
            q1, q3 = numeric.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0:
                outlier_count = 0
            else:
                outlier_count = int(((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)).sum())
            summary["outlier_count"] = outlier_count
        elif inferred_type in {"categorical", "text"}:
            counts = non_null.astype(str).value_counts().head(5)
            summary = {"top_values": {str(key): int(value) for key, value in counts.items()}}
        if series.dtype == object and len(non_null):
            numeric_fraction = float(pd.to_numeric(non_null, errors="coerce").notna().mean())
            if 0 < numeric_fraction < 1:
                summary["numeric_parse_fraction"] = numeric_fraction

        unique_count = int(non_null.nunique(dropna=True))
        name_text = str(name).strip().lower()
        candidate_identifier = bool(
            len(non_null) > 0
            and unique_count == len(non_null)
            and ("id" in name_text or name_text in {"key", "index"} or inferred_type == "numeric")
        )
        candidate_group = bool(
            inferred_type == "categorical"
            and 1 < unique_count <= min(20, max(2, int(len(non_null) * 0.2)))
        )
        candidate_target = bool(
            inferred_type == "numeric"
            and any(token in name_text for token in ("target", "label", "outcome", "result", "score", "temperature"))
        )
        return ColumnProfile(
            name=str(name),
            inferred_type=inferred_type,
            type_confidence=type_confidence,
            row_count=len(series),
            non_null_count=int(series.notna().sum()),
            missing_count=missing_count,
            missing_fraction=missing_count / len(series) if len(series) else 0.0,
            unique_count=unique_count,
            candidate_identifier=candidate_identifier,
            candidate_group=candidate_group,
            candidate_target=candidate_target,
            summary=summary,
        )

    @staticmethod
    def _infer_type(series: pd.Series) -> tuple[str, float]:
        if pd.api.types.is_numeric_dtype(series):
            return "numeric", 1.0
        if pd.api.types.is_datetime64_any_dtype(series):
            return "temporal", 1.0
        if series.dtype == object:
            non_null = series.dropna()
            if len(non_null):
                parsed_dates = pd.to_datetime(non_null, errors="coerce", format="mixed")
                if parsed_dates.notna().mean() >= 0.9:
                    return "temporal", float(parsed_dates.notna().mean())
                parsed_numbers = pd.to_numeric(non_null, errors="coerce")
                if parsed_numbers.notna().mean() >= 0.9:
                    return "numeric", float(parsed_numbers.notna().mean())
                cardinality = non_null.nunique() / len(non_null)
                if cardinality <= 0.2 or non_null.astype(str).str.len().mean() <= 24:
                    return "categorical", 0.8
            return "text", 0.7
        return "other", 0.5

    def _relationship_findings(self, frame: pd.DataFrame, columns: list[ColumnProfile]) -> list[EDAFinding]:
        numeric = frame.select_dtypes(include=[np.number])
        if numeric.shape[1] < 1:
            return []
        correlations = numeric.corr(numeric_only=True)
        findings = []
        if numeric.shape[1] >= 2:
            for left_index, left in enumerate(correlations.columns):
                for right in correlations.columns[left_index + 1:]:
                    value = correlations.loc[left, right]
                    if pd.notna(value) and abs(float(value)) >= 0.9:
                        findings.append(self._finding(
                            f"relationship_{left}_{right}", "relationship", "observation",
                            f"'{left}' and '{right}' have a strong observed correlation ({value:.2f}).",
                            {"left": left, "right": right, "correlation": float(value)},
                            locator=f"{left},{right}",
                            limitations=("Correlation does not establish causation.",),
                        ))
        categorical = [column.name for column in columns if column.inferred_type == "categorical"]
        for category in categorical:
            for numeric_name in numeric.columns:
                grouped = frame.groupby(category, dropna=False)[numeric_name].mean().dropna()
                if len(grouped) >= 2 and grouped.max() != grouped.min():
                    spread = float(grouped.max() - grouped.min())
                    findings.append(self._finding(
                        f"group_difference_{category}_{numeric_name}",
                        "relationship", "observation",
                        f"Mean '{numeric_name}' differs across '{category}' groups (spread {spread:.2f}).",
                        {"category": category, "numeric": numeric_name, "mean_spread": spread},
                        locator=f"{category},{numeric_name}",
                        limitations=("Group differences are descriptive and do not establish causation.",),
                    ))
        return findings

    def _temporal_findings(self, frame: pd.DataFrame, columns: list[ColumnProfile]) -> list[EDAFinding]:
        findings = []
        for column in columns:
            if column.inferred_type != "temporal":
                continue
            parsed = pd.to_datetime(frame[column.name], errors="coerce", format="mixed").dropna()
            if len(parsed) >= 2:
                ordered = bool(parsed.is_monotonic_increasing or parsed.is_monotonic_decreasing)
                findings.append(self._finding(
                    f"temporal_{column.name}", "logical", "interpretation",
                    f"Column '{column.name}' appears temporal and {'is ordered' if ordered else 'is not ordered'} in the input.",
                    {"column": column.name, "ordered": ordered, "min": str(parsed.min()), "max": str(parsed.max())},
                    locator=column.name,
                    limitations=("Temporal meaning is inferred from parseable values; confirm the time semantics.",),
                ))
        return findings

    @staticmethod
    def _next_actions(frame: pd.DataFrame, columns: list[ColumnProfile], findings: list[EDAFinding]) -> list[NextAction]:
        numeric_count = sum(column.inferred_type == "numeric" for column in columns)
        text_count = sum(column.inferred_type in {"text", "categorical"} for column in columns)
        quality_issues = any(finding.category == "quality" for finding in findings)
        target_available = any(column.candidate_target for column in columns)
        return [
            NextAction("clean_prepare", quality_issues, "Quality findings exist." if quality_issues else "No immediate quality issue was detected."),
            NextAction("analyse", numeric_count >= 1, "At least one numeric variable is available." if numeric_count else "No numeric variable was detected."),
            NextAction("explore_retrieve", text_count >= 1, "Categorical or text information is available." if text_count else "No text-like variable was detected."),
            NextAction("build_prediction", target_available and numeric_count >= 2, "A candidate numeric target and predictors exist." if target_available and numeric_count >= 2 else "No sufficiently supported numeric target/predictor set was detected."),
            NextAction("generate_report", True, "A structured profile is available for reporting."),
        ]

    def _finding(self, finding_id, category, kind, message, attributes, locator=None, limitations=()):
        return EDAFinding(
            finding_id=finding_id,
            category=category,
            kind=kind,
            message=message,
            evidence=(EvidenceRef(self.source.source_id, "observed_from", locator=locator),),
            attributes=attributes,
            limitations=tuple(limitations),
        )
