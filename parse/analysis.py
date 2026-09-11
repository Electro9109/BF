"""Domain-agnostic dataset analysis for PARSE.

This module observes tabular data without cleaning, transforming, imputing, or
assigning domain meaning. Semantic interpretation and task relevance belong to
later components. All findings retain a source reference and method details.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from parse.core.contracts import AnalysisResult, EvidenceRef, Issue, Provenance, SourceRef


KNOWLEDGE_STATES = {"observed", "statistical", "inferred", "confirmed", "unknown", "conflicting"}


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(item) for item in value]
    return str(value)


@dataclass(frozen=True)
class MethodSelection:
    method: str
    purpose: str
    rationale: str
    assumptions: tuple[str, ...] = ()


class AnalysisMethodSelector:
    """Small explainable selector, intentionally not a general rule engine."""

    def select_numeric_relationship(self, sample_size: int, pearson: float, spearman: float) -> MethodSelection:
        if sample_size < 3:
            return MethodSelection("insufficient_data", "numeric relationship", "Fewer than three complete pairs are available.")
        if abs(pearson) >= .8:
            return MethodSelection("pearson", "numeric relationship", "The observed linear association is strong.",
                                   ("Pairwise complete observations are used.",))
        return MethodSelection("spearman", "numeric relationship", "A rank-based monotonic measure is less dependent on linearity.",
                               ("This remains an association, not a causal test.",))

    def select_categorical_relationship(self, rows: int, dimensions: tuple[int, int]) -> MethodSelection:
        if rows < 3 or min(dimensions) < 2:
            return MethodSelection("insufficient_data", "categorical relationship", "The contingency table is too small.")
        return MethodSelection("contingency_table_cramers_v", "categorical relationship",
                               "Both categorical dimensions have at least two observed levels.",
                               ("Expected frequencies must be reviewed before inferential testing.",))


@dataclass(frozen=True)
class AnalysisRequest:
    """Input boundary for analysis; context is optional and never required for EDA."""

    dataset: pd.DataFrame
    source: SourceRef
    description: str | None = None
    context: Mapping[str, Any] = field(default_factory=dict)
    configuration: Mapping[str, Any] = field(default_factory=dict)
    sample: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, pd.DataFrame):
            raise TypeError("dataset must be a pandas DataFrame")
        if not isinstance(self.source, SourceRef):
            raise TypeError("source must be a SourceRef")
        object.__setattr__(self, "context", dict(self.context))
        object.__setattr__(self, "configuration", dict(self.configuration))


@dataclass(frozen=True)
class Finding:
    finding_id: str
    category: str
    subject: str | tuple[str, ...]
    observation: str
    method: str
    evidence: tuple[EvidenceRef, ...]
    assumptions: tuple[str, ...] = ()
    interpretation: str | None = None
    knowledge_state: str = "observed"
    limitations: tuple[str, ...] = ()
    result: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.knowledge_state not in KNOWLEDGE_STATES:
            raise ValueError(f"Unsupported knowledge_state: {self.knowledge_state}")
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(self, "limitations", tuple(self.limitations))
        object.__setattr__(self, "result", dict(self.result))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["evidence"] = [item.to_dict() for item in self.evidence]
        # Normalize subject to a plain string so DataFrame columns are homogeneous.
        # asdict() converts tuples to lists which causes PyArrow mixed-type errors.
        subj = value.get("subject")
        if isinstance(subj, (list, tuple)):
            value["subject"] = ", ".join(str(s) for s in subj)
        return _json_value(value)


@dataclass(frozen=True)
class DistributionProfile:
    count: int
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    median: float | None = None
    variance: float | None = None
    standard_deviation: float | None = None
    quantiles: Mapping[str, float | None] = field(default_factory=dict)
    iqr: float | None = None
    mad: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    zero_fraction: float | None = None
    outlier_count: int = 0
    outlier_method: str | None = None


@dataclass(frozen=True)
class AttributeProfile:
    name: str
    observed_type: str
    representation: str
    row_count: int
    non_null_count: int
    missing_count: int
    missing_rate: float
    unique_count: int
    cardinality_rate: float
    value_domain: Mapping[str, Any] = field(default_factory=dict)
    distribution: DistributionProfile | None = None
    structural_roles: tuple[str, ...] = ()
    temporal_properties: Mapping[str, Any] = field(default_factory=dict)
    quality_findings: tuple[str, ...] = ()
    provenance: Provenance | None = None
    knowledge_state: str = "observed"

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class DatasetProfile:
    row_count: int
    column_count: int
    column_names: tuple[str, ...]
    duplicate_row_count: int
    memory_bytes: int


@dataclass(frozen=True)
class StructuralProfile:
    empty_columns: tuple[str, ...]
    constant_columns: tuple[str, ...]
    near_constant_columns: tuple[str, ...]
    candidate_keys: tuple[str, ...]
    candidate_index_columns: tuple[str, ...]
    type_counts: Mapping[str, int]


@dataclass
class EDAResult:
    request: AnalysisRequest
    dataset_profile: DatasetProfile
    structural_profile: StructuralProfile
    attributes: list[AttributeProfile]
    findings: list[Finding]
    limitations: list[str]
    provenance: Provenance
    data_modified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _json_value({
            "source": self.request.source.to_dict(),
            "dataset_profile": asdict(self.dataset_profile),
            "structural_profile": asdict(self.structural_profile),
            "attributes": [attribute.to_dict() for attribute in self.attributes],
            "findings": [finding.to_dict() for finding in self.findings],
            "limitations": self.limitations,
            "provenance": self.provenance.to_dict(),
            "data_modified": self.data_modified,
        })

    def summary(self) -> str:
        quality = sum(item.category == "quality" for item in self.findings)
        relationships = sum(item.category == "relationship" for item in self.findings)
        return (f"Dataset contains {self.dataset_profile.row_count} rows and "
                f"{self.dataset_profile.column_count} columns; {quality} quality "
                f"finding(s) and {relationships} relationship finding(s) were observed.")


class AnalysisOrchestrator:
    """Sequence structural, attribute, relationship, and quality analysis."""

    def analyze(self, request: AnalysisRequest) -> EDAResult:
        frame = request.dataset
        dataset = DatasetProfile(
            row_count=len(frame),
            column_count=len(frame.columns),
            column_names=tuple(str(name) for name in frame.columns),
            duplicate_row_count=int(frame.duplicated().sum()),
            memory_bytes=int(frame.memory_usage(deep=True).sum()),
        )
        attributes = [self._attribute(frame, name, request.source) for name in frame.columns]
        structure = self._structure(frame, attributes)
        findings = self._attribute_findings(attributes, request.source)
        findings.extend(self._relationship_findings(frame, attributes, request.source))
        findings.extend(self._temporal_findings(attributes, request.source))
        if dataset.duplicate_row_count:
            findings.append(self._finding(request.source, "duplicate_rows", "quality", "dataset",
                f"{dataset.duplicate_row_count} duplicate row(s) were observed.",
                "duplicate_row_count", {"count": dataset.duplicate_row_count}))
        if not len(frame):
            findings.append(self._finding(request.source, "empty_dataset", "quality", "dataset",
                "The dataset has no rows.", "shape_inspection", {}))
        if not len(frame.columns):
            findings.append(self._finding(request.source, "empty_schema", "quality", "dataset",
                "The dataset has no columns.", "shape_inspection", {}))
        limitations = [
            "EDA reports observations and statistical signals; it does not establish domain meaning.",
            "An unusual observation is not automatically an error and no values were changed.",
            "Candidate structural roles require contextual confirmation.",
        ]
        return EDAResult(
            request=request, dataset_profile=dataset, structural_profile=structure,
            attributes=attributes, findings=findings, limitations=limitations,
            provenance=Provenance(sources=[request.source], operation="dataset_analysis",
                                  method="parse_eda_v1", created_at=datetime.now(timezone.utc).isoformat()),
        )

    def _attribute(self, frame: pd.DataFrame, name: Any, source: SourceRef) -> AttributeProfile:
        series = frame[name]
        non_null = series.dropna()
        observed_type, representation = self._type(series)
        unique = int(non_null.nunique())
        roles: list[str] = []
        name_text = str(name).strip().lower()
        if unique and unique == len(non_null) and ("id" in name_text or name_text in {"key", "index"}):
            roles.append("candidate_identifier")
        if observed_type == "categorical" and 1 < unique <= min(20, max(2, len(non_null) // 5)):
            roles.append("candidate_grouping")
        distribution = None
        domain: dict[str, Any] = {}
        if observed_type == "numeric" and len(non_null):
            numeric = pd.to_numeric(non_null, errors="coerce").dropna()
            distribution = self._distribution(numeric)
        if observed_type in {"categorical", "boolean", "text"}:
            domain["frequencies"] = _json_value(non_null.astype(str).value_counts().head(20).to_dict())
            if observed_type == "text":
                lengths = non_null.astype(str).str.len()
                domain.update({"length_min": int(lengths.min()) if len(lengths) else None,
                               "length_max": int(lengths.max()) if len(lengths) else None,
                               "length_mean": float(lengths.mean()) if len(lengths) else None})
        temporal = self._temporal_properties(series) if observed_type == "temporal" else {}
        quality = []
        if series.isna().any():
            quality.append("missingness")
        if unique == 0:
            quality.append("empty")
        return AttributeProfile(
            name=str(name), observed_type=observed_type, representation=representation,
            row_count=len(series), non_null_count=int(series.notna().sum()),
            missing_count=int(series.isna().sum()), missing_rate=float(series.isna().mean()) if len(series) else 0.0,
            unique_count=unique, cardinality_rate=unique / len(non_null) if len(non_null) else 0.0,
            value_domain=domain, distribution=distribution, structural_roles=tuple(roles),
            temporal_properties=temporal, quality_findings=tuple(quality),
            provenance=Provenance(sources=[source], operation="attribute_analysis", method="pandas_v1"),
        )

    @staticmethod
    def _type(series: pd.Series) -> tuple[str, str]:
        if pd.api.types.is_bool_dtype(series):
            return "boolean", str(series.dtype)
        if pd.api.types.is_numeric_dtype(series):
            return "numeric", str(series.dtype)
        if pd.api.types.is_datetime64_any_dtype(series):
            return "temporal", str(series.dtype)
        values = series.dropna()
        if len(values):
            dates = pd.to_datetime(values, errors="coerce", format="mixed")
            if dates.notna().mean() >= 0.9:
                return "temporal", "parseable_datetime"
            text = values.astype(str)
            if text.nunique() / len(text) <= 0.2 or text.str.len().mean() <= 24:
                return "categorical", "string_category"
            return "text", "string_text"
        return "unknown", str(series.dtype)

    @staticmethod
    def _distribution(values: pd.Series) -> DistributionProfile:
        q = values.quantile([.25, .5, .75])
        median = float(q.loc[.5])
        deviations = (values - median).abs()
        q1, q3 = float(q.loc[.25]), float(q.loc[.75])
        iqr = q3 - q1
        outliers = ((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)) if iqr else pd.Series(False, index=values.index)
        return DistributionProfile(
            count=int(values.size), minimum=float(values.min()), maximum=float(values.max()),
            mean=float(values.mean()), median=median, variance=float(values.var(ddof=1)) if len(values) > 1 else 0.0,
            standard_deviation=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            quantiles={"q25": q1, "q50": median, "q75": q3}, iqr=iqr,
            mad=float(deviations.median()), skewness=float(values.skew()) if len(values) >= 3 else None,
            kurtosis=float(values.kurt()) if len(values) >= 4 else None,
            zero_fraction=float((values == 0).mean()), outlier_count=int(outliers.sum()), outlier_method="IQR_1.5",
        )

    @staticmethod
    def _temporal_properties(series: pd.Series) -> dict[str, Any]:
        parsed = pd.to_datetime(series, errors="coerce", format="mixed").dropna()
        if len(parsed) < 2:
            return {"parseable_count": int(len(parsed)), "sufficient_observations": False}
        delta = parsed.sort_values().diff().dropna().dt.total_seconds()
        median = delta.median() if len(delta) else 0
        return {"parseable_count": int(len(parsed)), "minimum": parsed.min().isoformat(),
                "maximum": parsed.max().isoformat(), "ordered": bool(parsed.is_monotonic_increasing),
                "duplicate_count": int(parsed.duplicated().sum()),
                "median_interval_seconds": float(median) if len(delta) else None,
                "gap_count": int((delta > median * 1.5).sum()) if len(delta) and median else 0,
                "sufficient_observations": True}

    @staticmethod
    def _structure(frame: pd.DataFrame, attributes: list[AttributeProfile]) -> StructuralProfile:
        empty = tuple(a.name for a in attributes if a.non_null_count == 0)
        constant = tuple(a.name for a in attributes if a.unique_count == 1)
        near = tuple(a.name for a in attributes if a.non_null_count and a.cardinality_rate <= .01 and a.unique_count > 1)
        keys = tuple(a.name for a in attributes if a.non_null_count == len(frame) and a.unique_count == len(frame) and len(frame) > 0)
        indices = tuple(a.name for a in attributes if a.name.lower() in {"id", "index", "row_id", "record_id"})
        counts = pd.Series([a.observed_type for a in attributes]).value_counts().to_dict()
        return StructuralProfile(empty, constant, near, keys, indices, {str(k): int(v) for k, v in counts.items()})

    def _attribute_findings(self, attributes: list[AttributeProfile], source: SourceRef) -> list[Finding]:
        findings: list[Finding] = []
        for attribute in attributes:
            evidence = (EvidenceRef(source.source_id, "observed_from", locator=attribute.name),)
            if attribute.missing_count:
                findings.append(Finding(f"missing:{attribute.name}", "quality", attribute.name,
                    f"{attribute.missing_count} value(s) are missing.", "missingness_count", evidence,
                    result={"missing_count": attribute.missing_count, "missing_rate": attribute.missing_rate},
                    knowledge_state="statistical"))
            if attribute.distribution and attribute.distribution.outlier_count:
                findings.append(Finding(f"unusual:{attribute.name}", "distribution", attribute.name,
                    f"{attribute.distribution.outlier_count} observation(s) are potentially unusual under the IQR rule.",
                    "IQR_1.5", evidence, limitations=("Unusualness is not evidence of error.",),
                    result={"count": attribute.distribution.outlier_count}, knowledge_state="statistical"))
            if attribute.unique_count == 0:
                findings.append(Finding(f"empty:{attribute.name}", "quality", attribute.name,
                    "The attribute contains no observed values.", "non_null_count", evidence,
                    result={"count": 0}, knowledge_state="observed"))
            if "candidate_identifier" in attribute.structural_roles:
                findings.append(Finding(f"role:{attribute.name}", "structure", attribute.name,
                    "The attribute is a candidate identifier based on uniqueness and naming.",
                    "uniqueness_and_name_heuristic", evidence,
                    interpretation="Candidate identifier, not confirmed semantic meaning.",
                    knowledge_state="inferred", limitations=("Confirm with dataset context.",)))
        return findings

    def _relationship_findings(self, frame: pd.DataFrame, attributes: list[AttributeProfile], source: SourceRef) -> list[Finding]:
        findings: list[Finding] = []
        selector = AnalysisMethodSelector()
        numeric = [a.name for a in attributes if a.observed_type == "numeric"]
        for index, left in enumerate(numeric):
            for right in numeric[index + 1:]:
                pair = frame[[left, right]].dropna()
                if len(pair) < 3:
                    continue
                pearson = float(pair[left].corr(pair[right], method="pearson"))
                spearman = float(pair[left].corr(pair[right], method="spearman"))
                selection = selector.select_numeric_relationship(len(pair), pearson, spearman)
                method = selection.method
                value = pearson if method == "pearson" else spearman
                if abs(value) < .7:
                    continue
                findings.append(self._finding(source, f"relationship:{left}:{right}", "relationship", (left, right),
                    f"The attributes have a strong observed {method} association ({value:.3f}).", method,
                    {"pearson": pearson, "spearman": spearman, "sample_size": len(pair),
                     "selected_method": method, "selection_rationale": selection.rationale},
                    assumptions=selection.assumptions,
                    limitations=("Association does not establish causation or practical importance.",)))
        categorical = [a.name for a in attributes if a.observed_type in {"categorical", "boolean"}]
        for index, left in enumerate(categorical):
            for right in categorical[index + 1:]:
                table = pd.crosstab(frame[left], frame[right])
                selection = selector.select_categorical_relationship(int(table.to_numpy().sum()), table.shape)
                if selection.method == "insufficient_data":
                    continue
                observed = table.to_numpy(dtype=float)
                expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
                chi2 = float(((observed - expected) ** 2 / np.where(expected == 0, 1, expected)).sum())
                v = float(np.sqrt(chi2 / (observed.sum() * max(1, min(table.shape) - 1))))
                if v >= .3:
                    findings.append(self._finding(source, f"association:{left}:{right}", "relationship", (left, right),
                        f"The categorical attributes show an observed association (Cramer's V {v:.3f}).",
                        "contingency_table_cramers_v", {"cramers_v": v, "chi_square": chi2,
                        "sample_size": int(observed.sum()), "expected_frequency_min": float(expected.min()),
                        "selection_rationale": selection.rationale},
                        assumptions=selection.assumptions,
                        limitations=("This is descriptive and does not establish causation.",)))
        return findings

    def _temporal_findings(self, attributes: list[AttributeProfile], source: SourceRef) -> list[Finding]:
        return [self._finding(source, f"temporal:{a.name}", "temporal", a.name,
            f"The attribute has parseable temporal values covering {a.temporal_properties.get('minimum')} to {a.temporal_properties.get('maximum')}.",
            "timestamp_parsing_and_ordering", dict(a.temporal_properties),
            limitations=("Temporal semantics and sampling intent require confirmation.",))
            for a in attributes if a.observed_type == "temporal" and a.temporal_properties.get("sufficient_observations")]

    @staticmethod
    def _finding(source: SourceRef, finding_id: str, category: str, subject: str | tuple[str, ...], observation: str,
                 method: str, result: Mapping[str, Any], assumptions: tuple[str, ...] = (),
                 limitations: tuple[str, ...] = ()) -> Finding:
        return Finding(finding_id, category, subject, observation, method,
                       (EvidenceRef(source.source_id, "observed_from", locator=str(subject)),),
                       assumptions=assumptions, limitations=limitations, result=result,
                       knowledge_state="statistical" if category in {"relationship", "distribution"} else "observed")


class DatasetAnalysisAnalyzer:
    """Adapter implementing the generic PARSE ``Analyzer`` capability."""

    def analyze(self, inputs: Sequence[Any], **context: Any) -> AnalysisResult:
        if len(inputs) != 1 or not isinstance(inputs[0], pd.DataFrame):
            raise TypeError("DatasetAnalysisAnalyzer expects one pandas DataFrame")
        source = context.get("source") or SourceRef("in_memory_dataset", "user_input", label="DataFrame")
        result = AnalysisOrchestrator().analyze(AnalysisRequest(inputs[0], source, context=context))
        return AnalysisResult(
            result_id=f"analysis:{source.source_id}", outputs=result.to_dict(),
            input_refs=[EvidenceRef(source.source_id, "observed_from")], method="parse_eda_v1",
            provenance=result.provenance,
            issues=[Issue("empty_dataset", "error", "The dataset has no rows.", source=source)] if not len(inputs[0]) else [],
        )