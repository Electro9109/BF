# PARSE Automated Data Understanding

Status: Phase 4 complete; Phase 5 cleaning slice implemented
Date: 2026-09-08

## Scope

The first Phase 4 slice profiles structured tabular data without changing it. It accepts an in-memory pandas `DataFrame` or a CSV/Excel file and produces:

- dataset dimensions;
- cautious column type inference;
- missingness and cardinality;
- numeric descriptive statistics;
- candidate identifiers;
- candidate target variables;
- candidate grouping variables;
- duplicate-row findings;
- numeric distributions, skewness, quantiles, and IQR-based outlier signals;
- temporal ordering observations;
- categorical/numeric group comparisons;
- mixed numeric/text representation findings;
- strong observed numeric relationships;
- evidence-backed findings and limitations;
- context-aware next-action suggestions.

The capability is implemented in [parse/eda.py](parse/eda.py) as `DataUnderstanding`.

## Important behavior

EDA is an understanding step, not a cleaning step:

- no rows, columns, or values are modified;
- missing values are reported, not filled or dropped;
- statistical anomalies are not automatically called errors;
- correlations are reported as observations, not causal explanations;
- identifiers and targets are candidates, not confirmed meanings;
- unsupported file formats are rejected explicitly.

## Current input support

- `.csv`
- `.xlsx`
- `.xls`
- in-memory pandas `DataFrame`

Example:

```python
from parse.eda import DataUnderstanding

result = DataUnderstanding.from_file("dataset.csv")
print(result.summary())
```

## Current result shape

`EDAResult` contains:

- `source` and `provenance`;
- `row_count` and `column_count`;
- `ColumnProfile` values;
- `EDAFinding` values with `EvidenceRef` references;
- `Issue` values;
- `NextAction` applicability suggestions;
- explicit limitations;
- `data_modified=False`.

Findings distinguish:

- `observation`: directly measured/profiled property;
- `interpretation`: cautious inference such as candidate identifier or target;
- `relationship`: observed numeric association;
- `quality`: missingness or duplicate evidence.

## BF check

The existing BF workbook was profiled read-only:

- 134 rows;
- 17 columns;
- 46 findings with the expanded engine;
- applicable initial actions: cleaning review, exploration/retrieval, and reporting.

This is a structural understanding result, not a replacement for the existing Sinter feature-processing path.

## Phase 5 cleaning integration

The EDA result can be passed to [parse/cleaning.py](parse/cleaning.py) as the
input to a controlled cleaning review. `DataCleaner` detects missing values,
duplicate rows, mixed numeric/text representations, and EDA-reported IQR
outliers. It proposes only conservative missing-value and exact-duplicate
transformations; mixed values and outliers remain review-only.

Transformations are never implicit. Approved proposal IDs are applied to a
deep copy and recorded in `ChangeRecord` values containing the original value,
new value, reason, method, affected row, and proposal lineage. The result keeps
the original frame, before/after quality snapshots, unresolved issues, and
cleaning provenance.

The Streamlit Data Explorer now retains the uploaded frame, displays cleaning
issues, allows proposal selection, and exports a machine-readable cleaning
result.

## Scope boundary

Phase 4 is complete as a reusable understanding engine. The current prototype
also exposes it through the `Data Explorer` Streamlit tab. The engine provides
the structured result and action choices; broader capability chaining and the
homepage remain Phase 10 work.

## Not yet implemented

The following remain later increments:

- domain-specific logical rules beyond general heuristics;
- advanced statistical tests and multivariate/contextual anomaly methods;
- advanced cleaning rules such as units, contextual anomalies, and domain-specific transformations;
- capability chaining into analysis, recommendation, and reporting;
- evaluation across a broader dataset collection beyond the synthetic tests and BF workbook.
