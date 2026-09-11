# PARSE Dataset Analysis: Phases 1-8

The dataset-analysis core observes structured tabular data and produces
traceable evidence. It does not clean, transform, impute, delete, or assign
domain meaning to values. Semantic candidates, relevance, and human
confirmation are later stages.

## Boundary

```text
Dataset -> structural and statistical EDA -> evidence-backed profile
                                      |
                                      +-> human review / semantic analysis later
```

`AnalysisRequest` contains a DataFrame, a `SourceRef`, and optional description,
context, configuration, and sampling metadata. Domain context is optional for
EDA. `AnalysisOrchestrator` sequences dataset inspection, attribute profiling,
relationship analysis, temporal analysis, and quality findings.

## Contracts

- `DatasetProfile` records shape, column names, duplicate rows, and memory size.
- `StructuralProfile` records empty, constant, near-constant, candidate key,
  candidate index, and observed type counts.
- `AttributeProfile` records observed type and representation, cardinality,
  missingness, value frequencies, text lengths, numeric distribution, temporal
  properties, structural roles, quality indicators, and provenance.
- `Finding` records an observation, subject, method, evidence references,
  assumptions, limitations, result values, and `knowledge_state`.
- Knowledge states include `observed`, `statistical`, and `inferred`; the
  contract also reserves `confirmed`, `unknown`, and `conflicting` for later
  human/context stages. Candidate roles are never confirmed domain truth.

The `DatasetAnalysisAnalyzer` adapter implements the generic PARSE `Analyzer`
protocol and returns the shared `AnalysisResult` envelope.

## Method policy

Numeric attributes receive count, missingness, cardinality, extrema, mean,
median, sample variance and standard deviation, quartiles, IQR, MAD, skewness,
kurtosis, zero concentration, and an IQR-based candidate-unusual count. IQR is
used as a baseline because it is transparent and does not require a normality
assumption. A candidate-unusual observation is not an error.

Numeric pairs use Pearson when the observed linear association is strong and
Spearman otherwise, with a minimum pairwise sample size of three. Both values
are retained so method selection is inspectable. Categorical pairs use a
contingency table and Cramer's V when both dimensions have at least two
categories. Expected frequencies are recorded for assumption review; no
automatic significance claim is made.

Temporal attributes are identified by native datetime types or values that
parse successfully at least 90% of the time. The profile records coverage,
ordering, duplicate timestamps, median interval, and candidate gaps. It does
not infer the temporal semantics or apply rolling/seasonal methods without a
structure-specific reason.

## Quality and provenance

Missingness, empty attributes, duplicate rows, and candidate unusual numeric
values are quality evidence. Findings reference the source and attribute or
pair locator. Provenance records the operation, method, source, and creation
time. The input DataFrame is never modified.

Known limitations: type classification and structural roles are heuristics;
functional dependencies, units, entity meaning, validity, practical
significance, causality, and task relevance remain unknown until supported by
context or human confirmation.

## Semantic and relevance stages

`SemanticAnalyzer` consumes the EDA result and generates `SemanticCandidate`
records. Each candidate carries the attribute, candidate meaning, role,
evidence, method, assumptions, confidence, limitations, and knowledge state.
Name patterns, observed types, value domains, and optional supplied context are
evidence for a candidate only. They are not domain facts.

`HumanContext` stores `ConfirmationRecord` values for confirm, reject, modify,
unknown, and conflict actions. Confirmed context is explicit and traceable; it
does not mutate the DataFrame, the EDA result, or a candidate silently.

`RelevanceAnalyzer` requires a `RelevanceRequest` with an operation objective.
It may return potentially relevant, potentially irrelevant, selected, or
clarification candidates. These labels are scoped to that request and are
never attached permanently to an attribute.

## Orchestration and synthesis

`AnalysisPipeline` sequences EDA, semantic candidate generation, optional
task-specific relevance, and `AnalysisSynthesizer`. The synthesizer formats
structured evidence only; it does not calculate statistics or invent meaning.
`AnalysisMethodSelector` provides small, inspectable explanations for method
selection and reports insufficient data rather than forcing a test.

The generic PARSE `Analyzer` adapter remains `DatasetAnalysisAnalyzer`; its
outputs use the shared `AnalysisResult` envelope, source references, and
provenance. No BF/Sinter, FAISS, LLM, or Streamlit dependency is required by
the analytical core. The Streamlit Data Explorer presents the complete bundle
through separate Attributes, Relationships, Quality, Semantic Candidates,
Unknowns/Confirmation, and Summary views. The existing cleaner continues to
operate only after explicit user approval.

## Evaluation and limitations

The controlled tests cover descriptive statistics, known correlations,
categorical association, missingness, candidate unusual values, temporal gaps,
semantic uncertainty, confirmation state, relevance scoping, and method
selection. Evaluation metrics for false-positive rates, semantic accuracy,
heterogeneous datasets, and runtime should be added against representative
labelled datasets; more findings alone are not a success criterion.