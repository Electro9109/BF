# PARSE Minimal Contracts

Status: Phase 4 Automated Data Understanding complete at engine level
Date: 2026-09-07

These contracts describe the smallest stable information boundaries justified by the cleaned BF prototype. They are design contracts, not yet a runtime package. Phase 3 will implement them only after review.

## 1. Design Decisions

### Decision 1: Do not use one universal knowledge class for everything

Documents, structured observations, interpretations, predictions, retrieved evidence, generated explanations, and evaluations have different semantics. They should not be flattened into one indistinguishable record.

They may share provenance and issue representations, but their operation-specific contracts remain distinct.

### Decision 2: Provenance is shared infrastructure for knowledge

Every retained source-derived or derived item must be able to identify where it came from and what operation produced it.

### Decision 3: Missingness and validation issues are first-class metadata

Missing, invalid, inferred, repaired, and not-applicable values must remain distinguishable. A validation issue must not automatically discard the associated record.

### Decision 4: Evidence references connect outputs to inputs

Retrieval, analysis, synthesis, and evaluation results refer to source or result identifiers rather than copying authority into the result itself.

### Decision 5: Contracts use flexible attributes at the domain boundary

PARSE Core should not define Sinter chemistry fields, BF topics, or universal experiment columns. Domain modules provide typed or validated attributes and PARSE carries them through stable envelopes.

## 2. Shared Value Objects

### 2.1 SourceRef

Identifies an origin of information.

```text
SourceRef
- source_id: stable string
- source_type: document | csv | spreadsheet | user_input | model | generated_output
- locator: optional path, row, section, chunk, or external identifier
- label: optional human-readable name
```

`source_id` must be stable within the repository or persisted run context. A filesystem path alone is not always sufficient as an identity.

### 2.2 Provenance

Describes how an item relates to its sources.

```text
Provenance
- sources: list[SourceRef]
- operation: optional operation name
- method: optional implementation/model identifier
- parent_ids: list of prior knowledge/result identifiers
- created_at: optional timestamp
```

A provenance record may identify multiple sources because reconciliation, comparison, and synthesis can combine evidence.

### 2.3 Issue

Represents a quality or interpretation concern without silently deleting data.

```text
Issue
- code: stable string
- severity: info | warning | error
- message: human-readable explanation
- field: optional affected attribute
- source: optional SourceRef
```

Examples include `missing_value`, `invalid_numeric_value`, `repaired_derived_value`, `topic_fallback`, and `generation_failed`.

### 2.4 EvidenceRef

Connects an output to a source or prior result.

```text
EvidenceRef
- ref_id: source or result identifier
- relation: observed_from | interpreted_from | derived_from | retrieved_from | compared_with
- locator: optional precise location
```

## 3. Knowledge Envelopes

### 3.1 KnowledgeItem

This is the shared envelope for processed source information, not a universal replacement for domain records.

```text
KnowledgeItem
- item_id: stable string
- kind: document | chunk | structured_record | observation | interpretation
- attributes: mapping of domain-defined values
- provenance: Provenance
- issues: list[Issue]
- incomplete: boolean
```

The `attributes` mapping is intentionally not made into a universal schema. A Sinter record may expose chemistry and burden fields; another process may expose entirely different attributes.

The current BF `Chunk` can be adapted into a `KnowledgeItem(kind="chunk")`, while `ExperimentRow` can become a domain-specific `KnowledgeItem(kind="structured_record")` without changing their current implementations in Phase 3.

### 3.2 DerivedResult

All analytical, comparison, synthesis, and evaluation outputs share lineage semantics but retain operation-specific contracts.

```text
DerivedResult
- result_id: stable string
- result_type: analysis | comparison | synthesis | evaluation
- values: mapping of named outputs
- evidence: list[EvidenceRef]
- provenance: Provenance
- issues: list[Issue]
```

`values` may contain numbers, text, structured values, or references. Their meaning is defined by the producing operation or domain module.

## 4. Operation Contracts

### 4.1 Processing

Processing consumes raw source material and returns one or more `KnowledgeItem` values plus issues.

```text
process(source) -> list[KnowledgeItem]
```

Processing must preserve:

- source identity
- retained raw or source-linked information where practical
- missingness
- validation issues
- transformations and repairs in provenance

The contract does not require every input to produce the same item kind.

### 4.2 Retrieval

Retrieval returns evidence, not authoritative answers.

```text
retrieve(query, candidates/policy) -> RetrievalResult

RetrievalResult
- query: original or normalized query
- items: list[RetrievedEvidence]
- fallback: optional reason
- issues: list[Issue]

RetrievedEvidence
- item: KnowledgeItem or item_id
- score: numeric ordering score
- evidence: EvidenceRef
- metadata: implementation-independent metadata
```

FAISS, embeddings, keyword search, and reranking are implementations. Scores must not be called confidence without calibration.

### 4.3 Analysis

Analysis consumes domain records and returns named analytical outputs with applicability context.

```text
analyze(inputs, method) -> AnalysisResult

AnalysisResult
- outputs: mapping of named values
- input_refs: list[EvidenceRef]
- method: method/model identity
- applicability: optional Applicability
- issues: list[Issue]
- provenance: Provenance
```

```text
Applicability
- indicators: mapping of named values
- interpretation: optional text
- limitations: list[str]
```

A nearest-neighbor distance may be an applicability indicator. It must not automatically become statistical confidence.

### 4.4 Synthesis

Synthesis combines evidence and/or analysis results into a user-facing result.

```text
synthesize(evidence, analyses, policy) -> SynthesisResult

SynthesisResult
- content: text or structured presentation
- evidence_refs: list[EvidenceRef]
- analysis_refs: list[EvidenceRef]
- method: implementation/policy identity
- issues: list[Issue]
- limitations: list[str]
```

An LLM may implement synthesis, but generated content remains derived and must retain references to supplied evidence.

### 4.5 Evaluation

Evaluation assesses another result or operation rather than pretending every result is inherently reliable.

```text
evaluate(target, criteria, context) -> EvaluationResult

EvaluationResult
- target_ref: EvidenceRef
- criteria: list[str]
- metrics: mapping of named values
- interpretation: optional text
- evidence_refs: list[EvidenceRef]
- issues: list[Issue]
- limitations: list[str]
```

Initial criteria may include retrieval relevance, prediction error, applicability coverage, synthesis grounding, data completeness, and human review. The criteria must be explicit for each evaluation.

## 5. BF Mapping to Contracts

| BF behavior | Contract interpretation |
|---|---|
| document parser and loader | Processing -> `KnowledgeItem(kind="document"/"chunk")` |
| Sinter experiment loader | Processing -> domain structured record |
| missing-value warning | `Issue(code="missing_value")` |
| `Tm-Ts` repair | interpretation/derived value with provenance and repair issue |
| parsed natural-language condition | `KnowledgeItem(kind="interpretation")` or domain input record |
| retrieval chunk and score | `RetrievedEvidence` |
| Ts/Tm/Tm-Ts model output | `AnalysisResult` |
| historical nearest experiments | comparison `DerivedResult` plus applicability indicators |
| grounded LLM answer | `SynthesisResult` |
| benchmark and grounding checks | `EvaluationResult` |

## 6. Explicit Non-Contracts

The following are not PARSE Core concepts:

- FAISS
- SentenceTransformer
- Qwen
- Random Forest
- XGBoost
- Streamlit
- RAG as a business entity
- `PredictionService` or `SynthesisService` merely because the current code has pipelines
- a universal experiment schema
- an uncalibrated confidence field

They may appear in infrastructure or domain implementations, but they should not define the Core model.

## 7. Phase 3 Entry Criteria

Phase 3 may begin when these decisions are accepted:

1. `SourceRef`, `Provenance`, `Issue`, and `EvidenceRef` are the shared lineage vocabulary.
2. `KnowledgeItem` is an envelope, not a universal domain schema.
3. Operation-specific results remain distinct.
4. Processing preserves incomplete and conflicting information.
5. Analysis reports applicability separately from calibrated uncertainty.
6. Synthesis remains evidence-grounded and replaceable.
7. Evaluation is explicit and criteria-driven.

The first Phase 3 implementation should be a small Core package containing only the shared value objects and the minimum result envelopes needed to adapt one BF workflow. It should not migrate the whole repository at once.

## 8. Phase 3 First Slice

The initial implementation now exists under `parse/core/`:

- `contracts.py` implements the shared value objects, knowledge envelope, and operation result envelopes.
- `operations.py` defines implementation-neutral `Processor`, `Retriever`, `Analyzer`, `Synthesizer`, and `Evaluator` protocols.
- `parse/core` imports only Python standard-library modules and PARSE Core modules.
- `tests/test_parse_core.py` verifies validation, lineage round-tripping, operation protocols, and forbidden implementation dependencies.

This is not yet a complete PARSE Core integration. BF adapters, domain modules, infrastructure implementations, and end-to-end Core workflows remain Phase 3 follow-up work.

The first integration slice is now also implemented:

- `parse/core/workflow.py` runs the generic Processing -> Retrieval -> Analysis -> Synthesis -> Evaluation chain.
- `parse/adapters/bf.py` maps existing BF chunks, retrieval matches, prediction results, and generated answers into Core contracts.
- `tests/test_parse_bf_adapter.py` verifies BF lineage and evidence preservation.

This completes the minimal Phase 3 Core objective. It does not migrate BF into `parse/` or claim that the Sinter module is complete.

Phase 4 Increment 1 adds `DataUnderstanding` and structured EDA result types
above these Core contracts. The capability emits source-linked findings and
next-action suggestions without modifying its input dataset.
