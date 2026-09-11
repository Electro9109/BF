# PARSE Architecture Extraction

Status: Phase 4 Automated Data Understanding complete at engine level
Date: 2026-09-07
Source implementation: cleaned BF prototype

## 1. Purpose

PARSE (Process, Analysis, Retrieval, Synthesis & Evaluation) is a knowledge-processing and decision-support system for heterogeneous technical information.

The current BF application is the first evidence base for PARSE. It is not being renamed into PARSE and its folder structure is not being copied blindly. This document records the concepts and boundaries that can be justified by the cleaned BF implementation.

The current design goal is:

```text
Raw information
    -> processing and validation
    -> traceable knowledge and records
    -> retrieval and analysis
    -> synthesis
    -> evaluation and human decision support
```

The minimal PARSE runtime package is implemented under `parse/`. This document
continues to record the architectural reasoning and boundaries.

The finalized minimal contracts are recorded separately in
[PARSE_CONTRACTS.md](PARSE_CONTRACTS.md). This document explains the
reasoning and architectural boundaries; the contract document defines the
Phase 3 entry point.

## Phase 2 Decisions

- Documents, structured observations, interpretations, predictions, retrieved evidence, synthesized explanations, and evaluations remain semantically distinct.
- `SourceRef`, `Provenance`, `Issue`, and `EvidenceRef` are shared lineage concepts.
- `KnowledgeItem` is a flexible envelope for processed source information, not a universal domain schema.
- Operation-specific results remain distinct while sharing provenance and evidence references.
- Missingness, validation issues, repairs, and conflicts remain visible.
- Applicability indicators are separate from calibrated statistical uncertainty.
- FAISS, SentenceTransformers, Qwen, ML models, and Streamlit remain replaceable implementations.
- Evaluation is an explicit responsibility and is not yet assumed to be mature in BF.

## 2. What BF Actually Manipulates

BF currently handles three broad classes of information.

### 2.1 Source information

- Technical text documents under `docs/`.
- Structured Sinter experiment data in Excel and CSV files.
- User-entered chemistry, atmosphere, burden, and natural-language conditions.

### 2.2 Processed knowledge and records

- `Chunk`: a retrievable document segment with content, source, topic, and section metadata.
- Sinter experiment rows: structured observations with chemistry, burden, temperatures, pressure-related measurements, and source row identity.
- Parsed conditions: normalized atmosphere and burden values plus parser warnings.
- Validated feature rows: model-ready numerical representations derived from observations or user input.

### 2.3 Derived results

- Predictions for `Ts`, `Tm`, and `Tm-Ts`.
- Historical similarity results and applicability-distance indicators.
- Retrieval results containing chunks and relevance scores.
- Grounded generated explanations based on retrieved context and analytical results.

These are not currently one universal entity. They have different shapes and evidence status. A common future envelope may be useful, but BF does not yet justify collapsing them into one class.

## 3. Knowledge Model: Current Working Hypothesis

The minimum useful PARSE representation should distinguish the following:

```text
Source
  identifies where information came from

Observation / Record
  represents information directly present in a structured or unstructured source

Interpretation
  represents a normalized or parsed representation of source information

Derived Result
  represents an output produced by analysis, retrieval, comparison, or synthesis

Evidence Reference
  connects a result back to source observations or retrieved material
```

This is a conceptual model, not a finalized class hierarchy.

### 3.1 Document knowledge

A document currently needs at least:

- stable source identity
- source location or name
- raw or retained content
- optional topic and section metadata
- processing status or validation issues

A `Chunk` is a derived document segment used by retrieval. It should not be confused with the complete document or with authoritative knowledge.

### 3.2 Structured records

A structured record currently needs at least:

- record identity or source row identity
- process/domain identity
- observed attributes and values
- units or representation where known
- source reference
- missing-value state
- validation issues

Sinter `ExperimentRow` remains domain-specific. PARSE should not assume that every process has the same chemistry, conditions, or target fields.

### 3.3 Derived results

A derived result should identify:

- operation that produced it
- input evidence or record references
- output values or text
- method/model identity where relevant
- applicability or uncertainty indicators where available
- warnings and limitations

A model prediction is not an observation. A generated explanation is not a source fact. Both must remain traceable to their inputs and methods.

## 4. Core Responsibilities

### 4.1 Processing

BF evidence:

- document decoding and section parsing
- topic extraction
- structured experiment loading
- column mapping
- numeric coercion
- incomplete-data reporting
- `Tm-Ts` consistency repair
- condition parsing and normalization
- feature construction

Candidate PARSE responsibility:

> Transform heterogeneous source material into usable, traceable records while preserving source identity, missingness, and validation issues.

Processing must not silently equate completeness with correctness. Repairs such as `Tm-Ts = Tm - Ts` need an explicit policy and should preserve the original discrepancy when that discrepancy may be meaningful.

### 4.2 Analysis

BF evidence:

- fixed-range feature scaling
- chemistry/atmosphere/burden interaction construction
- Ts/Tm/Tm-Ts prediction
- nearest-neighbor distance
- historical experiment comparison

Candidate PARSE responsibility:

> Apply a domain or general analytical method to structured records and return a result with applicability, limitations, and evidence references.

BF demonstrates a Sinter analysis module, not a universal ML framework.

### 4.3 Retrieval

BF evidence:

- embedding generation
- vector indexing
- topic detection and candidate filtering
- semantic search
- section-based reranking
- source-bearing retrieval results

Candidate PARSE responsibility:

> Retrieve relevant evidence from processed knowledge without making a particular vector technology part of the conceptual model.

The retrieval result must preserve source and chunk identity. A relevance score is an ordering signal, not truth or confidence.

### 4.4 Synthesis

BF evidence:

- context-grounded prompt construction
- retrieved-document answer generation
- prediction plus historical comparison plus theory explanation
- hallucination/failure fallback behavior

Candidate PARSE responsibility:

> Combine evidence and analytical results into a human-readable result while keeping unsupported claims distinguishable from source-backed information.

An LLM is one synthesis implementation. It is not the knowledge source.

### 4.5 Evaluation

BF evidence is currently limited:

- model benchmark metrics are produced during training
- prediction distance is used as an applicability indicator
- retrieval scores are exposed in UI diagnostics
- a lexical overlap guard detects some unsupported generations

BF does not yet provide a mature Evaluation layer.

Candidate PARSE responsibility:

> Assess the quality, applicability, grounding, uncertainty, and limitations of processed, retrieved, analytical, or synthesized results.

Evaluation should be treated as a first-class responsibility to define, not retrofitted as a confidence label.

## 5. Core Contract Rationale

The contract boundaries are deliberately smaller than a full service/interface framework. Their finalized field-level definitions are in `PARSE_CONTRACTS.md`; this section records why they are needed.

### 5.1 Evidence reference

A result should be able to refer to:

- source identifier
- source location, row, section, or chunk identifier
- relationship type, such as `observed_from`, `derived_from`, or `retrieved_from`

### 5.2 Processing result

Potential contents:

- processed record or document representation
- source reference
- validation issues
- missing attributes
- transformations applied

### 5.3 Retrieval result

Potential contents:

- retrieved knowledge item or chunk
- relevance score
- source/evidence reference
- retrieval method metadata
- filtering or fallback information

### 5.4 Analysis result

Potential contents:

- named outputs
- input record references
- method/model identity
- applicability indicators
- warnings and limitations

### 5.5 Synthesis result

Potential contents:

- generated or assembled content
- evidence references
- analytical-result references
- unsupported/uncertain sections where detectable
- synthesis method metadata

### 5.6 Evaluation result

Potential contents:

- evaluated result reference
- metric or criterion
- value and interpretation
- evaluation dataset/context
- limitations

These contracts should only become code after at least one real BF workflow needs the boundary and its fields are agreed.

## 6. Provenance, Missingness, and Conflict

### 6.1 Provenance

Every retained observation or derived result should be traceable to its source or inputs.

Minimum provenance information may include:

- source type: document, spreadsheet, CSV, user input, model, or generated output
- source identifier/path
- row, section, or chunk location where available
- processing operation
- method/model version where relevant
- timestamp or run identity when reproducibility requires it

### 6.2 Incomplete information

Missing values must remain distinguishable from zero, default, inferred, and not-applicable values.

The BF cleanup now reports missing experiment values while retaining the rows. PARSE should preserve that behavior and make the state available to downstream analysis.

### 6.3 Conflicting information

Conflicting source values should not be silently overwritten.

If reconciliation is justified:

```text
original observation(s)
    + reconciliation rule and evidence
    + resulting interpretation
```

must remain available. The current BF `Tm-Ts` repair is useful operationally, but its original mismatch should eventually be preserved as a validation discrepancy rather than discarded.

### 6.4 Derived versus observed information

- A measured temperature is observed information.
- A parsed atmosphere is an interpretation of source text.
- A repaired `Tm-Ts` is a derived or reconciled value.
- A model prediction is a derived analytical result.
- A generated explanation is synthesized output.

These categories must not be flattened into indistinguishable facts.

## 7. Domain Module Boundary

Sinter/BF-specific knowledge currently includes:

- chemistry columns and units
- burden terminology and test types
- atmosphere and condition vocabulary
- Sinter feature ranges and interactions
- Ts/Tm/Tm-Ts targets
- Sinter topic vocabulary
- metallurgical prompt policy
- historical experiment schema

This knowledge belongs in a Sinter/BF domain module when PARSE extraction begins.

PARSE Core should not depend on:

- Sinter
- FeO, Ts, Tm, Basicity, or burden names
- Sinter model feature names
- BF topic keywords
- metallurgical prompt instructions

## 8. Infrastructure Boundary

Current BF technologies are implementation choices:

| Capability | Current implementation | PARSE concept |
|---|---|---|
| Embeddings | SentenceTransformers | evidence representation/retrieval support |
| Vector index | FAISS | retrieval implementation |
| Language model | local Qwen/Transformers | synthesis implementation |
| Prediction models | Random Forest/XGBoost | analysis implementation |
| Storage | filesystem, CSV, Excel, SQLite index artifacts | source/storage infrastructure |
| Interface | Streamlit | interface implementation |

The intended dependency shape is:

```text
PARSE capability/contract
        ↓
replaceable implementation
        ↓
FAISS, SentenceTransformers, Qwen, Random Forest, XGBoost, Streamlit
```

Infrastructure should not define the meaning of evidence, analysis, or synthesis.

## 9. BF to PARSE Mapping

| BF component | PARSE responsibility | Classification |
|---|---|---|
| `data.parser` | document processing | reusable mechanism with domain policy inputs |
| `data.loader` | document processing and Chunk creation | candidate core mechanism |
| `data.sinter_schemas` | structured Sinter records | Sinter domain module |
| `data.experiments_loader` | structured-data ingestion/validation | mechanism plus Sinter schema |
| `retrieval.embeddings` | embedding implementation | infrastructure |
| `retrieval.faiss_index` | vector index implementation | infrastructure |
| `retrieval.topic_filter` | candidate filtering mechanism | reusable mechanism plus domain vocabulary |
| `retrieval.reranker` | ranking adjustment mechanism | reusable mechanism plus domain policy |
| `retrieval.retriever` | retrieval orchestration | candidate core capability |
| `ml.feature_processing` | Sinter processing and feature construction | Sinter domain analysis support |
| `ml.condition_parser` | condition interpretation | Sinter domain processing |
| `ml.predictor` | model-backed analysis | Sinter analysis implementation |
| `ml.similarity` | historical comparison/applicability | Sinter analysis implementation |
| `llm.loader` | model loading | infrastructure |
| `llm.generator` | generation mechanism | infrastructure/capability implementation |
| `llm.domain_prompts` | synthesis policy | Sinter domain module |
| `pipeline.rag_pipeline` | retrieval plus synthesis orchestration | application workflow |
| `pipeline.prediction_pipeline` | analysis orchestration | application workflow |
| `pipeline.hybrid_pipeline` | combined analysis/retrieval/synthesis | application workflow |
| `app_web.py` | human interface | interface |

## 10. Dependency Direction

Desired direction:

```text
Interface
    ↓
Application workflows
    ↓
Core contracts and capabilities
    ↓
Domain modules and policies
    ↓
Infrastructure implementations
```

Practical constraints:

- UI should consume pipeline/application results.
- Pipelines should not require Streamlit.
- Core retrieval should not require an LLM or ML feature processing.
- Core processing contracts should not require FAISS or Qwen.
- Domain modules may depend on Core contracts.
- Infrastructure may implement Core-facing capabilities.
- Source and derived-result references should flow upward through the workflow.

The cleaned BF prototype now supports several of these boundaries, but the future PARSE Core should formalize only the boundaries proven by more than one workflow or by a genuine substitution need.

## 11. Evaluation Gap

Evaluation is the least mature PARSE responsibility in BF.

Questions still requiring design:

- What is a retrieval-quality result independent of FAISS scores?
- How is synthesis grounding evaluated beyond lexical overlap?
- How is prediction applicability reported without calling distance confidence?
- How are human reviews represented?
- How are conflicting evidence and unresolved uncertainty evaluated?
- Which metrics belong to a model, a workflow, or a knowledge item?

Until these are answered, PARSE should expose limitations and evidence rather than invent a universal confidence abstraction.

## 12. Resolved Phase 2 Questions

The following Phase 2 questions now have working decisions:

1. Documents and structured records share provenance/evidence vocabulary, but not one universal domain schema.
2. `Chunk` is treated as a retrieval projection of document knowledge rather than the complete PARSE knowledge model.
3. Observations, interpretations, and derived results remain distinguishable by kind and operation-specific result contracts.
4. Provenance identifies sources, parent results, operations, methods, and precise locations where available.
5. Conflicting values remain as source observations plus an explicit reconciliation or interpretation issue.
6. Validation issues are retained with severity and affected field; they do not automatically discard a record.
7. Historical similarity is an Analysis/comparison capability and may provide Evaluation applicability indicators.
8. Evaluation is criteria-driven and must not be represented by an uncalibrated universal confidence value.

## 13. Open Architectural Questions

1. Should documents and structured records share a common `KnowledgeItem` envelope, or only share provenance/evidence contracts?
2. Is `Chunk` a core knowledge representation or an infrastructure-specific retrieval projection?
3. Should observations, interpretations, and derived results be separate types or status fields on one record?
4. What is the minimum stable provenance structure that supports documents, experiment rows, user inputs, predictions, and generated text?
5. How should conflicting values be represented without forcing premature reconciliation?
6. Which validation issues are warnings, blocking errors, or data-quality observations?
7. Does historical similarity belong to Analysis, Evaluation, or both?
8. What evaluation contract is justified by actual BF workflows?
9. Which retrieval policy inputs should be supplied by a domain module rather than stored in configuration?
10. What is the smallest Core contract that a second process module could use without knowing Sinter terminology?

## 14. Next Controlled Step

Phase 2 is complete as a design phase. Phase 3 now has dependency-free contracts, operation protocols, a generic workflow coordinator, and a tested BF adapter under `parse/adapters/`. The next phase is to evolve the Sinter module and infrastructure integrations without moving the entire BF repository at once.
