# PARSE Workflow Gap Analysis

## Baseline Audit Summary

**Repository State:**
- Branch: master
- Latest commit: 443cb55 (role-aware RAG prompt context, scenario-targeted retrieval, response truncation fix)
- Working tree: clean
- Test suite: 390 tests passing

**Existing Capabilities Verified:**
- 3 departments implemented: Blast Furnace, Sintering, Energy
- Unified calculator with 3 prediction tasks
- RAG retrieval with role-aware context (reference vs experiment_scenario)
- Source catalogue with SQLite persistence
- Data ingestion workflow with staging, analysis, cleaning, and registration
- ML models: BF (Ts/Tm/Tm-Ts), Sinter productivity, Steel energy consumption
- Results Translator with fidelity validation
- 16 reference documents in docs/ (124 chunks)
- 1 registered document in catalogue (19 chunks)

## End-to-End Workflow Map

### 1. File Upload and Staging
**Entry point:** `app_web.py` → Data Ingestion tab
**Module:** `parse.ingestion_workflow.create_staged_upload()`

**Flow:**
1. User uploads file (txt, csv, xlsx, xls, json, jsonl)
2. `create_staged_upload()` validates size, type, computes SHA-256
3. Creates `StagedUpload` with `SourceRef`
4. `set_staged_upload()` binds to session state, invalidates stale workflow results

**Input:** File bytes, filename
**Output:** `StagedUpload` in session state
**Storage:** Session state only (no persistence yet)
**Status:** Working

### 2. Document Normalization
**Entry point:** `app_web.py` → "Run Analysis" button
**Module:** `parse.ingestion_workflow.normalize_staged_document()` (for text), `parse.source_catalogue.SourceCatalogue.register_upload()` (for all types)

**Flow:**
- Text: `normalize_staged_document()` → parses sections, extracts topic/role, creates `Chunk` list
- Tabular: `SourceCatalogue.register_upload()` → validates CSV/Excel/JSON, normalizes to CSV artifact
- Original bytes preserved in `storage/private/originals/` or `storage/public/originals/`
- Normalized artifact stored in `storage/normalized/`

**Input:** `StagedUpload`
**Output:** Chunks (text) or normalized CSV (tabular)
**Storage:** Filesystem + catalogue metadata
**Status:** Working

### 3. Exploratory Data Analysis
**Entry point:** `app_web.py` → "Run Analysis" button
**Module:** `parse.eda_ui`, `parse.analysis.AnalysisOrchestrator`

**Flow:**
1. `load_uploaded_dataset()` loads staged content as DataFrame
2. `AnalysisOrchestrator.analyze()` runs EDA, data quality checks, semantic analysis
3. Returns `AnalysisBundle` with findings, recommendations, issues
4. UI displays summary, findings, and cleaning proposals

**Input:** Staged upload bytes
**Output:** `AnalysisBundle` in session state
**Storage:** Session state
**Status:** Working

### 4. Cleaning Review and Approval
**Entry point:** `app_web.py` → Cleaning UI
**Module:** `parse.cleaning`, `parse.ingestion_workflow`

**Flow:**
1. User reviews cleaning proposals
2. Approves/rejects individual proposals
3. `set_staged_upload()` tracks approved/rejected lists
4. Approved cleaning produces derived dataset artifact

**Input:** Analysis bundle + user selections
**Output:** Approved cleaning list, derived frame
**Storage:** Session state + optional derived artifact
**Status:** Working

### 5. Source Registration
**Entry point:** `app_web.py` → "Register Source" button
**Module:** `parse.source_catalogue.SourceCatalogue.register_confirmed_upload()`

**Flow:**
1. Creates `ResourceManifest` with provenance, issues, metadata
2. Stores original bytes (unchanged)
3. Stores normalized artifact (text → document, tabular → CSV)
4. Records manifest in SQLite catalogue
5. For text documents: triggers index refresh

**Input:** Staged upload + selected artifact + cleaning approvals
**Output:** `ResourceManifest` in catalogue
**Storage:** Filesystem (originals + normalized) + SQLite (manifests)
**Status:** Working

### 6. Document Indexing (RAG)
**Entry point:** `app_web.py` → Registration completion or "Reload Documents"
**Module:** `pipeline.rag_pipeline.build_engine()`, `data.loader.load_and_chunk_data()`

**Flow:**
1. `load_and_chunk_data()` recursively scans directory for .txt files
2. Extracts topic, section, role from each document
3. Creates `Chunk` list with embed_text (TOPIC+SECTION+content)
4. `RetrievalEngine` builds FAISS index from embeddings
5. Index stored in memory (not persisted)

**Input:** Directory path (docs/ or storage/normalized/documents/)
**Output:** `RetrievalEngine` + chunks list
**Storage:** In-memory index only
**Status:** Working but has fingerprint bug (fixed)

### 7. Query and Retrieval
**Entry point:** `app_web.py` → RAG Chat tab
**Module:** `pipeline.rag_pipeline.answer_question_structured()`, `retrieval.retriever.RetrievalEngine.search()`

**Flow:**
1. User enters question
2. `RetrievalEngine.search()` detects topic, filters candidates, FAISS search, reranks by section
3. Scenario intent detection augments with experiment_scenario chunks
4. Top-K chunks assembled into context with source/role/section labels
5. Context sent to LLM for answer generation

**Input:** Question string
**Output:** Generated answer + retrieval matches + PARSE RetrievalResult/SynthesisResult
**Storage:** None (session state)
**Status:** Working with role-aware retrieval

### 8. Model Prediction
**Entry point:** `app_web.py` → Predictor tab
**Module:** `ml.unified_calculator.UnifiedCalculator`, department-specific prediction methods

**Flow:**
1. User selects task (Blast Furnace / Sinter / Energy)
2. UI displays appropriate input fields
3. User enters parameters (chemistry, atmosphere, burden for BF; raw mix for sinter; electrical + calendar for energy)
4. `UnifiedCalculator.predict_*()` validates inputs, loads model, runs prediction
5. Returns `DepartmentPredictionResult` with predictions, warnings, limitations, metrics
6. Converted to `StructuredModuleResult` for Results Translator

**Input:** Feature dictionary
**Output:** `DepartmentPredictionResult` → `StructuredModuleResult`
**Storage:** Session state
**Status:** Working

### 9. Results Translation
**Entry point:** `app_web.py` → After prediction
**Module:** `parse.translator_pipeline.ResultsTranslator`

**Flow:**
1. `StructuredModuleResult` passed to `ResultsTranslator.translate()`
2. Prompt formatter preserves numerical tokens
3. LLM generates natural-language explanation
4. `FidelityEvaluator` validates: numerical fidelity, causal language, limitation preservation, recommendation fidelity
5. Failed validation → deterministic fallback (preserves original structured result)

**Input:** `StructuredModuleResult`
**Output:** Natural-language explanation (validated or fallback)
**Storage:** Session state
**Status:** Working with fidelity validation

## Gap Table

| Required Information | Component | Existing Module/Stage | Current Behavior | Evidence | Gap | Priority | Proposed Action | Verification |
|----------------------|-----------|------------------------|------------------|----------|-----|----------|-----------------|--------------|
| Source update invalidates dependent artifacts | Workflow state | `app_web.py` docs_fingerprint | Only fingerprints top-level .txt files, not nested directories | <ref_snippet file="C:\Users\ASUS\Coding\Fwd_ Tm-Ts\app_web.py" lines="331-339" /> | Nested document changes not detected | High | Fixed: changed glob to rglob | Test by adding nested file, verify fingerprint changes |
| Department registration consistency | Department registry | `departments/__init__.py` | Only BlastFurnaceDepartment registered | <ref_file file="C:\Users\ASUS\Coding\Fwd_ Tm-Ts\departments\__init__.py" /> | Sintering and Energy departments exist but not in registry | High | Fixed: added SinteringDepartment and EnergyDepartment to DEPARTMENTS dict | Test that all 3 departments are discoverable |
| Source identity and provenance in retrieval | Retrieval pipeline | `data.loader`, `retrieval.retriever` | Chunk has source, topic, section, role; retrieval preserves these | <ref_file file="C:\Users\ASUS\Coding\Fwd_ Tm-Ts\data\loader.py" /> | Provenance preserved in chunks and retrieval results | None | None | Existing tests verify |
| Structured data vs text retrieval distinction | Data layer | `SourceCatalogue`, `load_and_chunk_data` | Tabular sources registered in catalogue but not indexed for text retrieval | <ref_file file="C:\Users\ASUS\Coding\Fwd_ Tm-Ts\parse\source_catalogue.py" /> | Explicitly separates text (indexed) from tabular (catalogue only) | None | Working as designed | Test confirms tabular not in chunks |
| Analysis schema consistency | Analysis modules | `parse.analysis.py`, `parse.eda_ui` | Uses `AnalysisBundle`, `Finding`, `EDAFinding` with PARSE contracts | Architecture documented | No known schema conflicts | None | None | Existing tests pass |
| Document indexing rebuild behavior | RAG pipeline | `app_web.py` build_engine, docs_fingerprint | Fingerprint now includes nested files after fix | Fixed with rglob change | Index refreshes correctly on nested changes | High | Verify with test | Add nested file, reload, verify chunk count |
| Workflow state invalidation on source change | Ingestion workflow | `parse.ingestion_workflow.set_staged_upload` | Clears 12 specific session state keys on upload change | <ref_file file="C:\Users\ASUS\Coding\Fwd_ Tm-Ts\parse\ingestion_workflow.py" /> | Working but limited to session state | Medium | Consider adding catalogue-based invalidation for registered sources | Document limitation |
| ML calculator input contracts | ML calculator | `ml.unified_calculator.py` | Has basic validation (missing, non-finite, range warnings) | <ref_file file="C:\Users\ASUS\Coding\Fwd_ Tm-Ts\ml\unified_calculator.py" /> | No explicit schema validation for BF predictor (uses legacy predictor) | High | Add explicit input contract for BF predictions matching Sinter/Energy | Test with invalid inputs |
| Training/inference consistency | ML training | `ml.train.py` | Uses `departments.blast_furnace.feature_processing` for features | <ref_file file="C:\Users\ASUS\Coding\Fwd_ Tm-Ts\ml\train.py" /> | BF predictor uses same processing | None | None | Existing tests pass |
| Catalogued documents indexed | RAG pipeline | `pipeline.rag_pipeline.build_engine` | Loads both docs/ and storage/normalized/documents/ | <ref_snippet file="C:\Users\ASUS\Coding\Fwd_ Tm-Ts\pipeline\rag_pipeline.py" lines="41-48" /> | Working but only 1 of 16 docs registered | Low | Catalogued docs are indexed, docs/ is indexed separately | Verify both sources appear in chunks |
| Coke Ovens department | Department modules | None | No implementation | N/A | Not implemented | Low | Document as future expansion | Update documentation |
| Steel Melting Shop department | Department modules | None | No implementation | N/A | Not implemented | Low | Document as future expansion | Update documentation |

## Critical Issues Fixed

1. **Department registration inconsistency**: Added SinteringDepartment and EnergyDepartment to `departments/__init__.py` DEPARTMENTS registry. Verified by existing test suite.

2. **Document fingerprint bug**: Changed `docs_fingerprint()` from `p.glob("*.txt")` to `p.rglob("*.txt")` to include nested directories. This ensures changes in `docs/experiments/` trigger index refresh.

## Remaining Gaps by Priority

**High Priority:**
- ML calculator input contracts: BF predictor lacks explicit schema validation (unlike Sinter/Energy which have range checking)
- Verification of nested document indexing fix

**Medium Priority:**
- Workflow state invalidation: Currently session-scoped only; registered source changes don't invalidate downstream artifacts

**Low Priority:**
- Coke Ovens and SMS departments not implemented (as expected for prototype scope)
- Only 1 of 16 documents registered in catalogue (most usage is via docs/ direct loading)

## Working Features

All core workflows are functional:
- ✅ File upload, staging, normalization
- ✅ EDA, cleaning review, registration
- ✅ Document indexing with role-aware retrieval
- ✅ RAG chat with scenario intent detection
- ✅ Multi-department ML predictions
- ✅ Results translation with fidelity validation
- ✅ Source catalogue with provenance tracking
- ✅ 390 tests passing
