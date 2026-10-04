# AUDIT.md — PARSE Repository Audit & Extensibility Analysis

**Audit Date:** 2026-10-04  
**Baseline Test Suite Status:** 166 passed, 0 failed (pytest 9.1.1 on Python 3.13.14)  
**Verification Standard:** Read-only audit with code inspection and runtime verification.

---

## 1. Confirmed Defects

### Defect 1: Data Explorer "Simplify" Helpers Display Silent Data Loss (`app_web.py`)
- **Severity:** High (Runtime UI defect / display data loss).
- **Location:** `app_web.py` lines 369–395, invoked at lines 1049, 1058, 1069.
- **Root Cause:**
  In Task 23, `_simplify_attribute_row` and `_simplify_finding_row` were added to produce clean summary tables in the Streamlit Data Explorer tab. However, the helper functions were written assuming dictionary keys from legacy `parse/eda.py` (`ColumnProfile` and `EDAFinding`), whereas `app_web.py` passes the dictionary output of modern `parse/analysis.py` (`AttributeProfile.to_dict()` and `Finding.to_dict()`):
  - `_simplify_attribute_row` reads:
    - `row.get("inferred_type", row.get("type", "N/A"))` -> Modern dictionary key is `"observed_type"`. Result: UI table displays `"Type": "N/A"`.
    - `row.get("missing_fraction", 0)` -> Modern dictionary key is `"missing_rate"`. Result: UI table displays `"Missing %": "0.0%"` even when missing data is present.
    - `row.get("summary", "")` -> Modern `AttributeProfile` has no `"summary"` field. Result: UI table displays `"Summary": ""`.
  - `_simplify_finding_row` reads:
    - `row.get("kind", "N/A")` -> Modern `Finding` has no `"kind"` field (it has `"category"` and `"knowledge_state"`). Result: UI table displays `"Type": "N/A"`.
    - `row.get("message", "N/A")` -> Modern `Finding` key is `"observation"`. Result: UI table displays `"Finding": "N/A"`.
- **Evidence / Reproduction:**
  1. Direct runtime inspection of `bundle.eda.attributes[0].to_dict()` on a sample dataframe shows keys: `['name', 'observed_type', 'representation', 'row_count', 'non_null_count', 'missing_count', 'missing_rate', 'unique_count', 'cardinality_rate', 'value_domain', 'distribution', 'structural_roles', 'temporal_properties', 'quality_findings', 'provenance', 'knowledge_state']`.
  2. Running `_simplify_attribute_row(bundle.eda.attributes[0].to_dict())` yields: `{'Column': 'a', 'Type': 'N/A', 'Missing %': '0.0%', 'Unique': 4, 'Summary': ''}`.
  3. Running `_simplify_finding_row(bundle.eda.findings[0].to_dict())` yields: `{'Type': 'N/A', 'Finding': 'N/A', 'Evidence': '1 ref(s)'}`.
  4. Notice why existing tests (`test_app_web_data_shapes.py`) did not catch this: `test_app_web_does_not_use_legacy_eda_attribute_paths` checks source text for AST access like `finding.kind` (which was removed from the AST and replaced with `row.get("kind")` dictionary lookup). `test_app_web_data_explorer_shape_contract` tests property access on `bundle.eda.findings` directly, but never executes `_simplify_finding_row` or `_simplify_attribute_row`.
- **Impact:** While the app does not raise an unhandled exception (due to `.get("...", "N/A")`), the user is presented with columns filled entirely with `"N/A"` for Type and Finding, and `"0.0%"` for Missingness, completely breaking the Data Explorer summary tab.

---

## 2. Consumer Map: Legacy vs. Modern EDA

Below is the complete map of every consumer across code, tests, `eval/`, UI, and documentation:

| Component / File | Legacy Engine (`parse/eda.py`) | Modern Engine (`parse/analysis.py` + `semantic_analysis.py`) | Notes / Relationship |
|---|---|---|---|
| `parse/eda.py` | Defines `DataUnderstanding`, `LegacyEDAResult`, `EDAFinding`, `ColumnProfile`, `NextAction` | None | Legacy engine implementation |
| `parse/analysis.py` | None | Defines `AnalysisOrchestrator`, `EDAResult`, `Finding`, `AttributeProfile`, `DatasetProfile`, `StructuralProfile` | Modern canonical analysis engine |
| `parse/semantic_analysis.py` | None | Imports `AnalysisOrchestrator`, `EDAResult`, `Finding` | Adds semantic interpretation, relevance, and candidate synthesis on top of modern EDA |
| `parse/eda_ui.py` | Imports `DataUnderstanding`, `LegacyEDAResult` for `profile_uploaded_dataset` | Imports `AnalysisRequest`, `AnalysisPipeline`, `AnalysisBundle` for `analyze_uploaded_dataset` / `analyze_loaded_dataset` | Dual engine split: legacy used for `profile_uploaded_dataset`, modern used for `analyze_*` |
| `parse/cleaning.py` | Dual compatibility in `detect()`: checks `hasattr(resolved_eda, "structural_profile")` vs legacy attributes; supports `legacy_eda` | Consumes `structural_profile.candidate_index_columns`, `attribute_profiles`, `resolved_eda.findings` | Fully adapted to accept either, but tests pass both |
| `eval/build_parse_eval_set.py` | Imports `DataUnderstanding` and calls `DataUnderstanding().profile(frame)` | **None** | **Eval/Training Drift:** Generates Explainer eval examples solely from legacy `EDAFinding` and `CleaningIssue` |
| `app_web.py` | No direct import (legacy imports forbidden by test) | Uses `AnalysisBundle` (`bundle.eda`) via `parse/eda_ui.py` | Uses modern bundle at runtime, but display helpers read legacy dictionary keys (Defect 1) |
| `tests/test_eda.py` | 8 tests targeting `DataUnderstanding`, `LegacyEDAResult`, `ColumnProfile` | None | Direct test suite for legacy engine |
| `tests/test_analysis.py` | None | 7 tests targeting `AnalysisOrchestrator`, `EDAResult`, `Finding`, `AttributeProfile` | Direct test suite for modern engine |
| `tests/test_eda_ui.py` | Tests `profile_uploaded_dataset` | Tests `analyze_uploaded_dataset`, `analyze_loaded_dataset` | Direct tests for both entry points |
| `tests/test_cleaning.py` | Passes legacy `DataUnderstanding().profile(df)` in tests | Passes modern `AnalysisOrchestrator().analyze(req)` in tests | Both used across test cases |
| `tests/test_cleaning_integration.py`| None | Uses `AnalysisOrchestrator` | Tests modern integration with cleaner |
| `tests/test_build_parse_eval_set.py`| Indirectly via `build_parse_eval_set.py` | None | Tests eval builder which runs legacy engine |
| `tests/test_app_web_data_shapes.py` | Asserts legacy attributes don't appear in `app_web.py` source text | Asserts modern attribute access on `bundle.eda` | Guards against legacy attribute access |
| `setup_check.py` | Imports `parse.eda` | Imports `parse.analysis`, `parse.semantic_analysis` | Verifies imports for both |
| `PARSE_EDA.md` | Documents legacy `DataUnderstanding` | None | Stale engine documentation |
| `PARSE_DATA_ANALYSIS.md` | Explains differences with legacy | Documents modern `AnalysisOrchestrator` and contracts | Architecture documentation for modern engine |

---

## 3. Capability Parity Matrix: Legacy vs. Modern Output

Comparing the output of `DataUnderstanding.profile()` (`LegacyEDAResult`) vs. `AnalysisOrchestrator.analyze()` (`EDAResult`):

| Finding / Capability | Legacy Output (`parse/eda.py`) | Modern Output (`parse/analysis.py` / `semantic_analysis.py`) | Parity Status & Action Plan |
|---|---|---|---|
| **Empty Dataset / Empty Schema** | Issues: `empty_dataset`, `empty_schema` | Findings: `empty_dataset`, `empty_schema` (category: `quality`, subject: `dataset`) | **Parity Achieved** (Modern produces findings). |
| **Row Count / Column Count / Memory** | Top-level fields on `LegacyEDAResult` | Encapsulated in `EDAResult.dataset_profile` | **Structural difference**, fully supported in modern. |
| **Duplicate Rows** | Finding `duplicate_rows` (category: `quality`, kind: `observation`), count in attributes | Finding `duplicate_rows` (category: `quality`, method: `duplicate_row_count`, result: `{"count": N}`) | **Equivalent semantics**, slightly different container dict (`result` vs `attributes`). |
| **Missing Values** | Finding `missing_<col>` (category: `quality`, kind: `observation`), `attributes={"missing_count", "missing_fraction"}` | Finding `missing:<col>` (category: `quality`, method: `missingness_count`, result: `{"missing_count", "missing_rate"}`) | **Equivalent semantics**; finding ID separator differs (`_` vs `:`). |
| **Sparse Column Warning** | Finding `sparse_<col>` when `missing_fraction >= 0.5` | Not emitted as a standalone finding (tracked in `AttributeProfile.missing_rate`) | **Gap in Modern:** Modern records the rate but doesn't emit a `sparse_<col>` finding. |
| **Statistical Outliers** | Finding `outliers_<col>` (category: `statistical`, kind: `observation`, method IQR) | Finding `unusual:<col>` (category: `distribution`, method IQR_1.5, includes MAD outlier count) | **Equivalent semantics**; finding ID and category differ (`statistical`/`outliers_<col>` vs `distribution`/`unusual:<col>`). |
| **Numeric Skewness Finding** | Finding `skew_<col>` when `abs(skewness) >= 1.0` (category: `statistical`, kind: `observation`) | Computed in `DistributionProfile.skewness`, but **no standalone Finding is emitted** | **Gap in Modern:** Explainer eval set explicitly asserts `skew_skewed_value` in `test_build_parse_eval_set.py`. Modern must emit skew finding or adapter must synthesize it. |
| **Mixed Numeric/Text Finding** | Finding `mixed_values_<col>` (category: `quality`, kind: `observation`) when `0 < numeric_fraction < 1` | Not emitted as a finding in `analysis.py`; detected ad-hoc in `cleaning.py` | **Gap in Modern:** Modern classifies type as `text` or `categorical` but does not emit a mixed values finding. |
| **Candidate Identifiers** | Finding `identifier_<col>` (category: `logical`, kind: `interpretation`) | Finding `role:<col>` (category: `structure`, kind: `role:<col>`, `interpretation="Candidate identifier..."`) | **Equivalent semantics**; modern labels role in `AttributeProfile.structural_roles`. |
| **Candidate Target Inference** | Finding `target_<col>` (category: `logical`, kind: `interpretation`, based on keyword tokens e.g. target, temperature, etc.) | Evaluated downstream in `parse/semantic_analysis.py` (`RelevanceEvaluator`), **not in base EDA** | **Architectural difference:** Modern purposefully deferred domain relevance/target to `semantic_analysis.py`. |
| **Candidate Grouping** | Finding `group_<col>` (category: `logical`, kind: `interpretation`) | Role in `AttributeProfile.structural_roles = ("candidate_grouping",)` | **Gap in Modern Findings:** Role is on attribute, but no standalone finding is emitted. |
| **Strong Correlation (Numeric)** | Finding `relationship_<left>_<right>` if `abs(corr) >= 0.9` (Pearson only) | Finding `relationship:<left>:<right>` if `abs(val) >= 0.7` using explainable `AnalysisMethodSelector` (Pearson vs Spearman) | **Modern is Superior:** Dynamically selects method (Spearman vs Pearson) and captures non-linear associations. |
| **Categorical Association** | Not supported in legacy | Finding `association:<left>:<right>` using Cramer's V and Chi-Square contingency table | **Modern Capability Only** |
| **Group Difference Finding** | Finding `group_difference_<cat>_<num>` (computes mean spread across categorical groups) | Not emitted in modern `analysis.py` | **Gap in Modern:** Legacy detects numeric spread across categories; modern currently has Cramer's V for cat-cat and Pearson/Spearman for num-num. |
| **Temporal Sequence Findings** | Finding `temporal_<col>` (monotonic ordering check) | Finding `temporal:<col>` with deep temporal properties (sampling interval, gaps, monotonic) | **Modern is Superior** |
| **Next Action Suggestions** | List of `NextAction` dataclasses (`clean_prepare`, `analyse`, `explore_retrieve`, `build_prediction`, `generate_report`) | None in `analysis.py` | **Gap in Modern:** Modern intentionally did not generate workflow recommendations inside raw data analysis. |

---

## 4. Explainer Training Data & Eval Contract Audit

### Training Data Existence Check
- Checked filesystem for pre-existing training sets (`*.json`, `*.jsonl`, `*.csv` in `eval/`, `data/`, `MLModels/`, etc.).
- **Result:** No static fine-tuning training dataset files exist in the repository.
- What exists is:
  1. `eval/build_parse_eval_set.py`: Script to generate eval datasets on demand (defaulting to synthetic dataset).
  2. `eval/fidelity_checks.py`: Unit-tested fidelity harness scoring explanations against finding records.
  3. `eval/run_comparison.py`: Benchmark runner scoring base vs. adapter models.
  4. `KAGGLE_TRAINING.md`: Describes tabular ML model training off-repo (not Explainer LLM training).

### Explainer Data Contract Specification
The Explainer consumer contract (`eval/fidelity_checks.py`, `eval/run_comparison.py`, `tests/test_build_parse_eval_set.py`) strictly expects each record to match:
```python
{
    "finding_id": str,
    "source": str,        # "eda" or "cleaning_issue"
    "category": str,      # e.g., "quality", "statistical", "relationship", "distribution"
    "kind": str,          # "observation" or "interpretation"
    "message": str,       # Plain-language statement containing the facts and numbers
    "attributes": dict,   # Numeric/categorical key-value facts checked for numerical fidelity
    "limitations": list   # List of limitation strings checked by limitation_preserved()
}
```
**Specific Finding IDs pinned in tests:**
In `tests/test_build_parse_eval_set.py`, the following `finding_id` values are explicitly asserted:
- `"duplicate_rows"`
- `"missing_measurement"` (format: `missing_<col>`)
- `"outliers_measurement"` (format: `outliers_<col>`)
- `"skew_skewed_value"` (format: `skew_<col>`)

**Critical Finding on Contract Drift:**
Modern findings use `finding_id` format `missing:<col>`, `unusual:<col>`, `relationship:<left>:<right>`.
If modern findings were emitted directly into `eval/build_parse_eval_set.py` without an adapter, `tests/test_build_parse_eval_set.py` would **fail immediately**. Furthermore, any Explainer model trained on legacy IDs/messages would suffer prompt and evaluation drift.

---

## 5. Stale Documentation & Renamed Paths

Comprehensive scan of docs vs. current filesystem:

1. **`ml/feature_processing.py` references:**
   - Mentioned in `README.md` lines 153, 166.
   - **Current Location:** `departments/blast_furnace/feature_processing.py`. (The root `ml/` contains `predictor.py`, `condition_parser.py`, `similarity.py`, `train.py`, etc., but feature processing was moved to the blast furnace department module).
2. **`data/sinter_schemas.py` references:**
   - Mentioned in `BF_BASELINE.md` line 49 and `PARSE_ARCHITECTURE.md` line 381.
   - **Current Status:** Module was refactored into `departments/blast_furnace/bf_experiment_schema.py`.
3. **`config/sinter.py` references:**
   - Mentioned in `BF_BASELINE.md` line 61.
   - **Current Status:** Moved into `departments/blast_furnace/config.py`.
4. **StandardScaler vs. RangeScaler in `README.md`:**
   - `README.md` line 31 mentions `StandardScaler` objects in `scalers.pkl`, but lines 147–164 state `StandardScaler` was replaced by `RangeScaler`. In code (`departments/blast_furnace/feature_processing.py`), `RangeScaler` is indeed used.
5. **`PARSE_EDA.md`:**
   - Documents `parse.eda.DataUnderstanding` as the active engine without mentioning `parse.analysis.AnalysisOrchestrator` or `parse.semantic_analysis`.

---

## 6. Small Smells Audit

1. **Unused `request` variable in `eda_ui.analyze_uploaded_dataset`:**
   - **Confirmed.** In `parse/eda_ui.py`:
     ```python
     def analyze_uploaded_dataset(...):
         frame, source = load_uploaded_dataset(filename, content)
         request = AnalysisRequest(frame, source)  # <-- UNUSED
         return analyze_loaded_dataset(frame, source, objective, selected_attributes)
     ```
     `analyze_loaded_dataset` recreates its own `AnalysisRequest(frame, source)`. Line 79 is dead code.
2. **Unused dependencies (`PyPDF2`, `python-docx`):**
   - **Confirmed.** Searched entire repository for `PyPDF2` and `docx`. Neither is imported or used in any `.py` file. Text loading is implemented purely in `data/loader.py` for `.txt` files. They are legacy dependencies lingering in `requirements.txt`.
3. **Import-time `os.environ` mutation in `llm/loader.py`:**
   - **Confirmed.** In `llm/loader.py` lines 14–21: executes `os.environ["TRANSFORMERS_VERBOSITY"] = "error"`, wipes proxy env vars, etc., unconditionally at top-level import. While harmless for an offline app, top-level mutation has side effects on test suites and external integrations.
4. **Hardcoded confidence-distance thresholds in `ml/predictor.py`:**
   - **Confirmed.** In `ml/predictor.py` lines 277–282:
     ```python
     if dist <= 1.05:
         confidence = "high"
     elif dist <= 1.82:
         confidence = "medium"
     ```
     Distances are arbitrary thresholds derived from historical training data. Should be documented as empirical Blast Furnace thresholds or moved to department configuration.
5. **`ml/` importing `departments.blast_furnace` directly:**
   - **Confirmed.** `ml/train.py`, `ml/predictor.py`, `ml/similarity.py`, `ml/import_trained_model.py`, and `ml/export_training_data.py` directly import from `departments.blast_furnace.feature_processing`. Bypasses generic `parse.core.department.Department` protocol.
6. **`pickle.load` of externally trained models in `ml/import_trained_model.py`:**
   - **Confirmed.** In `ml/import_trained_model.py`, uses Python `pickle.load()` on arbitrary files passed via `--model-path`. Inherently insecure if loading untrusted files (arbitrary code execution vulnerability inherent to pickle).
7. **Re-encoding candidate subset in `retrieval/retriever.py`:**
   - **Confirmed.** In `retrieval/retriever.py` lines 97–98: When a topic filter selects a subset of chunks, it re-encodes their text with `self._embedder.encode(cand_texts)` rather than slicing the precomputed `self.embeddings` matrix by candidate indices. Causes latency spike on search.
8. **`app_web.py` monolith:**
   - **Confirmed.** Single file of 1,288 lines mixing Streamlit session state, HTML/CSS styling strings, helper transformations, and UI view rendering with minimal unit test coverage.

---

## 7. Extensibility Risks

1. **Second Department Extensibility Risk:**
   - `ml/` currently imports `departments.blast_furnace.feature_processing` directly.
   - To add a second department (e.g. Sintering or Steelmaking), ML predictors and training routines cannot be reused without refactoring `ml/` to consume an injected or configured `Department` instance.
2. **Plugging in the Explainer as a `Synthesizer`:**
   - `parse/core/operations.py` defines `Synthesizer(Protocol)` expecting `synthesize(evidence: RetrievalResult, analyses: Sequence[AnalysisResult] = (), **context) -> SynthesisResult`.
   - The Explainer LLM currently consumes raw findings dicts (`{finding_id, source, category, kind, message, attributes, limitations}`) via `eval/run_comparison.py`, rather than wrapping its output into `SynthesisResult`. A clear bridge is needed.
3. **Dual EDA Engine Maintenance:**
   - Having two engines (`parse/eda.py` and `parse/analysis.py`) creates cognitive overhead, double-testing requirements, and risks silent drift (as demonstrated by `build_parse_eval_set.py` using legacy while `app_web.py` uses modern).

---

## 8. Proposed Ordered Fix List

Each item is categorized as **[Safe]** (no breaking changes, no approval needed for Phase 2) or **[Needs Approval]** (touches contracts, models, or architectural boundaries).

| Item | Description | Category | Proposed Phase |
|---|---|---|---|
| **1** | Fix `app_web.py` simplify helpers to read modern `AttributeProfile` and `Finding` fields with fallback support; write regression test calling helpers on real `AnalysisBundle` | **Safe** | Phase 2 |
| **2** | Remove dead code `request = AnalysisRequest(...)` in `parse/eda_ui.py` | **Safe** | Phase 2 |
| **3** | Remove unused dependencies `PyPDF2` and `python-docx` from `requirements.txt` | **Safe** | Phase 2 |
| **4** | Fix stale documentation paths in `README.md`, `BF_BASELINE.md`, and `PARSE_ARCHITECTURE.md` | **Safe** | Phase 2 |
| **5** | Add guard tests in `tests/test_docs.py` that grep markdown documentation for obsolete module paths (`ml/feature_processing`, `data/sinter_schemas`, `config/sinter`) | **Safe** | Phase 2 |
| **6** | Create `docs/EXTENDING.md` documenting Department protocol implementation, adding analysis methods, and integrating Explainer as `Synthesizer` | **Safe** | Phase 2 |
| **7** | **Phase 3 EDA Unification:** Port missing capabilities (skew, mixed-values, grouping spread) into modern engine or unified adapter | **Needs Approval** | Phase 3 (Design gate) |
| **8** | **Phase 3 Versioned Explainer Adapter:** Provide adapter emitting `{finding_id, source, category, kind, message, attributes, limitations}` with legacy ID preservation for Explainer compatibility | **Needs Approval** | Phase 3 (Design gate) |
| **9** | **Phase 3 Deprecation of `parse/eda.py`:** Add deprecation warning and migrate callers (`eval/build_parse_eval_set.py`, `parse/eda_ui.py`, `parse/cleaning.py`) to unified path | **Needs Approval** | Phase 3 (Design gate) |
| **10** | Document smells for future tasks in `TASKS.md` Parking Lot: slice embeddings in `retriever.py`, defer `os.environ` mutation in `loader.py`, decouple `ml/` from BF department, replace `pickle` with `safetensors`/`skops` | **Safe** | Phase 2 / Parking Lot |

---
*Audit complete. Awaiting user review and authorization to proceed to Phase 2.*
