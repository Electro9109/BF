# PHASE 3 DESIGN PROPOSAL: EDA UNIFICATION & EXPLAINER ADAPTER

**Status:** AWAITING USER APPROVAL (Phase 3 Gate)  
**Objective:** Unify `parse/eda.py` (legacy) and `parse/analysis.py` (modern) into a single canonical source of truth while strictly preserving the Explainer training/eval data contract, maintaining 100% identical cleaning detection/confidence behavior, and deprecating the legacy engine with a clear removal roadmap.

---

## 1. Architectural Strategy: Canonical Engine & Single Source of Truth

- **Canonical Engine:** `parse/analysis.py` (`AnalysisOrchestrator` / `EDAResult` / `Finding`) remains the primary engine.
- **Port Missing Capabilities:** Add the 4 capabilities currently in legacy that are absent or incomplete in `parse/analysis.py`:
  1. **Skewness Finding:** When `abs(distribution.skewness) >= 1.0`, emit finding `skew:<col>` (category: `"shape"` to isolate from outlier detection, observation: `"Column '{col}' is strongly skewed (skewness {val:.2f})."`).
  2. **Mixed Numeric/Text Finding:** When `is_text_like(series)` and `0 < numeric_fraction < 1`, emit finding `mixed_values:<col>` (category: `"quality"`, observation: `"Column '{col}' mixes numeric-like and non-numeric values ({pct:.0%} parse as numbers)."`).
  3. **Sparse Column Finding:** When `attribute.missing_rate >= 0.5`, emit finding `sparse:<col>` (category: `"quality"`, observation: `"Column '{col}' is sparse with {pct:.0%} missing values."`).
  4. **Categorical-Numeric Group Difference Finding:** When a categorical column has low cardinality and a numeric column displays mean spread across groups, emit `group_difference:<cat>:<num>` (category: `"group_difference"` to prevent skewing `RelevanceAnalyzer` relationship scoring).
  *(Note on Next Actions: In modern PARSE architecture, next actions are workflow guidance, not raw data observations. We will provide an explicit method `generate_next_actions(eda_result: EDAResult) -> list[NextAction]` in `parse/analysis.py` so downstream callers have access to it without conflating it with statistical observations.)*

---

## 2. Versioned Explainer Adapter (`parse/explainer_adapter.py`)

To prevent any breaking change to the Explainer LLM's evaluation harness (`eval/build_parse_eval_set.py`, `eval/fidelity_checks.py`, `eval/run_comparison.py`), we establish a versioned adapter.

### Explainer Record Contract
The adapter translates modern `EDAResult` and `Finding` instances into the exact legacy contract:
```python
{
    "schema_version": "1.0",
    "finding_id": str,       # Preserves legacy finding_id mapping (see below)
    "source": "eda",
    "category": str,         # Preserved: "quality", "statistical", "relationship", "distribution"
    "kind": str,             # "observation" or "interpretation"
    "message": str,          # Plain language text carrying numbers
    "attributes": dict,      # Number dict for numerical_fidelity check
    "limitations": list      # Limitations list for limitation_preserved check
}
```

### Exact ID & Category Mapping Table

| Modern Canonical Finding ID | Category in Modern | Mapped Finding ID for Explainer | Mapped Category | Mapped Kind |
|---|---|---|---|---|
| `duplicate_rows` | `quality` | `duplicate_rows` | `quality` | `observation` |
| `missing:<col>` | `quality` | `missing_<col>` | `quality` | `observation` |
| `sparse:<col>` | `quality` | `sparse_<col>` | `quality` | `observation` |
| `mixed_values:<col>` | `quality` | `mixed_values_<col>` | `quality` | `observation` |
| `unusual:<col>` | `distribution` | `outliers_<col>` | `statistical` | `observation` |
| `skew:<col>` | `distribution` | `skew_<col>` | `statistical` | `observation` |
| `relationship:<col1>:<col2>` | `relationship` | `relationship_<col1>_<col2>` | `relationship` | `observation` |
| `association:<col1>:<col2>` | `relationship` | `association_<col1>_<col2>` | `relationship` | `observation` |
| `group_difference:<cat>:<num>` | `relationship` | `group_difference_<cat>_<num>` | `relationship` | `observation` |
| `temporal:<col>` | `temporal` | `temporal_<col>` | `logical` | `interpretation` |
| `role:<col>` | `structure` | `identifier_<col>` | `logical` | `interpretation` |

**Preservation Guarantee:**
This mapping guarantees 100% backward compatibility for the exact IDs asserted in `tests/test_build_parse_eval_set.py` (`"duplicate_rows"`, `"missing_measurement"`, `"outliers_measurement"`, `"skew_skewed_value"`).

---

## 3. Migration Plan for Consumers

1. **`eval/build_parse_eval_set.py`:**
   - Migrate from `DataUnderstanding().profile(frame)` to `AnalysisOrchestrator().analyze(AnalysisRequest(frame, source))`.
   - Pipe output through `to_explainer_eval_record(eda_result)`.
   - `build_parse_eval_set.py` will now evaluate the exact representation that runs in production!
2. **`parse/eda_ui.py`:**
   - `profile_uploaded_dataset(filename, content)` will use `AnalysisOrchestrator` under the hood.
3. **`parse/cleaning.py`:**
   - Clean up dual-handling in `detect()` to treat `EDAResult` as canonical while preserving duck-typing compatibility for existing fixtures.
   - Guaranteed zero changes to detection rules, proposals, or confidence values.
4. **`parse/eda.py` Deprecation:**
   - Add `warnings.warn("parse.eda is deprecated and will be removed in PARSE 2.0. Use parse.analysis.AnalysisOrchestrator instead.", DeprecationWarning, stacklevel=2)` in `DataUnderstanding.__init__`.
   - Keep `parse/eda.py` fully intact with all existing tests running so zero downstream code is broken in this pass.

---

## 4. Contract Test for Explainer Schema

Add `tests/test_explainer_adapter.py`:
- Pins the 7 required fields: `finding_id`, `source`, `category`, `kind`, `message`, `attributes`, `limitations`.
- Asserts schema version is `"1.0"`.
- Asserts legacy ID preservation for planted issues (`duplicate_rows`, `missing_*`, `outliers_*`, `skew_*`).
- Asserts numerical fidelity of output records using `eval.fidelity_checks.numerical_fidelity`.

---

## 5. Verification & Diff Report

Upon approval, implementation will proceed in small commits, followed by:
1. Running `python eval/build_parse_eval_set.py --out eval/parse_eval_set.json`.
2. Generating a detailed Before/After Diff Report comparing finding counts, IDs, and messages between the legacy and canonical outputs.
3. Verifying the full test suite remains green.
