# TASKS.md — PARSE Dataset Analysis / Cleaning Hardening

## How to use this file
This file is written for whichever agent (human or Claude) implements the
next unit of work on `parse/`. Work **one numbered task at a time**, in
order, inside its own section. Do not start a later task before the
current one's Acceptance Criteria are met and tests pass. Do not expand
scope beyond what a task explicitly asks for — if you notice an unrelated
problem, add it to `## Parking Lot` at the bottom instead of fixing it
inline.

Every task below follows the same shape: Objective, Why, Scope (In/Out),
Design, Files, Non-Negotiables (DO NOT), Testing, Acceptance Criteria.

---

## Guiding rules (apply to every task in this file, not just Task 1)

1. **Evidence before action.** Every number the cleaner reports (a
   confidence, a shift, a severity) must be computed from the data in
   front of it, with the computation visible and inspectable — never a
   hardcoded placeholder (`{}`, `[]`, `None`) standing in for "not
   implemented yet." If something can't be computed, say so explicitly
   in `limitations`, don't fake a value.
2. **Human approval is the only thing that changes data.** Nothing in
   this file should introduce a path where `DataCleaner.clean()` applies
   a transformation without the proposal's `proposal_id` being in
   `approved_ids`. Confidence scores, severity, and evidence are
   *decision support for the human reviewer* — never a trigger for
   auto-approval, auto-threshold-apply, or "confidence > X therefore
   apply." If you're tempted to add such a shortcut, don't; flag it in
   Parking Lot instead and ask first.
3. **Explain, don't just score.** A bare float (`confidence: 0.62`) is
   not evidence. Every computed score must ship with the *factors* that
   produced it, in a structure the Streamlit UI (or any future UI) can
   render as "why this number" — not just a number to trust blindly.
4. **Don't break the existing contract.** `CleaningIssue`,
   `TransformationProposal`, `ValidationResult`, `DownstreamImpact` are
   already used by tests and (soon) the UI. Add fields/populate existing
   unused fields; don't rename or remove fields without checking all
   call sites (`grep -rn` first).
5. **Small, tested, reversible commits.** Each task should be
   independently testable and revertable. No task should require a
   later task to already be done in order to pass its own tests.

---

## Task 1 — Populate `TransformationProposal.confidence`
**Status: COMPLETE.** `parse/cleaning_confidence.py` implements all three
scorers; `parse/cleaning.py` populates `confidence` and
`parameters["confidence_basis"]` on both detection paths; approval
gating verified untouched; full suite green (79 passed). Do not reopen
this task — if a bug is found in it later, file it as a new task, don't
edit history here.


### Objective
`TransformationProposal.confidence: float | None` exists in
`parse/cleaning.py` but is **never set** — every proposal currently
ships `confidence=None`. Implement real, explainable confidence scoring
for the three proposal types the cleaner currently generates:
`remove_duplicates`, `impute_missing`, `flag_for_review` (outliers).

### Why
Per the PARSE design philosophy (`PARSE_ARCHITECTURE.md` /
`PARSE_CONTRACTS.md`), automation should scale with how well-evidenced
and reversible an operation is, and the human reviewer needs a
calibrated signal — not just "here's a proposal, trust it" — to decide
what to approve first, defer, or reject. Right now the reviewer has
rationale text but no quantified basis to prioritize review effort
across dozens of proposals on a large dataset.

### Scope
**In scope:**
- A new module `parse/cleaning_confidence.py` containing pure,
  independently-testable scoring functions — one per proposal type.
- Wiring `DataCleaner._detect_with_context()` (and, for parity,
  `_detect_legacy()`) to call these functions and populate
  `TransformationProposal.confidence` and a new
  `TransformationProposal.parameters["confidence_basis"]` dict holding
  the per-factor breakdown.
- Unit tests for each scoring function using constructed datasets with
  known statistical properties (not just "does it return a float").

**Out of scope (do not touch in this task):**
- `DownstreamImpact.distribution_shifts` (Task 2).
- `ValidationResult.newly_introduced_issues` (Task 3).
- MAD-based outlier detection / dual-method agreement (Task 4). Task 1
  uses only the existing IQR-based `DistributionProfile.outlier_count`
  and distance-from-fence, not a new detection method.
- MCAR/MAR missingness-correlation analysis (Task 5). Task 1's
  imputation confidence may *reference* missing_rate but must not
  attempt to detect whether missingness correlates with another column
  — that dependency doesn't exist yet.
- Collapsing `_detect_legacy` vs `_detect_with_context` (Task 6).
- Any new proposal types, or changing what actions the cleaner can take.

### Design

All three scorers return `(confidence: float, basis: dict[str, Any])`
where `confidence` is in `[0, 1]` and `basis` is a JSON-serializable dict
of the named sub-scores and raw inputs that produced it (so a human or
a later audit can see exactly why the number is what it is).

#### 1. `remove_duplicates` → `score_duplicate_removal(...)`
Exact full-row duplicates are a deterministic, unambiguous match —
there's no statistical uncertainty in "these rows are byte-identical."
- `confidence = 1.0` when the match is exact-row (`keep="first"` on all
  columns, which is the only kind currently detected).
- `basis = {"match_type": "exact_row", "columns_compared": <all column names>, "duplicate_row_count": N}`
- **Do not** invent a lower confidence for this case — inventing
  uncertainty where none exists is as dishonest as inventing false
  certainty. If a future task adds *partial-key* duplicate detection
  (e.g. same candidate-identifier, different other values), that
  proposal type needs its own, separate, lower-confidence scorer — not
  a hacked-down version of this one.

#### 2. `impute_missing` → `score_imputation(...)`
Confidence reflects how *representative* the proposed fill value
(median/mode) is likely to be, given what's observable right now.
Composite of four independently-computed sub-scores, weighted average:

```
confidence = 0.30 * s_missing_rate
            + 0.25 * s_sample_size
            + 0.25 * s_dispersion
            + 0.20 * s_placeholder_independence
```

- `s_missing_rate = 1 - missing_rate` (clipped to `[0, 0.95]` — never
  claim full certainty even at 0% missing, since this scorer only runs
  when there IS missingness).
- `s_sample_size = min(1.0, non_null_count / 30)` — the 30-observation
  floor is a conventional rule-of-thumb for a stable median/mode
  estimate, not a proven threshold; **flag this constant clearly in
  code as `MIN_STABLE_SAMPLE_SIZE = 30  # rule-of-thumb, revisit`** so
  it's easy to find and challenge later.
- `s_dispersion`:
  - numeric column: `1 / (1 + CV)` where `CV = std / abs(mean)` on the
    non-null values (guard `mean == 0` → fall back to
    `1 / (1 + std)`; guard `std == 0` → `s_dispersion = 1.0`, a
    constant column is maximally representable by any single value).
  - categorical/boolean column: `1 - normalized_entropy` of the value
    frequencies (0 = perfectly uniform / worst case for "use the mode",
    1 = one dominant value / best case).
- `s_placeholder_independence = 1.0` (fixed) in this task, **with an
  explicit note in `basis`**: `"missingness_pattern_checked": false,
  "note": "MAR/MNAR correlation check not yet implemented — see Task 5"`.
  Do not fabricate a computed value for this factor; a flat 1.0 with an
  honest disclaimer is correct, a fabricated computed-looking value is
  not.
- `basis` must include every sub-score by name plus the raw inputs
  (`missing_rate`, `non_null_count`, `cv` or `normalized_entropy`) used
  to compute them.

#### 3. `flag_for_review` (outliers) → `score_outlier_flag(...)`
This proposal doesn't change data (action is `flag_for_review`, not a
transform), so "confidence" here means *confidence the flagged value is
genuinely statistically unusual*, not confidence in a fill value.
- For each flagged observation, compute how many IQRs beyond the near
  fence it sits: `distance = (value - fence) / iqr` (use whichever
  fence — lower or upper — is relevant).
- `confidence = min(0.95, 0.5 + 0.1 * distance)` per observation; the
  **proposal-level** confidence is the mean across all flagged
  observations for that column, since one proposal currently covers all
  flagged rows in a column together.
- Cap at 0.95, floor at 0.5 (a value just past the fence by definition
  triggered detection, so it's never below "somewhat unusual").
- `basis = {"method": "IQR_1.5", "flagged_count": N, "mean_iqr_distance": ..., "per_row_distances": {...}}`
  — keep `per_row_distances` capped to the first ~20 rows to avoid
  bloating the payload on large datasets; note the cap in the dict if
  applied.
- **Do not** let this scorer decide severity or auto-escalate anything
  — it only produces the confidence number and its basis. Severity
  fields already on `CleaningIssue` are untouched in this task.

### Files
- **New:** `parse/cleaning_confidence.py` — the three scoring functions,
  plus `MIN_STABLE_SAMPLE_SIZE` constant, fully documented with
  docstrings explaining the formula and its limitations (same tone as
  this file — say what's a real statistical basis vs. a rule-of-thumb).
- **Edit:** `parse/cleaning.py` — import the scorers, call them at the
  point each proposal is constructed in `_detect_with_context` (and
  `_detect_legacy` for parity — do not let the two paths diverge in
  what confidence a user sees for the same detected issue), set
  `proposal.confidence` and `proposal.parameters["confidence_basis"]`.
- **New:** `tests/test_cleaning_confidence.py` — unit tests for each
  scorer in isolation (no `DataCleaner` involved), using hand-built
  `pandas.Series`/small DataFrames with known statistical properties
  (e.g. a column with `CV=0` should score `s_dispersion=1.0` exactly).
- **Edit:** `tests/test_cleaning.py` / `tests/test_cleaning_integration.py`
  — add assertions that proposals now carry non-`None` confidence and a
  non-empty `confidence_basis`, without weakening any existing
  assertion.

### Non-Negotiables (DO NOT)
- Do NOT use confidence to gate, filter, or auto-approve anything.
  `clean()`'s approval-gate logic must be untouched by this task.
- Do NOT compute `s_placeholder_independence` from a real correlation
  check — that infra doesn't exist until Task 5. A flat `1.0` with an
  honest disclaimer beats a fabricated computation.
- Do NOT change any existing `CleaningIssue` or proposal `action`/`kind`
  values, or introduce new proposal types.
- Do NOT touch `DownstreamImpact`, `ValidationResult`, or MAD-based
  detection — those are separate tasks.
- Do NOT silently change the `_detect_legacy` code path's *issue
  detection* logic — only add confidence scoring to the proposals it
  already produces.
- Do NOT invent statistical rigor you can't explain in one sentence in
  a docstring. If a formula is a heuristic/rule-of-thumb rather than a
  textbook statistical method, say so explicitly in the docstring and
  in `basis`.

### Testing
- Every scoring function gets direct unit tests with constructed
  inputs where the expected output is known by hand-calculation
  (not just "returns something between 0 and 1").
- At least one test per scorer for an edge case: zero variance,
  100% missing minus one value (smallest valid non-null count), a
  single flagged outlier vs. many.
- Integration test: run `DataCleaner.clean()` end-to-end on a small
  synthetic frame and assert that every generated proposal has
  `confidence is not None` and `parameters["confidence_basis"]` is a
  non-empty dict with the expected keys for its proposal type.
- Full existing suite must stay green — run it before and after.

### Acceptance Criteria
- [ ] `parse/cleaning_confidence.py` exists with the three documented
      scorer functions and the `MIN_STABLE_SAMPLE_SIZE` constant.
- [ ] Every `TransformationProposal` produced by both `_detect_legacy`
      and `_detect_with_context` has a non-`None` `confidence` and a
      populated `parameters["confidence_basis"]`.
- [ ] No change to approval-gating behavior — `clean()` still only
      applies proposals whose id is in `approved_ids`.
- [ ] New unit tests pass; full existing suite still passes; no test
      was weakened or deleted to make this true.
- [ ] `python -m py_compile` (or equivalent) clean on all touched files.

---

## Task 2 — Populate `DownstreamImpact.distribution_shifts` and `potential_information_loss`
**Status: COMPLETE.** `parse/cleaning_impact.py` implements
`compute_ks_statistic`, `compute_distribution_shifts`, and
`derive_information_loss_notes`, all wired into `clean()`; no scipy
dependency added; thresholds (`RELATIVE_STD_SHIFT_THRESHOLD = 0.10`,
`CATEGORICAL_TVD_THRESHOLD = 0.15`) documented as chosen, not derived;
full suite green (91 passed). Do not reopen — file bugs as new tasks.


### Objective
`DownstreamImpact.distribution_shifts` is currently hardcoded to `{}`
and `potential_information_loss` to `[]` in `parse/cleaning.py`'s
`clean()`. Compute both for real by comparing the `original` and
`cleaned` frames after approved changes are applied.

### Why
A reviewer approving proposals needs to know not just "this fixes issue
X" but "here's what actually happened to the data's shape as a result"
— an imputation can quietly shrink variance, a dedup can shift a mean
if the duplicates weren't uniformly distributed. Right now that's
invisible; the field exists and is exported to `to_dict()` but always
says nothing happened.

### Scope
**In scope:**
- A new module `parse/cleaning_impact.py` with a pure function
  `compute_distribution_shifts(original, cleaned, columns) -> dict`
  comparing the two frames column-by-column.
- A pure function `derive_information_loss_notes(distribution_shifts, changes) -> list[str]`
  that turns the computed shifts into plain-language notes using fixed,
  documented thresholds (rule-based, not a new statistical method).
- Wiring both into `DataCleaner.clean()` where `DownstreamImpact` is
  currently constructed with placeholders.
- Unit tests with hand-built before/after frames where the expected
  shift numbers are known.

**Out of scope:**
- `ValidationResult.newly_introduced_issues` (Task 3) — do not touch it
  even though it's constructed in the same method.
- Any new outlier-detection method (Task 4).
- Any change to what proposals exist or how they're approved.
- Adding scipy as a dependency — see Design note below; if you disagree
  with the no-scipy call, raise it, don't silently add the import.

### Design

`compute_distribution_shifts(original, cleaned, columns)` runs over the
**intersection of columns present in both frames**, restricted to
`columns` passed in (the caller passes all dataset columns, not just
`changed_attributes`, because removing duplicate rows can shift the
distribution of columns that were never individually edited — see
Non-Negotiables).

For each column, branch on dtype:

- **Numeric columns:**
  ```
  {
    "kind": "numeric",
    "mean_before": ..., "mean_after": ..., "mean_shift": after - before,
    "std_before": ..., "std_after": ..., "std_shift": after - before,
    "sample_size_before": N, "sample_size_after": N,
    "ks_statistic": max |ECDF_before(x) - ECDF_after(x)| over sorted union of values,
    "ks_statistic_note": "Two-sample KS statistic computed without a
       p-value (no scipy dependency). Use as a relative shift
       indicator, not a hypothesis-test result.",
  }
  ```
  Compute the KS statistic manually: sort both samples, walk the
  combined sorted values, track each sample's empirical CDF at each
  point, take the max absolute difference. Guard empty-column and
  single-unique-value cases (statistic = 0.0 with a note, not a
  divide-by-zero).

- **Categorical / boolean columns:**
  ```
  {
    "kind": "categorical",
    "total_variation_distance": 0.5 * sum(|p_after(c) - p_before(c)|) over the union of categories,
    "categories_added": [...], "categories_removed": [...],
    "sample_size_before": N, "sample_size_after": N,
  }
  ```

- **Everything else (text/id/unknown):**
  ```
  {"kind": "not_computed", "reason": "<why — e.g. free-text column, comparison not meaningful>"}
  ```
  Do not force a number out of a column type where one isn't
  meaningful — an honest `not_computed` beats a misleading statistic.

- **No-op case:** if `original` and `cleaned` are identical for a
  column (no rows removed, no values changed), still include an entry
  with all shift values `0` / `"unchanged": true` rather than omitting
  the column — omission reads as "not checked," zero reads as "checked,
  nothing happened," which is what actually occurred.

`derive_information_loss_notes(distribution_shifts, changes)` — fixed,
documented thresholds, applied only to columns that appear in `changes`
(i.e. were actually touched by an approved proposal):
- Numeric column where `abs(std_shift) / std_before > 0.10` (10%
  relative variance change) → note:
  `"'{column}': standard deviation changed by {pct}% after cleaning — check whether this reflects real signal removal (e.g. imputation reducing spread) rather than noise removal."`
- Numeric column where `sample_size_after < sample_size_before` and the
  column had no missing values before (i.e. the shift is purely from
  row removal, not imputation) → note about row-removal-driven shift.
- Categorical column where `total_variation_distance > 0.15` → note
  about category-frequency shift.
- **Document these thresholds (10%, 0.15) as named constants at the top
  of `cleaning_impact.py`, each with a one-line "this is a chosen
  threshold, not derived" comment** — same transparency standard as
  `MIN_STABLE_SAMPLE_SIZE` in Task 1.
- If nothing crosses a threshold, return `[]` legitimately — that's a
  real "no notable loss detected," not a placeholder.

### Files
- **New:** `parse/cleaning_impact.py`
- **Edit:** `parse/cleaning.py` — replace the hardcoded
  `distribution_shifts={}` and `potential_information_loss=[]` in the
  `DownstreamImpact(...)` construction inside `clean()` with real calls,
  passing `original`, `cleaned`, and all dataset columns.
- **New:** `tests/test_cleaning_impact.py`
- **Edit:** `tests/test_cleaning_integration.py` — add an assertion that
  after an imputation is approved and applied, `distribution_shifts`
  contains a non-placeholder entry for the imputed column with the
  expected shift direction (not exact float match unless you construct
  the fixture precisely enough to hand-calculate it — prefer a fixture
  precise enough to hand-calculate).

### Non-Negotiables (DO NOT)
- Do NOT restrict the shift computation to only `changed_attributes`
  (columns with a direct cell-level `ChangeRecord`). Duplicate-row
  removal has `field=None` on its `ChangeRecord`s but still shifts
  every column's distribution by removing rows — compute over all
  dataset columns, not just literally-edited ones.
- Do NOT add scipy or any new external dependency without flagging it
  first — implement KS statistic manually per Design.
- Do NOT let `derive_information_loss_notes` do anything but read
  already-computed `distribution_shifts` and `changes` — no new
  statistics invented inside it, thresholds only.
- Do NOT touch `ValidationResult` or its `newly_introduced_issues` field
  even though you'll be looking at the same `clean()` method — that's
  Task 3, separately scoped because it requires re-running detection,
  a different kind of operation than a before/after comparison.
- Do NOT change `DownstreamImpact`'s field names/shape — only populate
  what's already declared, plus whatever nested dict shape you need
  *inside* `distribution_shifts` (that part's schema is yours to define
  per this spec, since it was never specified before).

### Testing
- Hand-built before/after frame pairs where mean/std/KS-statistic are
  known by calculation, not just "returns a float."
- Explicit test for the `not_computed` branch (a free-text column).
- Explicit test for the no-op / unchanged-column case.
- Explicit test that `derive_information_loss_notes` returns `[]` when
  no threshold is crossed, and returns the expected note text when one
  is deliberately crossed via a constructed fixture.
- Full suite stays green.

### Acceptance Criteria
- [ ] `parse/cleaning_impact.py` exists with both functions, named
      threshold constants documented as chosen-not-derived.
- [ ] `clean()` no longer hardcodes `distribution_shifts={}` or
      `potential_information_loss=[]`; both are computed from the real
      before/after frames.
- [ ] Distribution shifts are computed over all dataset columns, not
      just cell-level `changed_attributes`.
- [ ] No scipy dependency added.
- [ ] New tests pass with hand-calculated expected values; full
      existing suite still green.

---

## Task 3 — Populate `ValidationResult.newly_introduced_issues`
**Status: COMPLETE.** Full-coverage re-detection implemented: fresh
`AnalysisOrchestrator` run on `cleaned`, outliers and structural
candidate-identifier conflicts included, diffed against `original`
issues by `(kind, field)`. Graceful degradation on empty frame /
exception verified. One implementation note worth recording: while
extending `_detect_legacy`'s outlier extraction, it turned out the
modern `AnalysisOrchestrator` emits findings under an `"unusual:{column}"`
key rather than the legacy `"outliers_{column}"` key — `_detect_legacy`
was extended to recognize both, with a defensive `getattr` for
`.message`/`.observation` since the two finding shapes don't guarantee
the same attribute name. This wasn't in the original spec but was
necessary for the fresh-EDA path to actually surface outliers at all —
correct call, in scope of "make re-detection real." Full suite green
(94 passed). Do not reopen — file bugs as new tasks.


### Objective
`ValidationResult.newly_introduced_issues` is currently hardcoded to
`[]` in `clean()`. After approved changes are applied, re-detect issues
on the `cleaned` frame and report any that weren't present in the
`original` frame's detection — e.g. an imputation that happens to
create a new duplicate row, or a dedup that leaves a column newly
all-null for a subset.

### Why
`ValidationResult` exists specifically to answer "did the fix work
without breaking something else," and right now it always silently says
"nothing new went wrong" regardless of what actually happened. That's
the same category of dishonest placeholder Task 1 and 2 fixed for
confidence and distribution shifts.

### Scope
**In scope:**
- Re-running full-coverage issue detection on `cleaned` after changes
  are applied, and diffing against the issues detected on `original` —
  including outliers and structural (candidate-identifier) issues, not
  just duplicates/missing/mixed-type.
- Populating `newly_introduced_issues` with a real, non-empty-when-true
  list.

**Out of scope — read this carefully, it's the main design constraint:**
- **Do NOT rebuild the full `CleaningContext` (semantic layer +
  `HumanContext`) for the cleaned frame.** Semantic candidates and
  human confirmations are about *meaning*, not about whether a new
  statistical/structural problem appeared — they're irrelevant to this
  task and rebuilding them would mean re-running semantic analysis
  inside `clean()`, a heavier operation this task doesn't need.
- **Do run a fresh `AnalysisOrchestrator().analyze(...)` on `cleaned`.**
  This is the same deterministic, dependency-light EDA step used
  everywhere else in `parse/` — it's not the thing we're avoiding.
  Running it gives a fresh `EDAResult` with distribution outliers and
  `structural_profile.candidate_index_columns` for the cleaned frame,
  which is exactly what's needed for full-coverage re-detection.
- Feed that fresh `EDAResult` into `self.detect(cleaned, eda_result=fresh_eda)`
  (the `_detect_legacy` path already branches on `eda_result is not None`
  to add outlier issues — reuse that, don't duplicate its logic).
- Additionally reconcile candidate-identifier/structural issues using
  the same `candidate_index_columns` check `_detect_with_context`
  already uses, applied here to the fresh `EDAResult` — see Design.
- This means the only thing this task does **not** cover is anything
  that depends on `HumanContext`/semantic confirmation state (there is
  no such issue category today, so this is a theoretical gap, not a
  practical one) — state that narrower, honest limitation on
  `ValidationResult` rather than the broader one originally drafted.

### Design

1. In `clean()`, after building `cleaned`, run:
   ```
   fresh_source = SourceRef(self.source.source_id, self.source.source_type, label=f"{self.source.label} (post-cleaning)")
   fresh_eda = AnalysisOrchestrator().analyze(AnalysisRequest(cleaned, fresh_source))
   ```
   (import `AnalysisOrchestrator`/`AnalysisRequest` from `parse.analysis`,
   `SourceRef` already imported).
2. Call `cleaned_issues, _ = self.detect(cleaned, eda_result=fresh_eda)`
   — this reuses `_detect_legacy`'s existing duplicate/missing/mixed/
   outlier logic against the cleaned frame with real outlier coverage.
3. Separately reconcile structural/candidate-identifier issues: for
   each `key in fresh_eda.structural_profile.candidate_index_columns`,
   check for duplicated values under that key in `cleaned` the same way
   `_detect_with_context` does for `original`, and append any resulting
   `CleaningIssue`s to `cleaned_issues` before diffing. (If this check
   only ran in `_detect_with_context` previously and `context` is
   `None` in this call to `clean()`, this task is what adds it to the
   post-cleaning check regardless of whether `context` was used
   up-front — this closes the exact gap you're worried about.)
4. Key both `original` and `cleaned` issue sets by `(kind, field)`:
   `original_keys = {(issue.kind, issue.field) for issue in issues}`
   (reuse the `issues` list already computed earlier in `clean()` — do
   not re-detect `original` a second time)
   `cleaned_keys = {(issue.kind, issue.field) for issue in cleaned_issues}`
   `new_keys = cleaned_keys - original_keys`
5. For each key in `new_keys`, produce a human-readable string using the
   matching `CleaningIssue.message` from `cleaned_issues`, e.g.:
   `"New issue after cleaning — missing_values in 'column_x': 3 value(s) are missing (not present before cleaning)."`
6. If the fresh `AnalysisOrchestrator` call or re-detection raises, or
   `cleaned` is empty, degrade gracefully: return `[]` for
   `newly_introduced_issues` and add a note to
   `ValidationResult.limitations` explaining re-detection was skipped
   and why — do not let this crash `clean()`.
7. `ValidationResult.limitations` should always include one standing
   note when re-detection succeeds:
   `"Re-detection covers duplicates, missingness, mixed-type values, statistical outliers, and candidate-identifier conflicts on the cleaned frame; it does not re-run semantic candidate generation or human confirmation state, which are unaffected by transformation actions."`
   — this is now a narrow, accurate statement, not a broad admitted gap.

### Files
- **Edit:** `parse/cleaning.py`:
  - Add `limitations: list[str] = dataclass_field(default_factory=list)`
    to `ValidationResult`.
  - Import `AnalysisOrchestrator`, `AnalysisRequest` from `parse.analysis`
    at module level (check for circular imports first — `parse.analysis`
    does not import `parse.cleaning`, so this should be safe; run
    `python -c "import parse.cleaning"` after adding the import to
    confirm).
  - In `clean()`, after building `cleaned`, run the fresh EDA + detect
    + structural reconciliation from Design, diff against `issues`,
    and populate `ValidationResult(..., newly_introduced_issues=[...], limitations=[...])`
    instead of the hardcoded `[]`.
- **Edit:** `tests/test_cleaning.py` — unit test that constructs a
  scenario where an approved change introduces a detectable new
  *outlier* or *duplicate* issue (not just missing/mixed, since those
  were already reachable before this fix — an outlier case is the one
  that actually proves the gap is closed) and asserts it shows up in
  `newly_introduced_issues`.
- **Edit:** `tests/test_cleaning_integration.py` — assert the "nothing
  new went wrong" case still correctly returns `[]` on a scenario where
  it genuinely shouldn't detect anything new, and assert
  `ValidationResult.limitations` contains the standing note whenever
  re-detection ran successfully.

### Non-Negotiables (DO NOT)
- Do NOT rebuild `CleaningContext`, semantic analysis, or `HumanContext`
  inside `clean()` — only `AnalysisOrchestrator` (EDA) on `cleaned`.
- Do NOT skip the structural/candidate-identifier reconciliation step —
  that was the specific gap flagged; it must be covered, not just
  outliers.
- Do NOT let the fresh EDA call or re-detection failure raise out of
  `clean()` — degrade to an empty list plus a limitation note.
- Do NOT change how `intended_issues_addressed` or `comparison` are
  computed — those are already correct and out of scope here.
- Do NOT let this task quietly get more expensive than it needs to be —
  `AnalysisOrchestrator().analyze()` runs once, on `cleaned`, not in a
  loop, not per-column manually re-invoked.

### Testing
- Constructed fixture that provably introduces a new **outlier** issue
  after an approved change (e.g. an imputation that pulls a value far
  from the rest of the column's distribution), proving the previously
  gapped detection path now works.
- Constructed fixture that provably introduces a new **duplicate** or
  **candidate-identifier conflict** issue after cleaning.
- Constructed fixture that provably introduces *no* new issue, asserting
  an empty list.
- Confirm `ValidationResult.limitations` always contains the narrow
  standing note when re-detection ran successfully, and the
  skip-reason note when it didn't.
- Full suite stays green.

### Acceptance Criteria
- [ ] `ValidationResult` has a new `limitations: list[str]` field,
      additive and defaulted, not breaking existing constructors.
- [ ] `newly_introduced_issues` is computed from a full-coverage diff:
      duplicates, missingness, mixed-type, **outliers, and structural/
      candidate-identifier conflicts** — not just the three original
      `_detect_legacy` categories.
- [ ] No semantic-layer or `HumanContext` re-run inside `clean()`.
- [ ] Re-detection failure degrades gracefully, never raises.
- [ ] New tests pass with a hand-verified outlier case, a
      duplicate/structural case, and a negative case; full existing
      suite still green.

---

## Parking Lot
*(Anything noticed while working that's out of scope for the current
task goes here, not into the current task's diff.)*

-

---

## Task 4 — Add MAD-based outlier detection alongside IQR and dual-method agreement
**Status: COMPLETE.** `compute_mad_outliers` implements MAD calculation with
asymptotic normal scale (1.4826) and thresholding (3.0); `score_outlier_flag`
applies a +0.05 consensus boost when both IQR and MAD agree; backwards
compatibility preserved with `method="IQR_1.5"` and `dual_method="dual_IQR_MAD"`;
`DistributionProfile.mad_outlier_count` added and populated in `_distribution`;
full suite green (100 passed). Do not reopen — file bugs as new tasks.

### Objective
Currently, outlier detection in `parse/` relies solely on Tukey's IQR rule
(`[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`), and `score_outlier_flag` computes confidence
purely based on distance beyond the IQR fences. Implement Median Absolute
Deviation (MAD)-based outlier detection alongside IQR, compute dual-method
agreement for flagged observations, and use method agreement to calibrate
and strengthen `score_outlier_flag` without replacing the IQR foundation.

### Why
IQR and MAD have complementary statistical properties:
- IQR spans the central 50% of data (between the 25th and 75th percentiles).
  In discrete, repeated, or zero-inflated distributions, IQR can collapse to 0
  or become artificially narrow, producing false positives or zero-fence breakdowns.
- MAD measures the median distance to the sample median across the entire
  dataset with a 50% breakdown point, scaled by 1.4826 for asymptotic normal
  consistency: `MAD_scaled = 1.4826 * median(|x_i - median(x)|)`.
- Dual-method agreement provides strong empirical consensus: when both IQR and
  MAD agree that an observation is an outlier, confidence in the flag should be
  higher. When methods diverge, confidence should reflect the disagreement
  without auto-filtering or deleting data.

### Scope
**In scope:**
- Pure helper function `compute_mad_outliers(series, threshold=3.0, normal_scale=1.4826)`
  in `parse/cleaning_confidence.py` computing median, raw MAD, scaled MAD, and
  per-row modified Z-scores with zero-MAD guards.
- Updating `score_outlier_flag(...)` in `parse/cleaning_confidence.py` to evaluate
  both IQR fences and MAD thresholds, applying a consensus boost (`+0.05`, capped
  at 0.95) when both methods flag an observation.
- Enriching `basis` with dual-method statistics: `"method": "dual_IQR_MAD"`,
  `"mad"`, `"scaled_mad"`, `"median"`, `"mad_flagged_count"`, `"agreement_count"`,
  `"agreement_rate"`, preserving existing `"mean_iqr_distance"` and
  `"per_row_distances"` for backwards compatibility.
- Exposing `mad_outlier_count: int = 0` on `DistributionProfile` in `parse/analysis.py`.
- Unit and integration tests covering consensus, disagreement, zero-variance/zero-MAD
  edge cases, and cleaner integration.

**Out of scope:**
- Changing proposal action: `flag_for_review` remains informational; no automatic
  row deletion or filtering.
- MCAR/MAR missingness analysis (Task 5).
- Collapsing legacy detection (Task 6).

### Design
1. **MAD computation:**
   - `raw_mad = median(|x - median(x)|)`
   - `scaled_mad = 1.4826 * raw_mad` (guard: if `raw_mad == 0`, no MAD outliers can be computed)
   - Modified Z-score per observation: `z_mad = |x - median| / scaled_mad`
   - Observation is flagged by MAD if `z_mad > 3.0` (`MAD_OUTLIER_THRESHOLD`).
2. **Confidence with dual-method agreement:**
   - For each observation flagged by IQR:
     - `base_conf = min(0.95, max(0.5, 0.5 + 0.1 * iqr_distance))`
     - If also flagged by MAD (`z_mad > 3.0`): `obs_conf = min(0.95, base_conf + 0.05)`
     - If only flagged by IQR: `obs_conf = base_conf`
   - Proposal confidence is the mean across flagged observations (floored at 0.5, capped at 0.95).
3. **Inspectable basis:**
   - `"method": "dual_IQR_MAD"`, `"mad": ...`, `"scaled_mad": ...`, `"median": ...`
   - `"iqr_flagged_count": ...`, `"mad_flagged_count": ...`, `"agreement_count": ...`, `"agreement_rate": ...`
   - `"per_row_distances"`: `{row_id: iqr_dist, ...}` (capped to 20)
   - `"per_row_mad_distances"`: `{row_id: mad_dist, ...}` (capped to 20)
   - `"per_row_agreement"`: `{row_id: bool, ...}` (capped to 20)

### Files
- **Edit:** `parse/cleaning_confidence.py` — add constants `NORMAL_SCALE_MAD = 1.4826`,
  `MAD_OUTLIER_THRESHOLD = 3.0`, `DUAL_AGREEMENT_BOOST = 0.05`; add `compute_mad_outliers`;
  update `score_outlier_flag`.
- **Edit:** `parse/analysis.py` — add `mad_outlier_count: int = 0` to `DistributionProfile`;
  populate in `_distribution`.
- **Edit:** `tests/test_cleaning_confidence.py` — add unit tests for `compute_mad_outliers`
  and dual-method agreement in `score_outlier_flag`.
- **Edit:** `tests/test_cleaning.py` — integration tests verifying dual-method metadata in proposals.

### Non-Negotiables (DO NOT)
- Do NOT auto-delete or auto-filter outlier rows. Proposal remains `flag_for_review`.
- Do NOT break existing `score_outlier_flag` call sites or existing basis keys (`mean_iqr_distance`,
  `per_row_distances` must be preserved).
- Do NOT divide by zero when MAD is 0 (e.g. constant columns or >50% duplicate values).
- Do NOT use scipy — compute MAD and median purely using pandas/numpy.

### Testing
- Direct unit tests for `compute_mad_outliers` with hand-calculated inputs.
- Test dual-method agreement: extreme outlier flagged by both IQR and MAD receives the +0.05 boost.
- Test divergence: mild outlier flagged by IQR but not MAD receives base confidence without boost.
- Test zero-MAD edge case: >50% identical values where MAD=0 degrades gracefully.
- Full test suite remains green.

### Acceptance Criteria
- [x] `compute_mad_outliers` exists with documented named constants (`NORMAL_SCALE_MAD = 1.4826`,
      `MAD_OUTLIER_THRESHOLD = 3.0`).
- [x] `score_outlier_flag` computes dual-method agreement and applies consensus boost.
- [x] `DistributionProfile` has `mad_outlier_count: int = 0`, populated in `_distribution`.
- [x] Existing contract and basis fields preserved.
- [x] No scipy dependency added.
- [x] Full existing test suite plus new tests green.

---

## Parking Lot
*(Anything noticed while working that's out of scope for the current
task goes here, not into the current task's diff.)*

-

---

## Upcoming (not started — for context only, do not work on these yet)

- **Task 5** — Add a missingness-independence check (does missingness
  in column A correlate with values in column B) and use it to compute
  a real `s_placeholder_independence` in the Task 1 imputation scorer.
- **Task 6** — Collapse `_detect_legacy` into `_detect_with_context`
  once all call sites pass a `CleaningContext`, and delete the legacy
  path.

