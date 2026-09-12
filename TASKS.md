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

---

## Task 5 — Real missingness-independence check for imputation confidence
**Status: COMPLETE.** Cross-column missingness association implemented via
`compute_missingness_association()` using manual Cohen's d (pooled SD, capped at
`NUMERIC_EFFECT_SIZE_CAP = 3.0`) for numeric columns and manual 2xk contingency table
$\chi^2$ / Cramér's V for categorical/boolean columns; `MIN_GROUP_SIZE_FOR_ASSOCIATION = 5`
threshold guards against small-sample noise; `score_imputation(series, frame=None, column_name=None)`
preserves backward compatibility while computing real `s_placeholder_independence = 1.0 - max_association`
and `missingness_pattern_checked: True` when DataFrame context is provided; both call sites
in `parse/cleaning.py` updated; full suite green (106 passed). Do not reopen — file bugs as new tasks.

> **Verified against the actual pushed code (commit `1721a4c`) before
> writing this spec** — the note from the previous draft is resolved,
> replaced with the finding below.
>
> **Signature blocker found and designed around:** `score_imputation()`
> currently takes only `series: pd.Series` — it has no access to the
> rest of the DataFrame, so it *cannot* compute a cross-column
> association as originally drafted without a signature change. Both
> call sites already have the full `frame` and the column name in scope
> (`parse/cleaning.py` lines ~252 and ~361: `score_imputation(series)`
> inside `_detect_legacy`, where `frame` is the enclosing method's
> parameter and the loop variable is `column`; and
> `score_imputation(frame[attr_name])` inside `_detect_with_context`,
> where `frame` and `attr_name` are both already available). This task
> now explicitly includes widening the signature — see Design step 0.

### Objective
`score_imputation()`'s `s_placeholder_independence` factor is currently
fixed at `1.0` with a disclaimer (`"missingness_pattern_checked": false`).
Replace it with a real check: does missingness in the target column
correlate with values in any other column? If so, imputing with an
unconditional median/mode is less trustworthy (the missingness may be
MAR/MNAR, not MCAR), and confidence should reflect that.

### Why
This is the last placeholder-disguised-as-a-real-factor left from
Task 1's imputation scorer. Per the same principle as every prior task:
an honest flat value with a disclaimer was correct *until* the real
check was feasible — it's feasible now, so it should be built, not left
permanently flat.

### Scope
**In scope:**
- A function that computes an association strength between "is this
  value missing in the target column" and "what is the value in every
  other column," for both numeric and categorical other-columns,
  without scipy (same dependency-light standard as every prior task).
- Wiring the result into `score_imputation()`'s
  `s_placeholder_independence` factor, replacing the flat `1.0`.
- Updating `basis["missingness_pattern_checked"]` to `true` with the
  real association evidence attached.

**Out of scope:**
- Do NOT use this check to change *what* fill value is proposed
  (still median/mode) — only the confidence score. Proposing a
  conditional/group-wise imputation (e.g. "impute median within groups
  of column B") is a materially different, higher-risk feature that
  needs its own task and its own human-approval framing — don't fold it
  in here even though the data to support it now exists.
- Do NOT run this check for every column pair in the dataset up front —
  only for the specific target column being scored, at the point
  `score_imputation()` is called for it.
- Do NOT touch outlier scoring, duplicate scoring, or Task 4's MAD work.

### Design

**Step 0 — widen `score_imputation()`'s signature.** Change it to:
```python
def score_imputation(
    series: pd.Series,
    frame: pd.DataFrame | None = None,
    column_name: str | None = None,
) -> tuple[float, dict[str, Any]]:
```
`frame`/`column_name` are optional and default to `None` so any other
caller (tests included) that only has a bare `Series` still works —
when either is `None`, skip the association check and fall back to the
current honest-disclaimer behavior (`s_placeholder_independence = 1.0`,
`"missingness_pattern_checked": False`) rather than erroring. This
keeps the change backward-compatible rather than forcing every call
site and every existing test to pass two new arguments.

Then update **both** call sites to pass the extra arguments:
- `parse/cleaning.py` line ~252 (inside `_detect_legacy`):
  `score_imputation(series)` → `score_imputation(series, frame=frame, column_name=str(column))`
- `parse/cleaning.py` line ~361 (inside `_detect_with_context`):
  `score_imputation(frame[attr_name])` → `score_imputation(frame[attr_name], frame=frame, column_name=attr_name)`

New function, e.g. `compute_missingness_association(frame, target_column) -> tuple[float, dict]`:

1. Build the missingness indicator: `is_missing = frame[target_column].isna()`.
2. For every other column `other` in `frame`:
   - Skip if `other` has fewer than **5 non-null observations in each
     of the missing/non-missing groups** (name this constant
     `MIN_GROUP_SIZE_FOR_ASSOCIATION = 5`, documented as a rule-of-thumb
     like `MIN_STABLE_SAMPLE_SIZE` in Task 1 — too small a group makes
     any association estimate noise, not signal).
   - **If `other` is numeric:** compute a bounded effect size between
     the two groups (values of `other` where `is_missing` vs. where
     not): `effect = abs(mean_missing_group - mean_nonmissing_group) / pooled_std`
     where `pooled_std` is the pooled standard deviation of the two
     groups (guard `pooled_std == 0` → `effect = 0.0` if means are also
     equal, else treat as maximal separation capped at the same bound
     below). Map to `[0, 1]` via `min(1.0, effect / 3.0)` — an effect
     size of 3 pooled-SDs of mean separation is treated as "as
     associated as this bound cares to distinguish further." Document
     the `3.0` divisor as a chosen cap, not a derived constant.
   - **If `other` is categorical/boolean:** build a 2×k contingency
     table (`is_missing` × categories of `other`), compute the
     chi-square statistic manually (`sum((observed - expected)^2 / expected)`
     over all cells, with `expected = row_total * col_total / grand_total`,
     skip cells where `expected == 0`), then Cramér's V:
     `V = sqrt(chi2 / (n * min(rows-1, cols-1)))` = `sqrt(chi2 / n)` since
     rows=2 always makes `min(rows-1, cols-1) = min(1, cols-1)`, which is
     `1` whenever `cols >= 2` — so this simplifies to `V = sqrt(chi2 / n)`,
     already bounded `[0, 1]` by construction, no extra capping needed
     (verify this bound holds in the unit tests rather than trusting the
     derivation blindly).
   - Skip `other == target_column` obviously, and skip any column that
     is itself the same missingness pattern trivially (not expected in
     practice, but guard divide-by-zero if `is_missing` is constant
     across the whole frame — shouldn't happen since `score_imputation`
     only runs when there's missingness, but a fully-missing column
     would make `is_missing` constant `True`, so guard `is_missing.nunique() < 2` → skip the whole check, return `(1.0, {"skipped": "target column has no non-missing values to compare against"})`).
3. Take `max_association = max(all computed associations)` across all
   compared columns (0.0 if no column qualified for comparison — no
   evidence of association is not the same as evidence of independence,
   but per the same "honest default" principle, absence of a detectable
   signal defaults toward *not* penalizing confidence here, since a
   false "possible bias" flag with no basis is its own kind of
   dishonesty — document this choice in the docstring).
4. `s_placeholder_independence = 1 - max_association`, clipped to `[0, 1]`.
5. `basis` must include: `"missingness_pattern_checked": true`,
   `"max_association": max_association`, `"associated_column": <name of the column that produced max_association, or null>`,
   `"association_method": "effect_size" | "cramers_v" | null`,
   `"columns_compared": N`, `"columns_skipped_insufficient_data": M`.

### Files
- **Edit:** `parse/cleaning_confidence.py` — widen `score_imputation()`'s
  signature per Design Step 0, add `MIN_GROUP_SIZE_FOR_ASSOCIATION = 5`
  and the effect-size cap constant (name it, e.g.
  `NUMERIC_EFFECT_SIZE_CAP = 3.0`), add
  `compute_missingness_association(...)`, wire it into
  `score_imputation()` replacing the flat `1.0` block when `frame`/
  `column_name` are provided.
- **Edit:** `parse/cleaning.py` — update both call sites (lines ~252
  and ~361, confirmed above) to pass `frame=` and `column_name=`.
- **Edit:** `tests/test_cleaning_confidence.py` — unit tests for
  `compute_missingness_association` with constructed data: a case with
  a strong numeric association (known effect size), a strong
  categorical association (known Cramér's V via hand-built contingency
  table), a case with no association (both should report low/near-zero
  and `s_placeholder_independence` near `1.0`), and the
  insufficient-data-skip case.
- **Edit:** `tests/test_cleaning.py` / `tests/test_cleaning_integration.py`
  — update any existing assertion that checked for the old
  `"missingness_pattern_checked": false` placeholder text (search for
  it — it will now be `true` with real values attached) so tests
  reflect the real behavior, not the old placeholder.

### Non-Negotiables (DO NOT)
- Do NOT make `frame`/`column_name` required — they must stay optional
  keyword arguments defaulting to `None`, falling back to the current
  disclaimer behavior when absent, so existing tests that call
  `score_imputation(series)` with one argument keep working unless you
  deliberately choose to update them (and if you do update them, that's
  fine — just don't let a missed call site silently break).
- Do NOT add scipy — chi-square and effect size are computed manually,
  per Design.
- Do NOT let this check change the imputation *action* (still
  median/mode) — confidence only.
- Do NOT run the association check across all columns for all
  imputation proposals eagerly at detection time if that would be
  expensive on wide datasets — it's fine to run per-proposal at
  scoring time (this is the existing call pattern), just don't add a
  separate all-pairs precomputation pass.
- Do NOT silently keep the old `1.0` fallback for cases you didn't
  anticipate — if a column type doesn't fit numeric or
  categorical/boolean cleanly (e.g. datetime, free text), treat it like
  Task 2's `not_computed` branch: skip it explicitly, don't force a
  number, and don't count it as "no association" if you didn't actually
  check it (it should reduce `columns_compared`, not silently inflate
  confidence).

### Testing
- Hand-built numeric case: two groups with a known, computable effect
  size (e.g. group means 10 apart, known pooled std) → assert the
  computed effect and the mapped `[0,1]` value match by calculation.
- Hand-built categorical case: a contingency table with a known
  chi-square value computed by hand → assert Cramér's V matches.
- Null/no-association case: missingness indicator uncorrelated with
  every other column (e.g. random assignment in a fixture) → assert
  `max_association` is low and `s_placeholder_independence` is close to
  `1.0`.
- Insufficient-data case: a column where one group has fewer than
  `MIN_GROUP_SIZE_FOR_ASSOCIATION` observations → assert it's skipped
  and reflected in `columns_skipped_insufficient_data`, not silently
  included.
- Fully-missing-column guard case.
- Full suite stays green.

### Acceptance Criteria
- [x] `compute_missingness_association()` exists, documented, with both
      named constants explained as chosen bounds/thresholds.
- [x] `score_imputation()` accepts optional `frame`/`column_name` and
      both call sites in `parse/cleaning.py` pass them.
- [x] `score_imputation()`'s `s_placeholder_independence` is real (not
      a flat `1.0`) whenever `frame`/`column_name` are provided, and
      `basis["missingness_pattern_checked"]` is `true` in that case;
      falls back to the honest `1.0`/`false` disclaimer when they're
      not provided.
- [x] No scipy dependency added.
- [x] Imputation *action* (fill value proposed) is unchanged — only the
      confidence score and its basis differ.
- [x] New tests pass with hand-calculated expected values; full
      existing suite still green; any test that previously asserted the
      old `false`/`1.0` placeholder is updated to assert real behavior.

---

## Task 6 — Collapse `_detect_legacy` and `_detect_with_context` into one path
**Status: COMPLETE.** Consolidated `_detect_legacy` and `_detect_with_context`
into a single unified `_detect(frame, eda_result=None, context=None)` method.
Resolved all 4 discrepancies per specification:
1. Candidate-identifier conflict detection runs whenever `structural_profile` is
   reachable from `eda_result` or `context.eda_result`.
2. Purpose-gating for imputation is preserved with zero behavior change for existing
   callers (imputation is proposed unless `purpose == "unknown"` explicitly).
3. Mixed-value detection bug fixed for `CleaningContext` callers by porting the
   `numeric_fraction` heuristic and deduplicating against `quality_issues`.
4. Outlier detection scans `resolved_eda.findings` for all matching findings per column.
`_detect_legacy` and `_detect_with_context` deleted; full test suite green (110 passed).
Do not reopen — file bugs as new tasks.

### Objective
`DataCleaner` currently has two independent detection methods that
`detect()` branches between based on whether a `CleaningContext` is
supplied. Five tasks' worth of confidence/impact/re-detection logic has
now been layered onto both in parallel. Collapse them into a single
detection method with no duplicated logic and no divergent behavior
that isn't deliberate and documented.

### Why — audit findings first (read before writing any code)
**Discrepancy 1 — candidate-identifier conflict detection exists only
in `_detect_with_context`.**
> **Resolution:** Rewired to run whenever `structural_profile` is reachable
> (either via `eda_result.structural_profile` or `context.eda_result.structural_profile`).

**Discrepancy 2 — imputation proposals are purpose-gated in `_detect_with_context`.**
> **Resolution:** Propose whenever non-null values exist, unless `resolved_purpose == "unknown"`.
> Zero behavior change for context-less callers.

**Discrepancy 3 — mixed-value detection is effectively dead code in `_detect_with_context`.**
> **Resolution:** Ported the `numeric_fraction` heuristic into the unified method
> and deduplicated against `quality_issues`.

**Discrepancy 4 — outlier detection reads two different sources.**
> **Resolution:** Unified on iterating all matching findings per column from `resolved_eda.findings`.

### Acceptance Criteria
- [x] Exactly one detection method exists in `DataCleaner`.
- [x] Discrepancy 1 resolved: candidate-identifier detection runs
      whenever any `eda_result` is reachable, in either entry mode.
- [x] Discrepancy 2 resolved with zero behavior change for existing
      callers (context-less callers still always get imputation
      proposals; explicit `purpose == "unknown"` still suppresses them).
- [x] Discrepancy 3 resolved: mixed-value detection now works
      correctly for `CleaningContext`-based callers too.
- [x] Discrepancy 4 resolved: outlier detection captures all matching
      findings per column in both entry modes.
- [x] `_detect_legacy` and `_detect_with_context` no longer exist.
- [x] Full suite green; new regression tests for all four discrepancies
      pass.

---

## Task 7 — Pandas 3.x-Safe Dtype Classification
**Status: COMPLETE.** Created `parse/dtype_utils.py` with exclusion-based
classification primitives (`is_numeric`, `is_boolean`, `is_datetime`,
`is_categorical_dtype_`, `is_text_like`, `is_empty_or_all_missing`).
Replaced fragile `dtype == object` checks at all 5 confirmed call sites:
- `parse/eda.py:283`
- `parse/eda.py:324`
- `parse/cleaning.py:321`
- `parse/cleaning_confidence.py:138`
- `parse/cleaning_impact.py:111`
`parse/analysis.py` untouched; unit tests in `tests/test_dtype_utils.py`
confirm `object` and nullable `string` dtypes classify identically;
full suite green (116 passed, 0 failures). Do not reopen — file bugs as new tasks.

### Objective
Replace every fragile `dtype == object` / `pd.api.types.is_object_dtype(...)`
check in the analysis/cleaning subsystem with a version-robust
classification utility, and fix the resulting behavior everywhere it's
currently silently broken.

### Scope
**In scope:**
- A new `parse/dtype_utils.py` with exclusion-based, version-robust
  classification primitives.
- Updating the five confirmed call sites to use them.
- Tests in `tests/test_dtype_utils.py` exercising dtype variation.

**Out of scope:**
- `parse/analysis.py` -- confirmed already correct, left untouched.
- Any change to cardinality/length/fraction heuristic thresholds.
- Requirements.txt pandas version pinning.

### Acceptance Criteria
- [x] `parse/dtype_utils.py` exists with all five primitives,
      exclusion-based, documented.
- [x] All five confirmed call sites updated; `parse/analysis.py`
      untouched.
- [x] No threshold values changed anywhere.
- [x] `pd.Series(..., dtype=object)` and `pd.Series(..., dtype="string")`
      both classify identically through every primitive.
- [x] Every test passes unmodified across the entire test suite.
- [x] Full suite green (116 passed).
- [x] No pandas version pin added.

---

## Parking Lot
*(Anything noticed while working that's out of scope for the current
task goes here, not into the current task's diff.)*

-

---

## Upcoming (not started — for context only, do not work on these yet)

*(Tasks 1-7 are now complete.)*



