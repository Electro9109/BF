# PARSE Final Pre-Development Audit and Handover

Audit date: 2026-10-14

## A. Repository baseline

- Repository: `Electro9109/BF`; branch `master`; starting commit
  `8401c7f71ad06fec190b81b4221421e6a2883cab`.
- At audit start the worktree was already dirty: 13 tracked files modified and
  4 untracked retrieval-evaluation files. These changes predate this audit and
  were preserved.
- No `AGENTS.md`, `CLAUDE.md`, or Copilot instruction files were found.
- Local environment: Python 3.13.14, pytest 9.1.1, NumPy 2.5.4,
  sentence-transformers 5.7.0, FAISS CPU 1.15.1, and PyTorch 2.14.1. The local
  embedding weights were present and used for the retrieval benchmark. The
  local generation model files were present, but generation inference was not
  smoke-tested. The repository does not declare a Python support floor in
  `pyproject.toml`.
- `pyproject.toml` configures pytest warning filters; no formatter, linter, or
  type-check command is configured there.

## B. Final implementation status

### Previous audit verification

| Area | Status | Evidence |
|---|---|---|
| Parsing and loading | Verified for checked formats and corpus | Regression tests cover pre-heading text, short sections, junk headings, Markdown/SECTION/legacy uppercase headings, UTF-8 BOM, UTF-16, CP1252, unreadable files, empty files, and no-indexable-section warnings. Strict mode raises on read/decode errors; non-strict mode logs and skips. |
| Existing corpus parsing | Verified | Strict loading produced 15 `.txt` documents and 117 chunks. No repeated section labels occur in the prose documents; the headingless experiment-scenarios document is intentionally split into multiple `body` paragraph chunks. No long all-caps body sentence was misread as a heading in this corpus. |
| Embeddings and FAISS | Verified for local model and deterministic reference | The embedder requests normalized vectors and FAISS uses inner product. Tests compare filtered FAISS ranking and scores with independently computed candidate dot products, and cover matrix validation, query dimensions, candidate mapping, empty candidates, and result bounds. |
| Topic filtering and reranking | Verified, with remaining retrieval misses | Ambiguous topics use broad retrieval; FeO, B2/B4, and Tm-Ts routing cases are covered. Low-candidate filtering falls back to the corpus. Reranking preserves ties and now adjusts negative similarity scores in the intended direction. |
| RAG context and failures | Verified by tests; generation itself unverified | Tests cover source/section labels, context ordering and size limits, passage truncation markers, and generation failure fallback. The lexical overlap guard is only a heuristic, not an entailment test. |
| Integration contracts | Partially verified | Existing callers in the web app, CLI, and hybrid pipeline retain the `RetrievalEngine` and `(Chunk, score)` interfaces. Core contract and BF adapter tests pass. Local LLM generation and production-data workflows were not exercised. |
| Evaluation set | Partially verified | The 22 questions have unique IDs, explicit answerability, and locators that exist in the loaded corpus. Source text was inspected for the labels. The labels are a small manually judged set, not an independent domain-expert relevance study; alternative relevant sections may be omitted. |

The prior report's `Recall@k` values were not standard passage Recall@k: the
implementation counted queries with any relevant hit (hit rate). During this
audit the metric was corrected, repeated latency sampling was added, and the
source/section labels were reviewed against the documents rather than optimized
to match one retrieval run. The benchmark now reports passage Recall@k and
query hit rate separately. There is no comparable pre-repair retrieval
baseline.

### Corrections made during this final audit

- `data/parser.py`: constrain generic all-uppercase legacy headings so long
  unpunctuated body sentences are not dropped as headings.
- `data/loader.py`: warn when a non-empty document produces no indexable
  sections.
- `retrieval/reranker.py`: make boost/penalty direction consistent for negative
  cosine scores while preserving the existing positive-score behavior.
- `config/retrieval_domain.py`: route B2/B4 and Tm-Ts overlaps to broad
  retrieval rather than one arbitrary topic.
- `eval/retrieval_eval.py`: compute passage-level Recall@k, report hit rate
  separately, exclude unanswerable queries from MRR/Recall, validate integer
  parameters, and use one warm-up plus repeated per-query median timings.
- `eval/retrieval_eval_set.json`: restore direct source-grounded locators and
  query wording that had been changed during an earlier benchmark iteration.
- Added or extended regression tests in `tests/test_parsing.py`,
  `tests/test_data_layer.py`, `tests/test_retrieval.py`, and
  `tests/test_retrieval_eval.py`.
- `README.md`: clarify that RAG ingestion currently supports `.txt`, not PDF,
  and document the accepted heading styles and optional topic header.
- `PARSE_CONTRACTS.md`: remove stale wording that said the implemented core
  contracts did not yet exist.
- `RETRIEVAL_EVALUATION.md`: document the corrected metric definitions,
  repeated timing method, and measured result.

No model, statistical calculation, Explainer, prediction implementation, or
UI workflow was changed during this audit.

## C. Verification results

Final commands run from the repository root with the configured Python 3.13.14
environment:

```text
python -m pytest -q -rs
python -m pytest -q tests/test_data_layer.py tests/test_parsing.py tests/test_retrieval.py tests/test_retrieval_eval.py
python -m eval.retrieval_eval --ks 1 3 5 --repeats 3
git diff --check
```

Results: the full run had **305 passed**, no failures, and no skips; the
focused parser/loader/retrieval/evaluation run had **55 passed**. FAISS was
installed and the local embedding model was loaded by the benchmark. The
local generation model was not invoked. No configured lint, formatter, or
type-checker was present.

## D. Retrieval benchmark and limitations

Final run: 22 queries (20 answerable, 2 unanswerable), 15 documents, 117
chunks; one warm-up search followed by three timed searches per query.
Initialization, document encoding, and index construction are excluded from
latency. Timing includes query embedding, filtering, vector search, and
reranking.

- Passage Recall@1/3/5: **0.475 / 0.692 / 0.767**
- Query hit rate@1/3/5: **0.50 / 0.75 / 0.85**
- Source coverage@1/3/5: **0.792 / 0.883 / 1.000**
- Complete passage coverage@1/3/5: **0.45 / 0.65 / 0.70**
- MRR (answerable queries only): **0.633**
- Mean / median of per-query median latency: **17.22 ms / 15.98 ms**
- Unanswerable retrieval non-empty rate: **1.00** (both returned passages)

Three answerable queries had no labelled source/section locator in the top
five. These are retrieval misses under the current judgments, not proof that
the generator cannot answer: the boundary and pressure-drop queries also have
plausible supporting sections not included in their current labels. The
width-values query did not retrieve its labelled section in the top five.
Independent domain review of the gold labels and additional questions is
recommended before making broader retrieval claims.

This verifies that the benchmark runs and characterizes retrieval on its
current examples. It does not establish generalization, generated-answer
correctness or grounding, answerability classification, or suitability for
real plant records.

Other established limitations:

- The loader recursively indexes `.txt` only. PDF, DOCX, HTML, and other
  formats are not ingested.
- Legacy uppercase heading recognition remains heuristic; section labels are
  truncated to 40 characters. The current checked-in corpus parsed without
  detected content loss after the audit correction.
- `Chunk.source` is a relative filename and `section` is a heading label,
  not a stable document revision, line span, or content hash. This is useful
  source-level attribution, not immutable provenance.
- Retrieval scores are ranking signals. The section reranker can adjust
  cosine scores; neither raw nor adjusted scores are calibrated confidence.
- The grounding guard uses lexical overlap and can both reject supported
  paraphrases and allow unsupported claims with enough word overlap.
- A non-empty retrieval result does not prove that the corpus can answer the
  question. Both unanswerable benchmark queries retrieved passages.

## E. Development handover

### Required before development can proceed

- **None for continued non-production development.** The current tests and
  local retrieval run establish a reproducible baseline with explicit
  limitations. This is not a production-readiness approval.

### Recommended early development tasks

1. Have a domain reviewer independently validate and expand retrieval
   relevance labels, especially questions requiring multiple or alternative
   supporting sections. Evidence: only 17/20 answerable queries retrieved a
   currently labelled locator in the top five, and two misses have plausible
   unlabelled support.
2. Add a separate generated-answer evaluation and human review protocol before
   relying on RAG answers. Evidence: this audit did not run generation, the
   lexical guard is not entailment, and unanswerable queries still retrieve.
3. Define stable document identity/revision and finer source locators if
   durable provenance is required. Evidence: current chunk attribution is
   filename plus a possibly truncated section label.
4. Declare and test the supported Python version range when the next phase
   formalizes runtime support. Evidence: only the local Python 3.13.14
   environment is established here; project metadata sets no floor.

### Future experiments (not implemented)

- Consider hybrid or hierarchical retrieval only after the expanded labelled
  baseline shows where vector retrieval misses useful passages.
- Evaluate other embeddings or reranking strategies only against a
  representative, independently reviewed set and with a comparable baseline.
- Add parsers for other formats only when their ingestion and provenance
  requirements are specified.

## F. Git and audit boundary

At audit start, the already modified tracked files were `app_web.py`,
`config/retrieval_domain.py`, `data/loader.py`, `data/parser.py`,
`pipeline/rag_pipeline.py`, `retrieval/faiss_index.py`,
`retrieval/reranker.py`, `retrieval/retriever.py`,
`retrieval/topic_filter.py`, `tests/test_data_layer.py`,
`tests/test_llm_pipeline.py`, `tests/test_parsing.py`, and
`tests/test_retrieval.py`. The already untracked files were
`RETRIEVAL_EVALUATION.md`, `eval/retrieval_eval.py`,
`eval/retrieval_eval_set.json`, and `tests/test_retrieval_eval.py`.

During this audit, changes were made in `README.md`, `PARSE_CONTRACTS.md`,
`config/retrieval_domain.py`, `data/loader.py`, `data/parser.py`,
`retrieval/reranker.py`, `tests/test_data_layer.py`, `tests/test_parsing.py`,
`tests/test_retrieval.py`, `eval/retrieval_eval.py`,
`eval/retrieval_eval_set.json`, `tests/test_retrieval_eval.py`, and
`RETRIEVAL_EVALUATION.md`; `PARSE_FINAL_AUDIT_HANDOVER.md` was added. Several
of those were already dirty at audit start, so their final file status alone
does not distinguish earlier changes from audit edits.

Final status is **15 tracked files modified and 5 untracked files**: the 13
pre-existing tracked files plus `README.md` and `PARSE_CONTRACTS.md`, and the
4 pre-existing untracked retrieval-evaluation files plus this handover.
`git diff --check` passed. Git reported line-ending conversion warnings for
modified Python files; no whitespace errors were reported. The changed-file
scan found no credential patterns, local absolute paths, or files larger than
5 MiB.

The worktree was not reset or cleaned. No commit, push, publication, or
deployment was performed.
