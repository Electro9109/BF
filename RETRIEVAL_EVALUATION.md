# Retrieval evaluation

Run the corpus retrieval reference evaluation from the repository root:

```bash
python -m eval.retrieval_eval
```

Use `--docs-dir`, `--eval-set`, or `--ks 1 3 5` to override the defaults. The
versioned JSON evaluation set pairs each question with answerability labels
and `source`/`section` passage locators. Tests verify that all labelled
passages exist in the checked-in corpus; their semantic relevance is a
manually judged locator choice, not an automated entailment check.

The default runner warms retrieval using the first question, then measures each
query three times and reports the median of its samples; `--repeats N` changes
the repetition count. Model initialization, corpus embeddings, and index
construction occur before timing.

The output includes:

- `recall@k`: mean fraction of each answerable query's labelled source/section
  passages retrieved in the first `k` results.
- `hit_rate@k`: fraction of answerable queries with at least one labelled
  passage in the first `k` results.
- `mrr`: mean reciprocal rank of the first labelled passage for answerable
  queries. Unanswerable queries are excluded from both MRR and Recall@k.
- `source_coverage@k`: mean fraction of each query's required source documents
  represented in the first `k` results.
- `complete_passage_coverage@k`: fraction of answerable queries for which all
  labelled source/section locators were retrieved.
- Query latency is measured around `engine.search`, including query embedding,
  filtering, search, and reranking; it excludes document loading, corpus
  embedding/index construction, the initial warm-up search, and answer
  generation. Mean and median are calculated across per-query medians.
- `unanswerable_retrieval_nonempty_rate` reports how often any result was
  returned for the small unanswerable subset. This is a retrieval diagnostic,
  not an answerability classifier or a measure of generation safety.

The benchmark is a small, manually labelled reference set, not a statistically
representative measure of all user questions. Its corrected passage-level
Recall@k differs from a prior report that called query hit rate "Recall@k";
there is no comparable pre-repair baseline, so these measurements do not
establish improvement. Do not tune against this small set and present its
scores as a general quality claim.

Reference run on the checked-in corpus (15 documents, 117 chunks; 20 answerable
and 2 unanswerable questions; one warm-up and three timed searches per query):
Recall@1/3/5 was 0.475/0.692/0.767, query hit rate@1/3/5 was 0.50/0.75/0.85,
and MRR was 0.633. Source coverage@1/3/5 was 0.792/0.883/1.000; complete
passage coverage@1/3/5 was 0.45/0.65/0.70. Mean and median per-query median
latency were 17.22 ms and 15.98 ms. Both unanswerable queries returned
passages. Three of 20 answerable queries had no labelled passage in the top
five; the boundary and pressure-drop labels may omit alternative directly
relevant sections, so those misses require expert review rather than being
treated as definitive answer failures.

## Retrieval behavior and limits

Corpus vectors are normalized and searched using FAISS inner product, so the
unadjusted scores are cosine similarities. Explanatory-query section boosts
and penalties modify ranking scores; the `RetrievalEngine.search` tuple's
score is therefore not calibrated confidence and may not be raw cosine
similarity.

The RAG pipeline's lexical grounding guard checks word overlap only. It is not
an entailment check: it can reject a supported paraphrase with low lexical
overlap and allow an unsupported claim that repeats enough source terms. The
retrieval evaluation measures passage ranking only; it does not evaluate
generated-answer correctness, factual entailment, or safety.
