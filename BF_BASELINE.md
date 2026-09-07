# BF Baseline

Recorded after Phase 1 completion on 2026-09-07.

## Increment 5: ML and analysis cleanup

- Feature construction now validates source columns and supports real scaler reuse.
- Prediction rejects missing or non-finite chemistry and missing required conditions.
- Natural-language parsing returns warnings instead of crashing when no LLM is available.
- Prediction results retain their feature vector for downstream applicability/comparison.
- Similarity integration accepts a callable or collaborator instead of an unfinished fixed API.
- Subset-trained model artifacts record their available targets.

## Increment 6: LLM and synthesis cleanup

- Generic prompt formatting is separated from BF/metallurgical synthesis policy.
- RAG handles missing indexes, retrieval failures, generation failures, and bounded context explicitly.
- LLM generation validates tokenizer/model availability and remains local-only.

## Increment 7: pipeline cleanup

- Prediction and hybrid pipelines expose explicit intermediate results and validate hybrid inputs.
- Historical comparison and feature-vector handoff are now connected rather than stubs.

## Increment 8: interface cleanup

- Streamlit reuses cached prediction pipelines instead of reloading models per submission.
- Uploaded document names are sanitized and restricted to `.txt` files.
- RAG unavailability no longer blocks manual prediction.
- The UI no longer requests Google Fonts, preserving offline behavior.

## Increment 9: testing and clean BF baseline

- Tests cover configuration, document/data processing, retrieval, ML contracts, prompt policy, RAG failures, and pipeline handoff.
- Phase 1 is complete; PARSE Core extraction is intentionally deferred until this baseline is reviewed.

## Increment 4: retrieval cleanup

- `RetrievalEngine` remains the retrieval facade.
- Embedding, index construction, topic detection/filtering, and reranking are injectable collaborators.
- Topic filtering now owns its candidate fallback behavior and reports when domain evidence is insufficient.
- Reranking accepts injected section and query policies while preserving BF defaults.
- FAISS and SentenceTransformers remain infrastructure implementations behind their existing wrappers.
- Retrieval tests cover topic policy, fallback behavior, reranking, engine orchestration, and the FAISS wrapper.

## Increment 3: data-layer cleanup

- Generic document contracts remain in `data/schemas.py`.
- Sinter experiment contracts and CSV mappings are isolated in `data/sinter_schemas.py`.
- Document loading supports explicit strict-mode failures for unreadable input.
- Historical experiment validation reports malformed or incomplete values while retaining source rows.
- Existing `Tm-Ts = Tm - Ts` repair behavior is preserved.
- Data-layer behavior is covered by `tests/test_data_layer.py`.

## Increment 2: configuration cleanup

- Infrastructure model settings remain in `config/models.py`.
- Repository and storage paths remain in `config/paths.py`.
- Generic retrieval mechanics remain in `config/retrieval.py`.
- BF/Sinter topic vocabulary and section policy are in `config/retrieval_domain.py`.
- Sinter ML features and targets are in `config/sinter.py`.
- Configuration boundaries are covered by `tests/test_config.py`.

## Active entry points

- `streamlit run app_web.py` starts the Streamlit RAG and prediction interface.
- `python main.py "<question>"` runs the current offline RAG workflow from the command line.
- `python -m ml.train` trains the Sinter prediction models.
- `python setup_check.py` validates dependencies, paths, and active module imports without installing or creating files.

## Current workflow

- Documents are loaded from `docs/`, converted into `Chunk` records, embedded, searched, and passed to the local LLM through `pipeline/rag_pipeline.py`.
- Sinter experiments are loaded from `data_files/data_result.xlsx` for feature construction and from `ignore/SMRF.csv` for historical comparison.
- Prediction is orchestrated by `pipeline/prediction_pipeline.py` and uses the trained models in `MLModels/`.
- The repaired hybrid workflow combines prediction, historical similarity, retrieval, and synthesis through `pipeline/hybrid_pipeline.py`.

## Measured checks

- `pytest -q`: 24 passed.
- `python -m compileall -q config data retrieval ml pipeline llm main.py setup_check.py app_web.py`: passed.
- `python setup_check.py`: passed for the installed environment, repository paths, and active imports.
- Historical CSV loader: 131 rows loaded with the existing `Tm-Ts = Tm - Ts` repair behavior.
- Document retrieval limitation: topic detection uses the first matching configured domain topic; ambiguous or underrepresented topics fall back to the full corpus.
- Model limitation: the local LLM and embedding model are still heavyweight runtime dependencies and were not launched during automated validation.

## Known scope

The local LLM and embedding models are large runtime dependencies, so automated validation uses mocked/fake collaborators. Domain vocabulary remains in retrieval configuration and Sinter-specific analysis remains in the BF modules; extracting reusable PARSE contracts is intentionally deferred until this cleaned implementation is reviewed.

## Prototype smoke check

- Real trained prediction path verified: `Ts=1294.9`, `Tm=1490.0`, `Tm-Ts=211.5` for the documented sample condition.
- Real local retrieval path verified: 57 chunks indexed and Sinter evidence returned.
- Streamlit health endpoint verified with HTTP 200 on port 8501.
- Streamlit prototype is currently available at `http://localhost:8501`.
