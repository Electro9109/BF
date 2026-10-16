# Structured Query and Analysis Integration

## Division of Responsibility

The PARSE architecture maintains clear separation of responsibilities:

**Structured Query Engine (`parse/structured_query.py`):**
- Retrieves or calculates exact results from tabular data
- Supported operations: get_columns, get_statistics, filter_rows, aggregate, get_top_rows
- Returns QueryResult with dataset_id, operation, data, row_count, and metadata
- Read-only, constrained execution (no arbitrary SQL, whitelisted operators)

**Analysis Engine (`parse/analysis.py`):**
- Performs exploratory data analysis
- Data-quality checks, anomaly identification
- Correlation analysis, distribution analysis
- Generates recommendations
- Returns AnalysisResult with evidence and provenance

**Document Retrieval (`pipeline/rag_pipeline.py`):**
- Finds relevant text in reference material and internal documents
- Semantic search with role-aware context (reference vs experiment_scenario)
- Returns chunks with source attribution

**Prediction Pipeline (`ml/unified_calculator.py`, `pipeline/prediction_pipeline.py`):**
- Generates predictions from supported models
- Input validation, range checking
- Returns DepartmentPredictionResult with metrics, warnings, limitations

**Explanation/Translation (`parse/translator_pipeline.py`):**
- Explains supplied numerical results and recommendations
- Preserves numerical values and units
- Deterministic fallback when fidelity checks fail

## Current Integration Status

**Structured Query Engine:**
- Integrated into Streamlit UI as "Data Query" tab
- Exposes all 5 operations through interactive interface
- Returns results with metadata (dataset_id, operation, filters, row_count)
- Results can be downloaded as CSV

**Provenance Preservation:**
- QueryResult includes dataset_id (identifies source)
- QueryResult includes operation (identifies what was computed)
- QueryResult includes metadata (filters, aggregation details, row count)
- This metadata provides sufficient provenance for traceability

**Explanation Integration:**
- **Not yet integrated** - Explanation LLM is currently used for prediction results and analysis results
- Structured query results are displayed directly as tables or JSON
- **Rationale:** The structured results are already useful and traceable. Explanation integration would add complexity without clear benefit for numerical query results. Future enhancement could add natural-language summaries.

## Separation of Concerns

**Structured computation remains separate from document retrieval:**
- Structured queries operate on tabular data (CSV, Excel)
- Document retrieval operates on text chunks
- The UI separates these into different tabs (RAG Chat vs Data Query)
- No mixing of structured computation with semantic retrieval

**No confusion between descriptive statistics and predictions:**
- Query results are descriptive statistics (mean, count, filter results)
- Prediction results are model-based estimates
- UI separates these into different tabs (Predictor vs Data Query)
- Prediction results include model metadata, query results do not

## Conclusion

The current architecture maintains proper separation of responsibilities. Structured query results are traceable through metadata. Explanation integration is deferred to a future phase as the structured results are already useful and clear.
