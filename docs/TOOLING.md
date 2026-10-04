# TOOLING.md — Tooling and Architecture Recommendations for PARSE

This document evaluates the current architecture and provides concrete, prioritized recommendations across code quality, dependency management, data storage, model serving, UI, and computational performance.

---

## 1. Quality & CI/CD Tooling

### Current State
- Testing runs via `pytest` (171 tests, ~30s execution time).
- No pre-commit hooks, automated linter, or static type checker are configured.
- Occasional path drift and dead dependencies (`PyPDF2`, `python-docx`) accumulated silently over time.

### Concrete Recommendations

1. **Ruff (Linter & Formatter)**
   - **Why:** Replaces Flake8, Black, isort, and pyupgrade in a single sub-second tool.
   - **Configuration:** Add `[tool.ruff]` to `pyproject.toml` selecting `E`, `F`, `I` (import sorting), `B` (bugbear), and `UP` (pyupgrade).
   - **Impact:** Eliminates unused imports, unifies formatting, catches mutable defaults, and runs in milliseconds.

2. **Mypy or Pyright (Static Type Checking)**
   - **Why:** Core protocols (`Synthesizer`, `Department`, `AnalysisPipeline`, `CleaningContext`) rely heavily on dataclasses, typing protocols, and runtime duck-typing.
   - **Configuration:** Run `mypy --check-untyped-defs parse/ departments/`.
   - **Impact:** Catches schema mismatches and contract violations before runtime.

3. **Pre-commit Hooks**
   - **Configuration:** `.pre-commit-config.yaml` running `ruff check --fix`, `ruff format`, and `pytest tests/test_docs.py` (to ensure documentation references remain valid).

4. **Continuous Integration (GitHub Actions / GitLab CI)**
   - **Matrix:** Test on Python 3.11, 3.12, and 3.13 on Windows and Ubuntu.
   - **Pipeline:** Lint -> Type Check -> Pytest -> Offline Setup Check (`setup_check.py`).

---

## 2. Dependency Management

### Current State
- Plain `requirements.txt` with loose version specifiers (`streamlit>=1.41.0`, `torch`, `transformers`).
- No lockfile exists. Re-installing tomorrow may pull breaking minor or patch versions of transitive dependencies.

### Concrete Recommendations

1. **Adopt `uv` as the Package Manager**
   - **Why:** Extremely fast package resolution and virtualenv creation. Ships with `uv pip compile` and native `uv.lock`.
   - **Workflow:**
     - Maintain abstract requirements in `pyproject.toml` dependencies.
     - Generate a pinned, reproducible `uv.lock` or `requirements.lock`.
   - **Alternative:** `pip-tools` (`pip-compile requirements.in -> requirements.txt`) if a zero-binary pure-pip solution is preferred.

2. **Offline Dependency Bundling (Wheelhouse)**
   - **Why:** PARSE must run fully offline in industrial/secure environments.
   - **Implementation:** Script `pip download -r requirements.lock -d wheelhouse/` to permit full air-gapped installation:
     ```bash
     pip install --no-index --find-links=wheelhouse -r requirements.lock
     ```

---

## 3. Data Storage & Query Layer

### Current State
- Operational data: 130 blast furnace experimental rows stored in Excel (`data_files/data_result.xlsx`) and CSV.
- Intermediate profiling: In-memory pandas DataFrames.

### Evaluation: Parquet vs. DuckDB vs. Excel at Current Scale (130 Rows)

| Format / Tool | Fit at 130 Rows | Fit at 100k+ Rows | Offline Complexity | Recommendation |
|---|---|---|---|---|
| **Excel (`.xlsx`)** | Adequate for human inspection; slow parse time (`openpyxl`), brittle schema, no type safety. | Unusable (row limits, slow). | Low | **Retain as import/export format only.** |
| **Parquet** | Lightweight; instant load; preserves exact dtypes (timestamps, categorical, numeric). | Industry standard for columnar analytics. | Low (`pyarrow` or `fastparquet`). | **Adopt as primary on-disk cache for ingested/cleaned datasets.** |
| **DuckDB** | Overkill for 130 rows in-memory, but provides zero-dependency embedded SQL and OLAP profiling. | Exceptional fast querying and relationship joins. | Low (single C++ binary packaged in Python wheel). | **Evaluate when dataset size exceeds 50,000 rows or when cross-table joins are needed.** |

**Recommendation:**
- Convert cleaned datasets and baseline historical experiments to `.parquet` alongside `.xlsx`.
- Retain `.xlsx` support for upload/download boundaries so plant operators can use Excel without friction.

---

## 4. Model Serving & Local Inference

### Current State
- Predictor models: Pickled scikit-learn models (`MLModels/*.pkl`) with manual feature scaler loading.
- Explainer / RAG LLMs: HuggingFace Transformers pipelines loaded in memory via PyTorch (`LocalModels/`).

### Recommendations

1. **Replace Pickle with `skops` or ONNX Runtime**
   - **Security Risk:** `pickle.load()` is vulnerable to arbitrary code execution if an unvetted model file is placed in `MLModels/`.
   - **`skops`:** Built by the scikit-learn team specifically for safe persistence without arbitrary code execution.
   - **ONNX Runtime:** Exports scikit-learn models to ONNX format. Yields faster CPU inference and eliminates dependency on exact scikit-learn versions across Python releases.

2. **Explainer LLM Serving Options**
   - **Current:** Local HuggingFace Transformers pipeline running PyTorch on CPU/GPU. High memory footprint (>4GB VRAM or RAM).
   - **Recommendation (Quantized Local Serving):**
     - **llama.cpp / llama-cpp-python:** Serve 4-bit/8-bit GGUF models on CPU/GPU with minimal memory footprint (e.g. Mistral-7B or Qwen-2.5-Coder running in <3GB RAM).
     - **vLLM / Ollama:** If deployed as a background daemon or container, provides high-throughput batched inference and standardized OpenAI-compatible REST endpoints.

---

## 5. UI Architecture: Streamlit vs. FastAPI + Frontend

### Current State
- Monolithic Streamlit application (`app_web.py`, 1,300 lines) with inline dark-theme CSS and state management.

### Trade-off Evaluation

| Dimension | Streamlit (Current) | FastAPI + React/Vue (Alternative) |
|---|---|---|
| **Development Velocity** | Extremely high for single-page Python-centric prototypes. | Lower initial velocity; requires dual-stack maintenance. |
| **State Management** | Re-runs script on interaction; requires defensive `st.session_state` guards. | Explicit, client-side state; robust lifecycle control. |
| **UI Customization & Aesthetics** | Requires heavy CSS injection overrides (`!important`). | Full styling freedom (Tailwind, Radix, Shadcn). |
| **Air-gapped Deployment** | Single process; simple `streamlit run app_web.py`. | Multi-process (API server + static build hosting). |
| **Testability** | Hard to unit-test visual UI components directly. | Clean decoupling: API tested via `pytest`, UI tested via Playwright/Vitest. |

**Recommendation:**
- **Short-to-Medium Term:** Retain Streamlit, but modularize `app_web.py` into separate view modules (`ui/components/data_explorer.py`, `ui/components/rag_chat.py`, `ui/components/predictor.py`) and move inline CSS into an external `assets/styles.css`.
- **Long Term (if enterprise web integration is required):** Extract core PARSE workflows into a FastAPI REST API (`parse-server`), maintaining an identical Python engine while allowing React/desktop web clients to connect.

---

## 6. Performance & Language Strategy (Python vs. Rust)

### Current State
- The core EDA and cleaning engines are implemented in pure Python with vectorized NumPy/pandas operations.
- At current data scales (tens to hundreds of rows), total profiling latency is <50ms.
- Test suite of 171 tests finishes in ~30s (dominated by PyTorch / Transformers model loading in integration tests).

### Evaluation: Python vs. Rust / Polars

1. **Current Bottlenecks:**
   - Profiling and data cleaning are **not** CPU-bound bottlenecks. Vectorized pandas operations take <20ms for 1,000 rows.
   - The primary latencies are:
     - LLM generation in PyTorch.
     - Pytest startup / torch imports.
     - Openpyxl parsing for large Excel files.

2. **Polars vs. Pandas:**
   - Polars (written in Rust with Python bindings) offers 10x-50x speedups on multi-million row datasets.
   - For PARSE's core tabular operations, migrating to Polars would add significant rewrite cost with zero perceptible latency benefit on datasets <100,000 rows.

3. **Where Rust Makes Sense in Future Iterations:**
   - If PARSE is embedded directly into real-time industrial PLC / SCADA data collection streams (millisecond-frequency blast furnace telemetry).
   - If tokenization or local vector retrieval needs sub-millisecond edge latency without Python runtime dependencies (via PyO3).

---

## 7. Actionable Implementation Roadmap

1. **Step 1 (Immediate):**
   - Add `ruff` configuration to `pyproject.toml`.
   - Extract `app_web.py` CSS into `assets/styles.css`.
2. **Step 2 (Near-Term):**
   - Add `uv.lock` for deterministic, reproducible offline builds.
   - Set up GitHub Actions CI for linting, type-checking, and pytest.
3. **Step 3 (Next Architecture Cycle):**
   - Transition pickled ML predictors to `skops` or ONNX.
   - Provide GGUF quantized runtime option (`llama-cpp-python`) for the Explainer LLM.
