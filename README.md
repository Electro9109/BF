# Metallurgical RAG + ML Assistant

A dual-purpose **fully-offline** tool combining:
- 🔍 **RAG Chat** — query metallurgical PDF/text documents with a local LLM
- 🧮 **ML Predictor** — estimate softening (Ts) and melting (Tm, Tm−Ts) temperatures from burden chemistry, gas atmosphere, and burden composition

---

## Quick Start

```bash
# 1. Install dependencies
pip install streamlit transformers torch sentence-transformers faiss-cpu xgboost scikit-learn openpyxl

# 2. Train ML models (once, or after new data)
python -m ml.train

# 3. Launch the UI
streamlit run app_web.py
```

---

## Setup

### Required files

| Item | Where to place it |
|---|---|
| Metallurgical theory docs (`.txt`) | `docs/` — each file needs `TOPIC:` and `SECTION:` header lines |
| Sentence-embedding model weights | `EmbedModels/` — see [Embedding model](#embedding-model) |
| Chat / generation LLM weights | `LocalModels/` — see [Changing the LLM](#changing-the-llm) |
| Raw experimental data | `data_files/data_result.xlsx` — sheet `Data Analysis` |

---

## Changing the LLM

The generation model drives both the RAG Chat answers and the natural-language condition parser.

### Step 1 — Download the new model

Download any Hugging Face `transformers`-compatible causal language model (e.g. Qwen2.5-1.5B-Instruct, Mistral-7B-Instruct, Phi-3-mini):

```bash
# Example: download via huggingface-cli
pip install huggingface-hub
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir LocalModels/
```

Or copy the model folder manually.  The folder must contain at minimum:
```
LocalModels/
  config.json
  tokenizer.json           (or tokenizer_config.json + vocab files)
  model.safetensors        (or pytorch_model.bin / shards)
```

### Step 2 — Update `config/models.py`

Open [`config/models.py`](config/models.py) and change the `LLM_MODEL_NAME` constant:

```python
# Generation model — used by llm/loader.py
LLM_MODEL_NAME = "Qwen2.5-1.5B-Instruct"   # ← change to match your model folder
```

> This name is used only as a human-readable label (shown in the sidebar) — the actual
> weights are always loaded from `LocalModels/` as set in `config/paths.py`.

### Step 3 — (Optional) Update `config/paths.py`

If you want the model weights in a different folder, edit [`config/paths.py`](config/paths.py):

```python
LLM_MODEL_DIR   = "./LocalModels"   # ← change this path
```

### Step 4 — Adjust generation settings (optional)

In [`config/models.py`](config/models.py):

```python
MAX_NEW_TOKENS          = 220    # max tokens the LLM generates per answer
GENERATION_TEMPERATURE  = 0.0   # 0.0 = deterministic; higher = more creative
```

### Step 5 — Restart the app

```bash
streamlit run app_web.py
```

The model is loaded once at startup and cached for the session.

> **Note on model size**: Models larger than 3 B parameters may be too slow on CPU.
> Use `--device cuda` or a quantised (`GGUF`/`AWQ`) variant for larger models.

---

## Changing the Embedding Model

The embedding model turns document chunks into vectors for similarity search (RAG retrieval).

### Step 1 — Place weights in `EmbedModels/`

```
EmbedModels/
  config.json
  tokenizer.json
  model.safetensors
  ...
```

If the folder is empty or absent, the app automatically downloads `all-MiniLM-L6-v2`
from Hugging Face on first run.

### Step 2 — Update `config/models.py`

```python
# Embedding model — used by retrieval/embeddings.py
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # ← change to match your model folder
```

> Changing the embedding model **invalidates** any cached FAISS index.
> Click **Reload Documents** in the sidebar after restarting to rebuild it.

---

## Train the ML Models

```bash
python -m ml.train
```

- Reads `data_files/data_result.xlsx` (sheet `Data Analysis`, 144 rows)
- Engineers features and runs `RandomizedSearchCV` (n_iter = 25, cv = 5) for Random Forest and XGBoost, independently per target
- Saves all models to `MLModels/` and writes cross-validation metrics to `MLModels/benchmark_results.csv`
- Auto-selects the best model per target (lowest RMSE)

### Feature schema

| Group | Features | Count |
|---|---|---|
| Chemistry | T Fe %, FeO %, SiO2 %, CaO %, Al2O3 %, MgO%, Basicity | 7 |
| Atmosphere | CO_pct, H2_pct, N2_pct (parsed from string or NL input) | 3 |
| Burden | sinter_pct, ore_pct, pellet_pct, other_pct, num_components | 5 |
| Interactions | CO×Basicity, Reducibility Ratio = (CO+H2)/N2, Basicity×Sinter | 3 |
| Test type | type_SO, type_SOP, type_P (one-hot) | 3–5 |

### Current benchmark (5-fold CV, 132 samples)

| Target | Best model | RMSE | R² |
|---|---|---|---|
| Ts | Random Forest | ~58 °C | ~0.64 |
| Tm | Random Forest | ~44 °C | ~0.35 |
| Tm-Ts | Random Forest | ~57 °C | ~0.51 |

> Tm R² is limited by the small dataset and high scatter. Adding more experimental rows
> is the most effective way to improve it.

---

## Predictor Features

| Feature | Description |
|---|---|
| **Natural language input** | Describe conditions in plain English — hybrid regex+LLM parser extracts numbers automatically |
| **Manual string input** | Paste raw atmosphere (`CO=40% & N2=60%`) and burden (`S1-70%+O1-30%`) strings directly |
| **Confidence badge** | 🟢 High / 🟡 Medium / 🔴 Low based on nearest-neighbour Euclidean distance to training data |
| **Parser warnings** | Auto-alerts when percentages don't sum to 100 %, values are out of range, or CO < 30 % |
| **Parsed condition preview** | Expandable panel showing exact numeric values sent to the model |
| **Consistency check** | Flags when predicted `Tm − Ts` deviates from `(Tm) − (Ts)` by more than 15 °C |
| **Prediction history** | Last 10 runs with timestamps in a collapsible table |

---

## Codebase Architecture

```
├── app_web.py                    Streamlit UI (RAG Chat + Predictor tabs)
├── config/
│   ├── models.py                 LLM name, embed name, generation settings   ← change model here
│   ├── paths.py                  All directory paths                          ← change folders here
│   └── retrieval.py              Chunking strategy, TOP_K
├── data/ & data_files/
│   ├── experiments_loader.py     Loads data_result.xlsx
│   └── data_result.xlsx          Raw experimental dataset
├── docs/                         Metallurgical theory .txt files (user-supplied)
├── EmbedModels/                  Sentence-embedding weights (auto-downloaded)
├── LocalModels/                  Generation LLM weights (user-supplied)
├── MLModels/                     Trained ML model .pkl files + benchmark CSV
├── ml/
│   ├── condition_parser.py       Hybrid regex+LLM parser; normalisation & warnings
│   ├── feature_processing.py     Feature engineering for chemistry/atmosphere/burden
│   ├── predictor.py              Single-sample & batch prediction; confidence scoring
│   ├── similarity.py             Nearest-neighbour distance for confidence estimation
│   └── train.py                  RandomizedSearchCV training; per-target model selection
├── pipeline/
│   ├── prediction_pipeline.py    Orchestrates feature build → predict → summary
│   └── rag_pipeline.py           RAG retrieval + LLM answer generation
├── retrieval/                    FAISS index, embeddings, chunk ranking
└── llm/                          Offline LLM loader and generator wrapper
```

---

## Eight Validated Experiment Scenarios

Used to validate the end-to-end pipeline (text description → parser → ML → prediction):

| # | CO % | H2 % | N2 % | Sinter % | Ore % | Pellet % | Expected trend |
|---|---|---|---|---|---|---|---|
| 1 | 40 | 0 | 60 | 70 | 30 | 0 | Baseline SO — reference point |
| 2 | 30 | 0 | 70 | 70 | 30 | 0 | ↑ Ts — lower CO reducibility delays softening |
| 3 | 40 | 8 | 52 | 70 | 30 | 0 | ↓ Ts — H2 reduces wüstite faster |
| 4 | 36 | 4 | 60 | 65 | 25 | 10 | SOP mix — pellet fraction narrows Tm-Ts |
| 5 | 40 | 0 | 60 | 100 | 0 | 0 | Pure sinter — narrow cohesive zone |
| 6 | 40 | 0 | 60 | 0 | 0 | 100 | Pure pellet — wider Tm-Ts |
| 7 | 34 | 6 | 60 | 60 | 40 | 0 | H2 + SO blend — moderate Tm drop |
| 8 | 40 | 0 | 60 | 50 | 30 | 20 | SOP near-equal mix |

---

## Notes on Large Model Files

`LocalModels/model.safetensors` (~1.2 GB) and `EmbedModels/model.safetensors` (~87 MB)
exceed GitHub's 100 MB file size limit and are excluded via `.gitignore`.
Copy them into the respective directories manually after cloning.