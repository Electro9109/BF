# Metallurgical RAG + ML Assistant

A fully-offline prototype combining:
- 📊 **Data Explorer + Cleaner** — profile CSV/Excel datasets, review quality issues, and apply approved traceable cleaning proposals
- 🔍 **RAG Chat** — query metallurgical PDF/text documents with a local LLM
- 🧮 **ML Predictor** — estimate softening (Ts) and melting (Tm, Tm−Ts) temperatures from burden chemistry, gas atmosphere, and burden composition

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Check the local installation
python setup_check.py

# 3. Train ML models only if MLModels/ is empty or data/features changed
python -m ml.train

# 4. Launch the UI
streamlit run app_web.py
```

The verified prototype URL is `http://localhost:8501` when Streamlit uses its default port.
The interface provides Data Explorer/Cleaner, RAG Chat, and the Sinter predictor. Data Explorer profiles
CSV/Excel inputs read-only and presents findings, limitations, possible next actions, and a controlled cleaning review. The Cleaner never changes data until selected proposals are approved, preserves the original frame, and records before/after quality snapshots and value-level changes. The predictor can run with the
trained local artifacts even when the local LLM is unavailable; RAG Chat and natural-language
condition parsing require the files in `LocalModels/`.

For a lightweight command-line RAG check:

```bash
python main.py "Explain sinter reducibility"
```

The first UI launch builds the local document index and loads the local models. This can take
several minutes on CPU and does not require network access.

> **First launch is slow.** Loading the ~1.2 GB local LLM into memory on CPU can take
> several minutes the first time. This is a one-time cost per session — subsequent
> interactions are fast. A loading spinner is shown while this happens.

---

## Setup

### Required files

| Item | Where to place it |
|---|---|
| Metallurgical theory docs (`.txt`) | `docs/` — each file needs `TOPIC:` and `SECTION:` header lines |
| Sentence-embedding model weights | `EmbedModels/` — see [Embedding model](#changing-the-embedding-model) |
| Chat / generation LLM weights | `LocalModels/` — see [Changing the LLM](#changing-the-llm) |
| Raw experimental data | `data_files/data_result.xlsx` — sheet `Data Analysis` |

---

## Changing the LLM

The generation model drives both the RAG Chat answers and the natural-language condition parser.

### Step 1 — Download the new model

Download any Hugging Face `transformers`-compatible causal language model (e.g. Qwen2.5-1.5B-Instruct, Mistral-7B-Instruct, Phi-3-mini):

```bash
pip install huggingface-hub
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir LocalModels/
```

Or copy the model folder manually. The folder must contain at minimum:
```
LocalModels/
  config.json
  tokenizer.json           (or tokenizer_config.json + vocab files)
  model.safetensors        (or pytorch_model.bin / shards)
```

### Step 2 — Update `config/models.py`

```python
LLM_MODEL_NAME = "Qwen2.5-1.5B-Instruct"   # ← change to match your model folder
```

> This name is used only as a human-readable label (shown in the sidebar) — the actual
> weights are always loaded from `LocalModels/` as set in `config/paths.py`.

### Step 3 — (Optional) Update `config/paths.py`

```python
LLM_MODEL_DIR = "./LocalModels"   # ← change this path
```

### Step 4 — Adjust generation settings (optional)

```python
MAX_NEW_TOKENS         = 220   # max tokens the LLM generates per answer
GENERATION_TEMPERATURE = 0.0   # 0.0 = deterministic; higher = more creative
```

### Step 5 — Restart the app

```bash
streamlit run app_web.py
```

The model is loaded once at startup and cached for the session.

> **Note on model size**: Models larger than 3B parameters may be too slow on CPU.
> Use `--device cuda` or a quantised (`GGUF`/`AWQ`) variant for larger models — this
> also reduces the first-launch load time.

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
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # ← change to match your model folder
```

> Changing the embedding model **invalidates** any cached FAISS index.
> Click **Reload Documents** in the sidebar after restarting to rebuild it.

---

## Feature Normalization

All numeric input features are normalized using **fixed, domain-knowledge ranges**
rather than ranges derived from the training data. This means:

- A "T Fe %" of 60 always normalizes to the same value, regardless of what range
  happens to appear in the current dataset — important since the ~132-row dataset
  doesn't cover the full practical range of any feature.
- Ranges are defined once in `ml/feature_processing.py` as `CHEM_RANGES`,
  `ATM_RANGES`, `BURDEN_RANGES`, and `INTERACTION_RANGES`, and applied via a
  lightweight `RangeScaler` (min-max scaling: `(x - lo) / (hi - lo)`).
- This replaced the previous `StandardScaler` (z-score) approach. For tree-based
  models (Random Forest, XGBoost) this has **no effect on prediction accuracy** —
  splits are scale-invariant — but it makes nearest-neighbour distance (used for the
  confidence badge) and the parsed-condition preview consistent and interpretable
  across runs.
- `RangeScaler` does **not clip** — inputs outside the defined practical range will
  normalize to values <0 or >1. This is intentional; combine with `Parser warnings`
  (see below) if you want out-of-range inputs flagged to the user.

If you add new features or change a chemistry/atmosphere/burden range, update the
relevant `*_RANGES` dict in `ml/feature_processing.py` and **retrain**
(`python -m ml.train`) — old `.pkl` files in `MLModels/` are not compatible with a
changed feature space.

---

## Train the ML Models

```bash
python -m ml.train
```

- Reads `data_files/data_result.xlsx` (sheet `Data Analysis`, 132 valid rows after
  dropping incomplete records)
- Engineers features (see schema below) and runs `RandomizedSearchCV`
  (n_iter = 25, cv = 5) for Random Forest and XGBoost, independently per target
- Saves all models to `MLModels/` and writes cross-validation metrics to
  `MLModels/benchmark_results.csv`
- Auto-selects the best model per target (lowest RMSE)

### Feature schema

| Group | Features | Count | Normalization |
|---|---|---|---|
| Chemistry | T Fe %, FeO %, SiO2 %, CaO %, Al2O3 %, MgO%, Basicity | 7 | `CHEM_RANGES` (fixed domain ranges) |
| Atmosphere | CO_pct, H2_pct, N2_pct (parsed from string or NL input) | 3 | `ATM_RANGES` |
| Burden | sinter_pct, ore_pct, pellet_pct, other_pct, num_components | 5 | `BURDEN_RANGES` |
| Interactions | CO×Basicity, Reducibility Ratio = (CO+H2)/N2, Basicity×Sinter | 3 | `INTERACTION_RANGES` (derived from the above) |
| Test type | type_SO, type_SOP, type_P (one-hot) | 3–5 | none |

### Current benchmark (5-fold CV, 132 samples)

| Target | Best model | MAE | RMSE | R² |
|---|---|---|---|---|
| Ts | Random Forest | ~46 °C | ~58 °C | ~0.64 |
| Tm | Random Forest | ~36 °C | ~44 °C | ~0.35 |
| Tm-Ts | Random Forest | ~46 °C | ~57 °C | ~0.51 |

> Tm R² is limited by the small dataset and high scatter. Adding more experimental rows
> is the most effective way to improve it. Normalization scheme does not materially
> change these metrics for tree-based models.

---

## Predictor Features

| Feature | Description |
|---|---|
| **Natural language input** | Describe conditions in plain English — hybrid regex+LLM parser extracts numbers automatically |
| **Manual string input** | Paste raw atmosphere (`CO=40% & N2=60%`) and burden (`S1-70%+O1-30%`) strings directly |
| **Confidence badge** | 🟢 High / 🟡 Medium / 🔴 Low based on nearest-neighbour Euclidean distance to training data (computed on normalized features) |
| **Parser warnings** | Auto-alerts when percentages don't sum to 100%, values are out of range, or CO < 30% |
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
│   ├── feature_processing.py     Feature engineering + RangeScaler normalization (chemistry/atmosphere/burden/interactions)
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

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `TypeError: build_features() got an unexpected keyword argument 'fit_scalers'` | `build_features` signature was edited but `load_and_build`/callers still pass `fit_scalers`/`scalers`. Keep these as accepted (unused) kwargs for backward compatibility. |
| `ValueError: Shape of passed values is (N, X), indices imply (N, Y)` | A feature block's array wasn't appended to `parts` while its column names were still appended to `feature_names`. Check every block does both `parts.append(...)` and `feature_names += [...]`. |
| Predict button does nothing — no error, no result | Check the terminal for a traceback caught by an inner `try/except` in `pipeline/prediction_pipeline.py` that may be returning empty `predictions`. Confirm `result_obj.predictions` actually contains `Ts`/`Tm`/`Tm-Ts` keys. |
| `.pkl` predictions look wrong after changing `*_RANGES` | Re-run `python -m ml.train` — models trained on the old feature scale are incompatible with new ranges. |

---

## Notes on Large Model Files

`LocalModels/model.safetensors` (~1.2 GB) and `EmbedModels/model.safetensors` (~87 MB)
exceed GitHub's 100 MB file size limit and are excluded via `.gitignore`.
Copy them into the respective directories manually after cloning.