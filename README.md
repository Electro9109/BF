# Metallurgical RAG + ML Assistant

A dual-purpose tool combining an offline **RAG (Retrieval-Augmented Generation)** chatbot for querying metallurgical documentation and a **Machine Learning Predictor** for estimating softening and melting temperatures (Ts, Tm, Tm-Ts) based on burden chemistry.

## Setup
1. Place theory `.txt` files (with `TOPIC:` and `SECTION:` headers) in `docs/`
2. Place `all-MiniLM-L6-v2` weights in `EmbedModels/` (or leave empty for automatic download)
3. Place Qwen-0.5B-Instruct weights in `LocalModels/`
4. Make sure raw data Excel file is placed in `data_files/data_result.xlsx`

## Train the ML models
The ML model calculates softening-melting parameters based on the chemistry profile (7 features: `T Fe %`, `FeO %`, `SiO2 %`, `CaO %`, `Al2O3 %`, `MgO%`, `Basicity`).

Train and save models (Random Forest & XGBoost) on chemistry features:
```bash
python -m ml.train --no-atmosphere --no-burden --no-type
```

## Running the Application
Launch the offline Web UI (Streamlit):
```bash
streamlit run app_web.py
```

This starts a local dashboard with two main tabs:
1. **🔍 RAG Chat** — offline chat interface querying the documents index.
2. **🧮 Predictor** — chemistry parameter calculator page for Ts, Tm, and Tm-Ts predictions.

## Codebase Architecture
- **`config/`**: System paths and model configuration variables.
- **`data/` & `data_files/`**: Raw dataset parsing and loader.
- **`retrieval/`**: Embeddings, FAISS index search, and ranking functions.
- **`llm/`**: Off-line LLM loaders, generator wrapper, and prompt templates.
- **`ml/`**: Chemistry feature engineering, training script, and predictions interface.
- **`pipeline/`**: Integrated RAG and prediction workflows.
- **`app_web.py`**: Streamlit application UI.