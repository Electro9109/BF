"""
config/models.py - Model identifiers and fallback names.

Nothing outside this file should hardcode a model name. If you swap
all-MiniLM-L6-v2 for BAAI/bge-small-en-v1.5, or Qwen-0.5B for
Qwen2.5-1.5B, change it here only.
"""

# Embedding model — used by retrieval/embeddings.py
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # HF fallback name if local folder absent

# Generation model — used by llm/loader.py
LLM_MODEL_NAME = "Qwen1.5-0.5B-Chat"

# Generation settings
MAX_NEW_TOKENS = 220
GENERATION_TEMPERATURE = 0.0

# ML prediction — feature columns and default target
ML_FEATURES = ["T Fe %", "FeO %", "SiO2 %", "CaO %", "Al2O3 %", "MgO%", "Basicity"]
ML_TARGET = "Tm-Ts"
