"""Repository-owned filesystem paths.

Paths are anchored to this file so imports and commands behave the same
regardless of the process working directory.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCS_DIR = PROJECT_ROOT / "docs"
EMBED_MODEL_DIR = PROJECT_ROOT / "EmbedModels"
LLM_MODEL_DIR = PROJECT_ROOT / "LocalModels"
ML_MODEL_DIR = PROJECT_ROOT / "MLModels"
DATA_FILE = PROJECT_ROOT / "data_files" / "data_result.xlsx"
EXPERIMENTS_CSV = PROJECT_ROOT / "ignore" / "SMRF.csv"
