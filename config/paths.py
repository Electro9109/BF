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
STORAGE_DIR = PROJECT_ROOT / "storage"
SOURCE_CATALOGUE_DB = STORAGE_DIR / "catalogue" / "sources.sqlite3"
SOURCE_ORIGINALS_DIR = STORAGE_DIR / "private" / "originals"
SOURCE_PUBLIC_ORIGINALS_DIR = STORAGE_DIR / "public" / "originals"
SOURCE_NORMALIZED_DIR = STORAGE_DIR / "normalized"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SINTER_RAW_DIR = RAW_DATA_DIR / "sintering"
ENERGY_RAW_DIR = RAW_DATA_DIR / "energy"
SINTER_PRODUCTIVITY_CSV = SINTER_RAW_DIR / "SinterProductivityData.csv"
STEEL_ENERGY_CSV = ENERGY_RAW_DIR / "Steel_industry_data.csv"


