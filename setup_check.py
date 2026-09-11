"""Validate the current BF runtime layout without installing dependencies."""

import importlib
import sys
from pathlib import Path

from config.paths import DATA_FILE, DOCS_DIR, EMBED_MODEL_DIR, LLM_MODEL_DIR, ML_MODEL_DIR


REQUIRED_PACKAGES = ("numpy", "pandas", "streamlit", "torch", "transformers")
REQUIRED_DIRS = (DOCS_DIR, EMBED_MODEL_DIR, LLM_MODEL_DIR, ML_MODEL_DIR)


def check_python() -> bool:
    version = sys.version_info
    ok = version >= (3, 9)
    print(f"{'OK' if ok else 'WARN'} Python {version.major}.{version.minor}.{version.micro}")
    return ok


def check_packages() -> bool:
    healthy = True
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
            print(f"OK package {package}")
        except ImportError:
            healthy = False
            print(f"MISSING package {package}")
    return healthy


def check_paths() -> bool:
    healthy = True
    for directory in REQUIRED_DIRS:
        exists = Path(directory).is_dir()
        healthy &= exists
        print(f"{'OK' if exists else 'MISSING'} directory {directory}")

    data_exists = DATA_FILE.is_file()
    healthy &= data_exists
    print(f"{'OK' if data_exists else 'MISSING'} experiment data {DATA_FILE}")

    docs = sorted(DOCS_DIR.glob("*.txt"))
    print(f"OK {len(docs)} document(s) in {DOCS_DIR}" if docs else f"WARN no documents in {DOCS_DIR}")
    return healthy


def check_imports() -> bool:
    healthy = True
    modules = (
        "data.loader",
        "retrieval.retriever",
        "pipeline.rag_pipeline",
        "pipeline.prediction_pipeline",
        "pipeline.hybrid_pipeline",
        "parse.core.contracts",
        "parse.eda",
        "parse.analysis",
        "parse.semantic_analysis",
        "parse.cleaning_api",
        "parse.adapters.bf",
    )
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"OK import {module}")
        except Exception as exc:
            healthy = False
            print(f"BROKEN import {module}: {exc}")
    return healthy


def main() -> int:
    checks = (check_python(), check_packages(), check_paths(), check_imports())
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
