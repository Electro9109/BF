"""
setup_check.py — Run this once before launching the app.
It verifies your environment, installs missing packages, and
confirms the folder structure is correct.

Usage:  python setup_check.py
"""

import sys
import subprocess
from pathlib import Path


REQUIRED_PACKAGES = [
    "streamlit",
    "transformers",
    "torch",
]

REQUIRED_MODEL_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    # model.safetensors  (may be split into shards — we check for any .safetensors)
]

REQUIRED_DIRS = ["docs", "LocalModels"]


def ok(msg):  print(f"  ✅  {msg}")
def warn(msg): print(f"  ⚠️   {msg}")
def err(msg):  print(f"  ❌  {msg}")
def section(title): print(f"\n{'─'*55}\n  {title}\n{'─'*55}")


def check_python():
    section("Python version")
    v = sys.version_info
    if v >= (3, 9):
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        warn(f"Python {v.major}.{v.minor} detected — 3.9+ recommended.")


def install_packages():
    section("Python packages")
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg.replace("-", "_"))
            ok(f"{pkg} — already installed")
        except ImportError:
            print(f"  📦  Installing {pkg} …")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg, "-q"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                ok(f"{pkg} — installed successfully")
            else:
                err(f"{pkg} — install FAILED\n{result.stderr.strip()}")


def check_folders():
    section("Folder structure")
    for d in REQUIRED_DIRS:
        p = Path(d)
        if p.exists():
            ok(f"./{d}/  exists")
        else:
            p.mkdir(parents=True)
            warn(f"./{d}/  was missing — created it for you")


def check_docs():
    section("Documents (./docs)")
    docs = list(Path("docs").glob("*.txt"))
    if docs:
        ok(f"{len(docs)} .txt file(s) found:")
        for f in docs:
            print(f"       • {f.name}")
    else:
        warn("No .txt files found in ./docs")
        print("       Add at least one .txt document, then restart.")


def check_model():
    section("Local model (./LocalModels)")
    model_dir = Path("LocalModels")

    for fname in REQUIRED_MODEL_FILES:
        p = model_dir / fname
        if p.exists():
            ok(fname)
        else:
            err(f"{fname}  — NOT FOUND")

    safetensors = list(model_dir.glob("*.safetensors"))
    if safetensors:
        ok(f"{len(safetensors)} .safetensors shard(s) found")
    else:
        err("No .safetensors weight file found in ./LocalModels")
        print("""
       Download Qwen/Qwen1.5-0.5B-Chat files and place them here.
       Required files:
         config.json
         model.safetensors   (or sharded: model-00001-of-N.safetensors …)
         tokenizer.json
         tokenizer_config.json
        """)


def check_rag_engine():
    section("rag_engine.py import")
    try:
        import ignore.rag_engine as rag_engine  # noqa: F401
        ok("rag_engine.py imported without errors")
    except Exception as exc:
        err(f"rag_engine.py failed to import: {exc}")


def main():
    print("\n🤖  Air-Gapped RAG Engine — Setup Check\n")
    check_python()
    install_packages()
    check_folders()
    check_docs()
    check_model()
    check_rag_engine()
    print(f"\n{'─'*55}")
    print("  Done.  Fix any ❌ items above, then run:\n")
    print("    streamlit run app_web.py       (web UI)")
    print("    python main.py                 (CLI)\n")


if __name__ == "__main__":
    main()
