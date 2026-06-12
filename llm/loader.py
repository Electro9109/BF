"""
llm/loader.py - Loads Qwen-0.5B-Chat from a local folder.

Air-gap enforcement was previously at the top of rag_engine.py and ran
at import time. It's reproduced here, in the module that actually
imports transformers, so the guarantee travels with the code that
needs it rather than living in a now-deleted monolith.
"""

import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
for _var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ[_var] = ""

from config.paths import LLM_MODEL_DIR  # noqa: E402


def load_model(model_path: str = LLM_MODEL_DIR):
    """
    Load Qwen-0.5B-Chat from a local folder.

    Memory optimisations for 8 GB RAM machines
    ───────────────────────────────────────────
    • float16  -> halves RAM vs float32 (~1 GB saved for 0.5B model)
    • low_cpu_mem_usage=True  -> avoids a transient double-copy spike on load
    • Falls back to float32 if float16 raises on this CPU

    Returns (tokenizer, model).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
    return tokenizer, model
