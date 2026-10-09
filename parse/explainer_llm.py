"""PARSE Explainer LLM — model loader and inference engine.

Loads the Qwen3-4B-Instruct-2507 base model + the local v0.3 LoRA adapter
and exposes a single ``explain()`` function.

Design rules
------------
- Model loading is isolated here; nothing else in PARSE imports transformers/peft.
- The model can be disabled by setting PARSE_EXPLAINER_ENABLED=0 in the
  environment; in that case explain() returns the fallback immediately.
- Model loading is lazy (done on first call or explicit load_model()).
- Inference is deterministic: do_sample=False (greedy decoding).
- If validation fails, explain() returns the analyser's original message as
  a deterministic fallback rather than a hallucinated explanation.

Adapter path
------------
The v0.3 adapter is expected at::

    parse/models/qwen3-explainer-v0.3/

relative to the repository root.  The base model is loaded from the
Hugging Face hub by model ID ("Qwen/Qwen3-4B-Instruct-2507"); the HF cache
is used, so the weights are never committed to Git.

Usage
-----
    from parse.explainer_llm import explain

    record = {...}           # Explainer record dict from explainer_adapter.py
    result = explain(record, recommendation="Consider removing batch_id.")
    print(result.text)       # Human-readable explanation (or fallback)
    print(result.fallback_used)   # True if fidelity validation failed
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from parse.explainer_prompt import build_chat_messages

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

_REPO_ROOT = Path(__file__).parent.parent
ADAPTER_PATH = _REPO_ROOT / "parse" / "models" / "qwen3-explainer-v0.3"

MAX_NEW_TOKENS = 256
ENABLED_ENV_VAR = "PARSE_EXPLAINER_ENABLED"
BASE_MODEL_PATH_ENV_VAR = "PARSE_BASE_MODEL_PATH"

# ── State ──────────────────────────────────────────────────────────────────────

_model = None
_tokenizer = None


# ── Public API ─────────────────────────────────────────────────────────────────


@dataclass
class ExplainerResult:
    """Output of one explanation call."""

    text: str
    """The explanation text (model output or fallback)."""

    fallback_used: bool = False
    """True when fidelity validation failed and the fallback was returned."""

    validation_detail: str = ""
    """Human-readable reason if fallback_used is True."""

    finding_id: str = ""
    """Echo of the finding_id for logging/tracing."""

    prompt_token_count: int = 0
    """Number of tokens in the formatted prompt (for diagnostics)."""

    generated_token_count: int = 0
    """Number of newly generated tokens (prompt tokens excluded)."""


def is_enabled() -> bool:
    """Return True unless PARSE_EXPLAINER_ENABLED is set to '0' or 'false'."""
    val = os.environ.get(ENABLED_ENV_VAR, "1").strip().lower()
    return val not in ("0", "false", "no", "off")


def _resolve_base_model_path() -> str:
    """Return the base model path/ID.

    Priority:
    1. PARSE_BASE_MODEL_PATH env var (for air-gapped or local-weights usage)
    2. BASE_MODEL_ID constant (downloads from / uses HF cache)
    """
    custom = os.environ.get(BASE_MODEL_PATH_ENV_VAR, "").strip()
    if custom:
        p = Path(custom)
        if p.is_dir():
            return str(p)
        logger.warning(
            "%s=%r is not a directory; falling back to HF model ID %s",
            BASE_MODEL_PATH_ENV_VAR, custom, BASE_MODEL_ID,
        )
    return BASE_MODEL_ID


def load_model() -> None:
    """Explicitly load the base model + LoRA adapter into the module cache.

    Safe to call multiple times; subsequent calls are no-ops.
    Raises ImportError if transformers/peft are not installed.
    Raises FileNotFoundError if the adapter directory is missing.
    """
    global _model, _tokenizer
    if _model is not None:
        return

    if not ADAPTER_PATH.is_dir():
        raise FileNotFoundError(
            f"LoRA adapter not found at {ADAPTER_PATH}. "
            "Place the v0.3 adapter files there before loading the model."
        )

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required for the Explainer LLM. "
            "Install it with: pip install transformers"
        ) from exc

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError(
            "peft is required for the Explainer LLM. "
            "Install it with: pip install peft"
        ) from exc

    base_model_path = _resolve_base_model_path()
    logger.info("Loading base model: %s", base_model_path)

    # Prefer adapter tokenizer when it ships its own files (e.g. v0.3 adapter
    # includes tokenizer.json + tokenizer_config.json); fall back to the base.
    adapter_has_tokenizer = (ADAPTER_PATH / "tokenizer_config.json").is_file()
    tokenizer_source = str(ADAPTER_PATH) if adapter_has_tokenizer else base_model_path
    logger.info("Loading tokenizer from: %s", tokenizer_source)
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    base = AutoModelForCausalLM.from_pretrained(base_model_path)
    logger.info("Loading LoRA adapter from: %s", ADAPTER_PATH)
    _model = PeftModel.from_pretrained(base, str(ADAPTER_PATH))
    _model.eval()
    logger.info("Explainer LLM ready.")


def unload_model() -> None:
    """Release the model from memory (useful for testing or resource management)."""
    global _model, _tokenizer
    _model = None
    _tokenizer = None


def _generate_raw(
    messages: list[dict[str, str]],
) -> tuple[str, int, int]:
    """Apply chat template, run greedy inference, return (generated_text, n_prompt_tokens, n_new_tokens).

    Scores ONLY the newly generated tokens; the prompt is never counted as
    model output.
    """
    import torch

    text_input = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = _tokenizer(text_input, return_tensors="pt")
    input_ids = inputs["input_ids"]
    n_prompt = input_ids.shape[1]

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,          # greedy — deterministic
        )

    # Slice off the prompt tokens so we score only new tokens
    new_ids = output_ids[0][n_prompt:]
    generated_text = _tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return generated_text, n_prompt, len(new_ids)


# Checks that trigger fallback if failed.  Advisory checks (like
# unsupported_novelty) are logged for human review but do not trigger fallback.
HARD_FIDELITY_CHECKS = {"numerical_fidelity", "causal_language", "limitation_preserved"}


def _validate(record: dict[str, Any], explanation: str) -> tuple[bool, str]:
    """Run fidelity checks on the explanation.  Returns (passed, detail).

    Only hard checks (numerical fidelity, causal language, limitation preservation)
    trigger fallback. Advisory checks are logged for audit/review.
    """
    from eval.fidelity_checks import run_all_checks

    checks = run_all_checks(record, explanation)
    failures = [
        f"{name}: {r.detail}"
        for name, r in checks.items()
        if name in HARD_FIDELITY_CHECKS and not r.passed
    ]
    if failures:
        return False, "; ".join(failures)

    advisory_warnings = [
        f"{name}: {r.detail}"
        for name, r in checks.items()
        if name not in HARD_FIDELITY_CHECKS and not r.passed
    ]
    if advisory_warnings:
        logger.warning(
            "Explainer fidelity advisory for finding %s: %s",
            record.get("finding_id", "unknown"),
            "; ".join(advisory_warnings),
        )

    return True, "All checks passed."


def explain(
    record: dict[str, Any],
    recommendation: str | None = None,
) -> ExplainerResult:
    """Translate one Explainer record into a human-readable explanation.

    Parameters
    ----------
    record:
        An Explainer record dict (schema_version, finding_id, source,
        category, kind, message, attributes, limitations).
    recommendation:
        Optional recommendation string from the analysis engine.
        Pass only when one genuinely exists; do NOT invent one.

    Returns
    -------
    ExplainerResult
        ``.text`` is the explanation or deterministic fallback.
        ``.fallback_used`` is True when validation failed.
    """
    finding_id = record.get("finding_id", "")
    fallback_text = record.get("message", "")

    if not is_enabled():
        logger.debug("Explainer LLM disabled; returning fallback for %s", finding_id)
        return ExplainerResult(
            text=fallback_text,
            fallback_used=True,
            validation_detail="Explainer LLM disabled via environment variable.",
            finding_id=finding_id,
        )

    # Lazy load
    if _model is None:
        try:
            load_model()
        except (ImportError, FileNotFoundError) as exc:
            logger.warning("Explainer LLM unavailable: %s", exc)
            return ExplainerResult(
                text=fallback_text,
                fallback_used=True,
                validation_detail=str(exc),
                finding_id=finding_id,
            )

    messages = build_chat_messages(record, recommendation)
    try:
        generated_text, n_prompt, n_new = _generate_raw(messages)
    except Exception as exc:  # noqa: BLE001
        logger.error("Inference failed for %s: %s", finding_id, exc)
        return ExplainerResult(
            text=fallback_text,
            fallback_used=True,
            validation_detail=f"Inference error: {exc}",
            finding_id=finding_id,
        )

    passed, detail = _validate(record, generated_text)
    if not passed:
        logger.warning(
            "Fidelity validation failed for %s (%s); using fallback.",
            finding_id,
            detail,
        )
        return ExplainerResult(
            text=fallback_text,
            fallback_used=True,
            validation_detail=detail,
            finding_id=finding_id,
            prompt_token_count=n_prompt,
            generated_token_count=n_new,
        )

    return ExplainerResult(
        text=generated_text,
        fallback_used=False,
        validation_detail=detail,
        finding_id=finding_id,
        prompt_token_count=n_prompt,
        generated_token_count=n_new,
    )
