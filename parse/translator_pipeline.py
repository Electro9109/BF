"""Deterministic inference pipeline for the PARSE Results Translator.

Handles structured validation, prompt building, LLM execution (with greedy decoding),
and fidelity validation. Falls back gracefully when model weights are not loaded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from parse.core.translator_contract import StructuredModuleResult
from parse.explainer_llm import is_enabled, load_model
from parse.translator_prompt import build_translator_messages

logger = logging.getLogger(__name__)


@dataclass
class TranslationOutput:
    """Output envelope returned by the Results Translator."""

    text: str
    is_fallback: bool
    status: str
    metadata: dict[str, Any]
    warnings: list[str]


def deterministic_fallback_translation(result: StructuredModuleResult, reason: str = "") -> str:
    """Generate a clean, faithful text summary when LLM weights are unavailable or disabled."""
    sections = [
        f"### PARSE Analysis Summary — {result.department.replace('_', ' ').title()}",
        f"**Dataset**: `{result.dataset_id}` | **Operation**: {result.operation.replace('_', ' ').title()}",
        f"\n**Summary**: {result.summary}",
    ]

    if result.target:
        sections.append(f"- **Target Variable**: {result.target} ({result.units or 'units not specified'})")

    if result.statistics:
        stat_strs = [f"{k.replace('_', ' ')}: {v}" for k, v in result.statistics.items() if not isinstance(v, (dict, list))]
        if stat_strs:
            sections.append(f"- **Key Statistics**: {', '.join(stat_strs[:5])}")

    if result.findings:
        sections.append("\n#### Key Findings:")
        for idx, f in enumerate(result.findings[:5], 1):
            obs = f.get("observation", str(f))
            sections.append(f"{idx}. {obs}")

    if result.predictions:
        sections.append("\n#### Model Predictions:")
        for tgt, val in result.predictions.items():
            unit_str = f" {result.units}" if result.units else ""
            sections.append(f"- **{tgt}**: {val}{unit_str}")

    if result.evaluation_metrics:
        sections.append("\n#### Model Evaluation Metrics:")
        metrics = ", ".join(f"{k.upper()}={v}" for k, v in result.evaluation_metrics.items())
        sections.append(f"- {metrics}")

    if result.anomalies:
        sections.append("\n#### Detected Anomalies:")
        for a in result.anomalies[:3]:
            desc = a.get("description", str(a))
            sections.append(f"- ⚠️ {desc}")

    if result.recommendations:
        sections.append("\n#### Upstream Recommendations:")
        for r in result.recommendations:
            sections.append(f"- 💡 {r}")
    else:
        sections.append("\n- *No upstream recommendations were provided by the analyser.*")

    if result.limitations:
        sections.append("\n#### Scope & Limitations:")
        for lim in result.limitations:
            sections.append(f"- ℹ️ {lim}")

    if reason:
        sections.append(f"\n*(Explanation generated via deterministic fallback: {reason})*")

    return "\n".join(sections)


class ResultsTranslator:
    """High-level Results Translator coordinating prompt generation and model execution."""

    def __init__(self, enabled: bool | None = None):
        self._enabled = enabled

    def is_enabled(self) -> bool:
        if self._enabled is not None:
            return bool(self._enabled) and is_enabled()
        return is_enabled()

    def translate(self, result: StructuredModuleResult) -> TranslationOutput:
        """Translate structured module results into an evidence-grounded explanation."""
        if not isinstance(result, StructuredModuleResult):
            raise TypeError(f"Expected StructuredModuleResult, got {type(result).__name__}")

        # Check for malformed or empty input
        if not result.summary and not result.findings and not result.statistics and not result.evaluation_metrics:
            return TranslationOutput(
                text="The supplied module result is empty and cannot be translated.",
                is_fallback=True,
                status="malformed_input",
                metadata={"dataset_id": result.dataset_id},
                warnings=["Supplied result contained no summary, findings, statistics, or metrics."],
            )

        # Check if LLM inference is enabled
        if not self.is_enabled():
            fallback_text = deterministic_fallback_translation(result, reason="PARSE_EXPLAINER_ENABLED=0")
            return TranslationOutput(
                text=fallback_text,
                is_fallback=True,
                status="fallback_disabled",
                metadata={"dataset_id": result.dataset_id, "department": result.department},
                warnings=[],
            )

        # Attempt to load model and tokenizer
        try:
            tokenizer, model = load_model()
        except Exception as exc:
            logger.warning("Could not load LLM model weights: %s. Using deterministic fallback.", exc)
            fallback_text = deterministic_fallback_translation(result, reason=f"Model unavailable ({exc})")
            return TranslationOutput(
                text=fallback_text,
                is_fallback=True,
                status="fallback_model_unavailable",
                metadata={"dataset_id": result.dataset_id, "department": result.department},
                warnings=[str(exc)],
            )

        messages = build_translator_messages(result)

        try:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(prompt_text, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            import torch
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,  # greedy deterministic decoding
                    pad_token_id=tokenizer.eos_token_id,
                )

            prompt_len = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][prompt_len:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            if not generated_text:
                raise ValueError("Model produced empty explanation.")

            # ── Fidelity Validation Gate ─────────────────────────────────────
            # Verify that generated text faithfully reflects the structured inputs
            from eval.fidelity_checks import (
                causal_language_check,
                limitation_preserved,
                numerical_fidelity,
                recommendation_fidelity,
            )

            # Flatten numbers/facts from statistics, predictions, and metrics
            combined_attributes: dict[str, Any] = {}
            if result.statistics:
                combined_attributes.update(result.statistics)
            if result.predictions:
                combined_attributes.update(result.predictions)
            if result.evaluation_metrics:
                combined_attributes.update(result.evaluation_metrics)

            source_facts_text = result.summary
            for f in result.findings:
                obs = f.get("observation", str(f))
                source_facts_text += f" {obs}"

            fidelity_warnings = []
            
            # 1. Numerical fidelity
            num_check = numerical_fidelity(source_facts_text, combined_attributes, generated_text)
            if not num_check.passed:
                fidelity_warnings.append(f"numerical_fidelity_failed: {num_check.detail}")

            # 2. Causal language check
            causal_check = causal_language_check(source_facts_text, "observation", generated_text)
            if not causal_check.passed:
                fidelity_warnings.append(f"causal_language_violation: {causal_check.detail}")

            # 3. Recommendation fidelity
            rec_str = " ".join(result.recommendations) if result.recommendations else None
            rec_check = recommendation_fidelity(rec_str, generated_text)
            if not rec_check.passed:
                fidelity_warnings.append(f"recommendation_fidelity_failed: {rec_check.detail}")

            # 4. Limitation preservation
            if result.limitations:
                lim_check = limitation_preserved(tuple(result.limitations), generated_text)
                if not lim_check.passed:
                    fidelity_warnings.append(f"limitation_omitted: {lim_check.detail}")

            if fidelity_warnings:
                logger.warning(
                    "Generated explanation failed fidelity checks: %s. Using deterministic fallback.",
                    fidelity_warnings,
                )
                fallback_text = deterministic_fallback_translation(
                    result,
                    reason=f"Fidelity validation failed ({'; '.join(fidelity_warnings)})",
                )
                return TranslationOutput(
                    text=fallback_text,
                    is_fallback=True,
                    status="fallback_fidelity_failed",
                    metadata={
                        "dataset_id": result.dataset_id,
                        "department": result.department,
                        "fidelity_warnings": fidelity_warnings,
                        "raw_generated_text": generated_text,
                    },
                    warnings=fidelity_warnings,
                )

            return TranslationOutput(
                text=generated_text,
                is_fallback=False,
                status="success",
                metadata={
                    "dataset_id": result.dataset_id,
                    "department": result.department,
                    "prompt_tokens": prompt_len,
                    "generated_tokens": len(generated_tokens),
                },
                warnings=[],
            )

        except Exception as exc:
            logger.warning("Generation failed: %s. Using deterministic fallback.", exc)
            fallback_text = deterministic_fallback_translation(result, reason=f"Inference error ({exc})")
            return TranslationOutput(
                text=fallback_text,
                is_fallback=True,
                status="fallback_inference_error",
                metadata={"dataset_id": result.dataset_id},
                warnings=[str(exc)],
            )
