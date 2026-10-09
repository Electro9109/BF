"""
run_comparison.py
-------------------
Runs a PARSE-shaped eval set (built by build_parse_eval_set.py) through
both the base Qwen3-4B-Instruct-2507 model and the fine-tuned v0.3 adapter,
scoring each with the automated fidelity checks, side by side.

Usage
-----
  python eval/run_comparison.py \\
      --eval-set eval/parse_eval_set.json \\
      --base-model Qwen/Qwen3-4B-Instruct-2507 \\
      --adapter-path parse/models/qwen3-explainer-v0.3 \\
      --out eval/comparison_results.json

  # Smoke-test the harness without any model:
  python eval/run_comparison.py --dry-run

Design constraints
------------------
- Uses the SAME prompt builder (parse.explainer_prompt) as production.
  There is no separate prompt format for evaluation.
- Uses Qwen's chat template via tokenizer.apply_chat_template().
- Uses greedy decoding (do_sample=False) for determinism.
- Scores ONLY newly generated tokens.  The prompt is sliced off before
  scoring so the harness never grades the input as model output.
"""

import argparse
import json
from typing import Callable

from eval.fidelity_checks import run_all_checks
from parse.explainer_prompt import build_chat_messages


def _build_generator_fn(model_id: str, adapter_path: str | None = None) -> Callable[[dict], str]:
    """Return a finding-dict -> explanation-text function.

    transformers/peft are lazy-imported so that ``--dry-run`` mode (or unit
    tests that import this module) never require them.

    The generator:
    1. Uses build_chat_messages() (same as production) to format the prompt.
    2. Applies the tokenizer's chat template.
    3. Runs greedy decoding (do_sample=False).
    4. Slices off the prompt tokens before decoding so that only newly
       generated text is returned (prompt-echo protection).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    def generate(finding: dict) -> str:
        import torch

        # Use the shared prompt builder — same format as production
        messages = build_chat_messages(finding)

        text_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text_input, return_tensors="pt")
        n_prompt = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,      # greedy — deterministic
            )

        # Slice off prompt tokens; score only newly generated tokens
        new_ids = output_ids[0][n_prompt:]
        return tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    return generate


def build_generator(model_id: str, adapter_path: str | None = None) -> Callable[[dict], str]:
    """Public entry point for building a generator function.

    Kept as an alias so external callers and existing tests continue to work.
    """
    return _build_generator_fn(model_id, adapter_path)


HARD_CHECKS = {"numerical_fidelity", "causal_language", "limitation_preserved"}


def score_examples(examples: list[dict], generate: Callable[[dict], str]) -> list[dict]:
    """Run ``generate`` on every finding and score the explanation."""
    results = []
    for finding in examples:
        explanation = generate(finding)
        checks = run_all_checks(finding, explanation)
        results.append(
            {
                "finding_id": finding["finding_id"],
                "explanation": explanation,
                "checks": {
                    name: {"passed": r.passed, "detail": r.detail}
                    for name, r in checks.items()
                },
                "all_passed": all(
                    r.passed for name, r in checks.items() if name in HARD_CHECKS
                ),
            }
        )
    return results


def summarize(results: list[dict]) -> dict:
    total = len(results)
    if total == 0:
        return {"total": 0}
    summary = {
        "total": total,
        "all_passed_rate": sum(r["all_passed"] for r in results) / total,
    }
    for check_name in (
        "numerical_fidelity",
        "causal_language",
        "limitation_preserved",
        "unsupported_novelty",
    ):
        if results and check_name in results[0]["checks"]:
            passed = sum(r["checks"][check_name]["passed"] for r in results)
            summary[f"{check_name}_pass_rate"] = passed / total
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Compare base vs. fine-tuned Explainer LLM on a PARSE eval set."
    )
    parser.add_argument("--eval-set", default="eval/parse_eval_set.json")
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--adapter-path",
        default="parse/models/qwen3-explainer-v0.3",
        help="Path to the local v0.3 LoRA adapter directory.",
    )
    parser.add_argument("--out", default="eval/comparison_results.json")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Skip model loading entirely; echoes the finding message back as "
            "a stand-in 'explanation', to smoke-test the harness plumbing "
            "without any model installed."
        ),
    )
    args = parser.parse_args()

    with open(args.eval_set) as f:
        examples = json.load(f)

    if args.dry_run:
        # Echo the finding message — never echoes the full prompt, only the message field.
        # This correctly simulates the 'generated text only' slice.
        generate_base = generate_finetuned = lambda finding: finding["message"]
    else:
        generate_base = build_generator(args.base_model)
        generate_finetuned = build_generator(args.base_model, adapter_path=args.adapter_path)

    base_results = score_examples(examples, generate_base)
    finetuned_results = score_examples(examples, generate_finetuned)

    output = {
        "base": {"results": base_results, "summary": summarize(base_results)},
        "finetuned": {"results": finetuned_results, "summary": summarize(finetuned_results)},
    }

    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Base model summary:       {output['base']['summary']}")
    print(f"Fine-tuned model summary: {output['finetuned']['summary']}")
    print(f"Full results written to   {args.out}")


if __name__ == "__main__":
    main()
