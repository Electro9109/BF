"""
run_comparison.py
-------------------
Runs a PARSE-shaped eval set (built by build_parse_eval_set.py) through
both the base Qwen3-4B-Instruct-2507 model and the fine-tuned adapter,
scoring each with the automated fidelity checks, side by side -- the
training plan's own stated bar ("Cochrane performance alone isn't
enough").

This does NOT run the comparison itself as part of Task 24 -- it's the
harness, ready to point at the Kaggle adapter once it's downloaded. See
Task 24's scope notes.

Usage
-----
  python eval/run_comparison.py \\
      --eval-set eval/parse_eval_set.json \\
      --base-model Qwen/Qwen3-4B-Instruct-2507 \\
      --adapter-path /path/to/downloaded/adapter \\
      --out eval/comparison_results.json

Generator functions are pluggable (see build_generator()) so this can
run against a local transformers pipeline, an API endpoint, or a stub
for dry-testing the harness itself without any model loaded.
"""

import argparse
import json
from typing import Callable

from eval.fidelity_checks import run_all_checks


PROMPT_TEMPLATE = (
    "Explain the following technical finding clearly to a non-specialist. "
    "Preserve all numbers, the relationship direction, and any stated "
    "limitations exactly. Do not turn an association into a causal claim. "
    "Do not invent facts, causes, or recommendations not present below.\n\n"
    "Finding: {message}\n"
    "Details: {attributes}\n"
    "Limitations: {limitations}"
)


def build_prompt(finding: dict) -> str:
    return PROMPT_TEMPLATE.format(
        message=finding["message"],
        attributes=finding.get("attributes", {}),
        limitations=finding.get("limitations", []),
    )


def build_generator(model_id: str, adapter_path: str | None = None) -> Callable[[str], str]:
    """Returns a prompt -> text function for the given model, loading it lazily.

    transformers/peft are heavy optional dependencies -- only imported if
    this is actually called, so importing this module (e.g. for --dry-run)
    never requires them installed.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

    def generate(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=256)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    return generate


def score_examples(examples: list[dict], generate: Callable[[str], str]) -> list[dict]:
    results = []
    for finding in examples:
        explanation = generate(build_prompt(finding))
        checks = run_all_checks(finding, explanation)
        results.append(
            {
                "finding_id": finding["finding_id"],
                "explanation": explanation,
                "checks": {name: {"passed": r.passed, "detail": r.detail} for name, r in checks.items()},
                "all_passed": all(r.passed for r in checks.values()),
            }
        )
    return results


def summarize(results: list[dict]) -> dict:
    total = len(results)
    if total == 0:
        return {"total": 0}
    summary = {"total": total, "all_passed_rate": sum(r["all_passed"] for r in results) / total}
    for check_name in ("numerical_fidelity", "causal_language", "limitation_preserved"):
        passed = sum(r["checks"][check_name]["passed"] for r in results)
        summary[f"{check_name}_pass_rate"] = passed / total
    return summary


def main():
    parser = argparse.ArgumentParser(description="Compare base vs. fine-tuned Explainer LLM on a PARSE eval set.")
    parser.add_argument("--eval-set", default="eval/parse_eval_set.json")
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter-path", default=None, help="Path to the downloaded Kaggle LoRA adapter")
    parser.add_argument("--out", default="eval/comparison_results.json")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip model loading entirely; echoes the finding message back as a stand-in "
        "'explanation', to smoke-test the harness plumbing without any model installed.",
    )
    args = parser.parse_args()

    with open(args.eval_set) as f:
        examples = json.load(f)

    if args.dry_run:
        generate_base = generate_finetuned = lambda prompt: prompt.split("Finding: ")[-1].split("\n")[0]
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

    print(f"Base model summary:      {output['base']['summary']}")
    print(f"Fine-tuned model summary: {output['finetuned']['summary']}")
    print(f"Full results written to {args.out}")


if __name__ == "__main__":
    main()
