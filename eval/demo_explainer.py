#!/usr/bin/env python3
"""demo_explainer.py
-------------------
End-to-end demonstration of the PARSE Qwen Explainer pipeline.

Shows how raw data is analyzed by the canonical Analysis Engine, adapted into
explainer records, translated into natural language by the Explainer, validated
against fidelity checks, and protected by zero-downtime fallback.

Usage:
  python eval/demo_explainer.py
  python eval/demo_explainer.py --dry-run
  python eval/demo_explainer.py --max-examples 5
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import numpy as np

from parse.analysis import AnalysisOrchestrator, AnalysisRequest
from parse.core.contracts import SourceRef
from parse.explainer_adapter import eda_result_to_explainer_examples
from parse.explainer_llm import explain, is_enabled, ENABLED_ENV_VAR
from eval.fidelity_checks import run_all_checks


def create_sample_dataset() -> pd.DataFrame:
    """Create a sample metallurgical blast furnace dataset with planted features."""
    rng = np.random.default_rng(42)
    n = 60
    # Blast furnace temperature values with outliers
    temperatures = rng.normal(1450, 25, n).tolist()
    temperatures[3] = 1750.0  # Planted high outlier
    temperatures[12] = 1100.0  # Planted low outlier

    # Silicon content (wt %)
    silicon = rng.normal(0.45, 0.08, n).tolist()

    # Slag basicity (B2 = CaO/SiO2)
    basicity = rng.exponential(1.15, n).tolist()  # Skewed

    # Tuyere velocity mix
    velocity = [f"{v:.1f}" if i % 6 != 0 else "SENSOR_ERR" for i, v in enumerate(rng.normal(210, 10, n))]

    df = pd.DataFrame({
        "tuyere_temp_c": temperatures,
        "hot_metal_si_pct": silicon,
        "slag_basicity": basicity,
        "blast_velocity": velocity,
        "cast_id": range(1001, 1001 + n),
    })
    # Planted missing values
    df.loc[7:10, "hot_metal_si_pct"] = np.nan
    return df


def run_demo(dry_run: bool = False, max_examples: int = 5) -> None:
    print("=" * 80)
    print(" PARSE QWEN EXPLAINER: END-TO-END DEMO")
    print("=" * 80)

    if dry_run:
        print("[MODE] Dry-run enabled: model generation disabled (using deterministic fallback).")
        os.environ[ENABLED_ENV_VAR] = "0"

    print("\n1. Generating sample metallurgical dataset (Blast Furnace Process Data)...")
    df = create_sample_dataset()
    print(f"   Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"   Columns: {', '.join(df.columns)}")

    print("\n2. Running canonical Analysis Engine (AnalysisOrchestrator & DataCleaner)...")
    source = SourceRef("bf_sample", "user_input", label="Blast Furnace Sample")
    request = AnalysisRequest(df, source)
    orchestrator = AnalysisOrchestrator()
    eda_result = orchestrator.analyze(request)

    from parse.cleaning import DataCleaner
    from parse.explainer_adapter import cleaning_issues_to_explainer_examples
    issues, _ = DataCleaner().detect(df, eda_result=eda_result)
    print(f"   Analysis complete. Generated {len(eda_result.findings)} EDA findings and {len(issues)} cleaning issues.")

    print("\n3. Converting findings to Explainer records via parse.explainer_adapter...")
    records = eda_result_to_explainer_examples(eda_result)
    records += cleaning_issues_to_explainer_examples(issues)
    print(f"   Created {len(records)} standardized Explainer records.")

    records_to_show = records[:max_examples]
    print(f"\n4. Translating {len(records_to_show)} findings with Explainer LLM & verifying fidelity:\n")

    for idx, rec in enumerate(records_to_show, start=1):
        print("-" * 80)
        print(f"[{idx}/{len(records_to_show)}] Finding ID: {rec['finding_id']} ({rec['category']} / {rec['kind']})")
        print(f"Source Message: {rec['message']}")
        if rec.get("limitations"):
            print(f"Limitations   : {'; '.join(rec['limitations'])}")

        result = explain(rec)
        checks = run_all_checks(rec, result.text)

        status = "[FALLBACK]" if result.fallback_used else "[GENERATED]"
        print(f"\nResult {status}:")
        print(f"  {result.text}")

        print(f"\nExecution Metadata:")
        print(f"  Fallback Used        : {result.fallback_used}")
        print(f"  Prompt Tokens        : {result.prompt_token_count}")
        print(f"  Generated Tokens     : {result.generated_token_count}")
        if result.validation_detail:
            print(f"  Validation Detail    : {result.validation_detail}")

        print("Fidelity Checks:")
        for check_name, chk in checks.items():
            mark = "PASS" if chk.passed else ("FLAG" if check_name == "unsupported_novelty" else "FAIL")
            print(f"  [{mark}] {check_name:20s}: {chk.detail}")

    print("\n" + "=" * 80)
    print(" DEMO COMPLETED SUCCESSFULLY")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="PARSE Qwen Explainer End-to-End Demo")
    parser.add_argument("--dry-run", action="store_true", help="Force fallback mode without loading model weights")
    parser.add_argument("--max-examples", type=int, default=4, help="Maximum number of findings to demonstrate")
    args = parser.parse_args()

    run_demo(dry_run=args.dry_run, max_examples=args.max_examples)


if __name__ == "__main__":
    main()
