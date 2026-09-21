"""
build_parse_eval_set.py
------------------------
Builds a PARSE-shaped eval set for the Explainer LLM from what
DataUnderstanding.profile() and DataCleaner.detect() actually produce
today -- not from Cochrane/SciTLDR/No Robots, none of which contain
anything in PARSE's own finding format. See Task 24.

Runs against a synthetic dataset with deliberately planted issues
(missing values, mixed types, outliers, skew, duplicates) by default,
so this works without any real plant data. Pass --data to run against
a real file instead (CSV/XLSX) -- nothing here writes or modifies the
source, same as DataUnderstanding/DataCleaner themselves.

Usage
-----
  python eval/build_parse_eval_set.py
  python eval/build_parse_eval_set.py --data path/to/real_data.xlsx --out eval/parse_eval_set.json
"""

import argparse
import json

import numpy as np
import pandas as pd

from parse.eda import DataUnderstanding
from parse.cleaning import DataCleaner


def _synthetic_dataset() -> pd.DataFrame:
    """A small dataset with deliberately planted, known issues, so the
    eval set doesn't depend on real plant data being present."""
    rng = np.random.default_rng(42)
    n = 60
    values = list(rng.normal(100, 10, n))
    values[0], values[1] = 500, -300  # obvious IQR outliers
    mixed = [str(v) if i % 5 else "unknown" for i, v in enumerate(rng.normal(50, 5, n))]
    skewed = list(rng.exponential(2, n))  # strongly right-skewed
    frame = pd.DataFrame(
        {
            "measurement": values,
            "mixed_column": mixed,
            "skewed_value": skewed,
            "record_id": range(1, n + 1),
            "group_flag": rng.choice(["A", "B"], n),
        }
    )
    frame.loc[5:9, "measurement"] = np.nan  # planted missingness
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)  # planted duplicate row
    return frame


def _eda_findings_to_eval_examples(eda_result) -> list[dict]:
    examples = []
    for finding in eda_result.findings:
        examples.append(
            {
                "finding_id": finding.finding_id,
                "source": "eda",
                "category": finding.category,
                "kind": finding.kind,
                "message": finding.message,
                "attributes": finding.attributes,
                "limitations": list(finding.limitations),
            }
        )
    return examples


def _cleaning_issues_to_eval_examples(issues) -> list[dict]:
    examples = []
    for issue in issues:
        examples.append(
            {
                "finding_id": issue.issue_id,
                "source": "cleaning_issue",
                "category": issue.kind,
                "kind": "observation",  # CleaningIssue has no separate interpretation/observation split
                "message": issue.message,
                "attributes": {},  # CleaningIssue carries no separate numeric-attributes dict;
                                    # any numbers already live in `message` and are still checked there.
                "limitations": list(issue.limitations),
            }
        )
    return examples


def build_eval_set(frame: pd.DataFrame) -> list[dict]:
    eda_result = DataUnderstanding().profile(frame)
    issues, _proposals = DataCleaner().detect(frame, eda_result=eda_result)

    examples = _eda_findings_to_eval_examples(eda_result)
    examples += _cleaning_issues_to_eval_examples(issues)
    return examples


def main():
    parser = argparse.ArgumentParser(description="Build a PARSE-shaped eval set for the Explainer LLM.")
    parser.add_argument("--data", default=None, help="Path to a real CSV/XLSX. Defaults to synthetic data.")
    parser.add_argument("--out", default="eval/parse_eval_set.json")
    args = parser.parse_args()

    if args.data:
        frame = pd.read_csv(args.data) if args.data.endswith(".csv") else pd.read_excel(args.data)
    else:
        frame = _synthetic_dataset()

    examples = build_eval_set(frame)

    with open(args.out, "w") as f:
        json.dump(examples, f, indent=2, default=str)

    print(f"Wrote {len(examples)} PARSE-shaped eval example(s) to {args.out}")


if __name__ == "__main__":
    main()
