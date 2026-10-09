"""Explainer adapter for PARSE dataset analysis findings.

Emits the exact record shape expected by the Explainer LLM eval and training harness:
    {
        "schema_version": "1.0",
        "finding_id": str,
        "source": str,
        "category": str,
        "kind": str,
        "message": str,
        "attributes": dict,
        "limitations": list
    }

Preserves legacy finding IDs and categories so existing Explainer models and eval
tests (e.g. tests/test_build_parse_eval_set.py) remain 100% compatible.
"""

from __future__ import annotations

from typing import Any, Sequence

from parse.analysis import EDAResult, Finding
from parse.cleaning import CleaningIssue

SCHEMA_VERSION = "1.0"


def finding_to_explainer_example(finding: Finding) -> dict[str, Any]:
    """Convert a modern canonical Finding into an Explainer-shaped record.

    Maps modern ID prefixes to legacy format where required by the Explainer contract:
      - 'unusual:<col>' -> 'outliers_<col>' (category 'statistical', kind 'observation')
      - 'skew:<col>' -> 'skew_<col>' (category 'statistical', kind 'observation')
      - 'missing:<col>' -> 'missing_<col>' (category 'quality', kind 'observation')
      - 'sparse:<col>' -> 'sparse_<col>' (category 'quality', kind 'observation')
      - 'mixed_values:<col>' -> 'mixed_values_<col>' (category 'quality', kind 'observation')
      - 'relationship:<left>:<right>' -> 'relationship_<left>_<right>' (category 'relationship', kind 'observation')
      - 'association:<left>:<right>' -> 'association_<left>_<right>' (category 'relationship', kind 'observation')
      - 'group_difference:<cat>:<num>' -> 'group_difference_<cat>_<num>' (category 'relationship', kind 'observation')
      - 'role:<col>' -> 'identifier_<col>' (category 'logical', kind 'interpretation')
      - 'temporal:<col>' -> 'temporal_<col>' (category 'logical', kind 'interpretation')
    """
    fid = finding.finding_id
    cat = finding.category
    kind = "observation"

    # Normalize finding_id and categories for legacy consumer contract
    if fid.startswith("unusual:"):
        col = fid.split(":", 1)[1]
        fid = f"outliers_{col}"
        cat = "statistical"
    elif fid.startswith("skew:"):
        col = fid.split(":", 1)[1]
        fid = f"skew_{col}"
        cat = "statistical"
    elif fid.startswith("missing:"):
        col = fid.split(":", 1)[1]
        fid = f"missing_{col}"
    elif fid.startswith("sparse:"):
        col = fid.split(":", 1)[1]
        fid = f"sparse_{col}"
    elif fid.startswith("mixed_values:"):
        col = fid.split(":", 1)[1]
        fid = f"mixed_values_{col}"
    elif fid.startswith("relationship:"):
        parts = fid.split(":")
        fid = f"relationship_{parts[1]}_{parts[2]}"
    elif fid.startswith("association:"):
        parts = fid.split(":")
        fid = f"association_{parts[1]}_{parts[2]}"
    elif fid.startswith("group_difference:"):
        parts = fid.split(":")
        fid = f"group_difference_{parts[1]}_{parts[2]}"
        cat = "relationship"
    elif fid.startswith("role:"):
        col = fid.split(":", 1)[1]
        fid = f"identifier_{col}"
        cat = "logical"
        kind = "interpretation"
    elif fid.startswith("temporal:"):
        col = fid.split(":", 1)[1]
        fid = f"temporal_{col}"
        cat = "logical"
        kind = "interpretation"
    elif fid.startswith("target:"):
        col = fid.split(":", 1)[1]
        fid = f"target_{col}"
        cat = "logical"
        kind = "interpretation"
    elif fid.startswith("group:"):
        col = fid.split(":", 1)[1]
        fid = f"group_{col}"
        cat = "logical"
        kind = "interpretation"

    # Merge finding attributes from result dict so numbers survive for fidelity check
    attributes: dict[str, Any] = dict(finding.result)
    if isinstance(finding.subject, str):
        attributes.setdefault("column", finding.subject)

    message = finding.observation
    if fid.startswith("missing_") and "missing_count" in attributes:
        col = attributes.get("column", finding.subject)
        cnt = attributes["missing_count"]
        message = f"Column '{col}' has {cnt} missing value(s)."
        # In legacy EDA, missing_fraction was in attributes; only keep missing_count if fraction not in message
        if "missing_rate" in attributes:
            attributes.pop("missing_rate", None)
    elif fid.startswith("outliers_") and "count" in attributes:
        col = attributes.get("column", finding.subject)
        cnt = attributes["count"]
        message = f"Column '{col}' contains {cnt} IQR-based statistical outlier(s)."
        attributes.pop("mad_count", None)
    elif fid.startswith("skew_") and "skewness" in attributes:
        col = attributes.get("column", finding.subject)
        sk = round(float(attributes["skewness"]), 2)
        attributes["skewness"] = sk
        message = f"Column '{col}' is strongly skewed (skewness {sk:.2f})."
    elif fid.startswith("mixed_values_") and "numeric_parse_fraction" in attributes:
        col = attributes.get("column", finding.subject)
        npf = round(float(attributes["numeric_parse_fraction"]), 2)
        attributes["numeric_parse_fraction"] = npf
        message = f"Column '{col}' mixes numeric-like and non-numeric values ({npf:.0%} parse as numbers)."
    elif fid.startswith("relationship_"):
        method = attributes.get("selected_method", "pearson")
        corr_val = attributes.get(method, attributes.get("pearson", 0.0))
        left, right = finding.subject if isinstance(finding.subject, tuple) else (None, None)
        attributes.setdefault("left", left)
        attributes.setdefault("right", right)
        message = f"'{left}' and '{right}' have a strong observed {method} correlation ({corr_val:.3f})."
    elif fid.startswith("association_"):
        v = float(attributes.get("cramers_v", 0.0))
        left, right = finding.subject if isinstance(finding.subject, tuple) else (None, None)
        attributes.setdefault("left", left)
        attributes.setdefault("right", right)
        message = f"'{left}' and '{right}' show an observed categorical association (Cramer's V {v:.3f})."
    elif fid.startswith("group_difference_"):
        spread = float(attributes.get("mean_spread", 0.0))
        cat_attr, num_attr = finding.subject if isinstance(finding.subject, tuple) else (None, None)
        attributes.setdefault("category", cat_attr)
        attributes.setdefault("numeric", num_attr)
        message = f"Mean '{num_attr}' differs across '{cat_attr}' groups (spread {spread:.2f})."

    return {
        "schema_version": SCHEMA_VERSION,
        "finding_id": fid,
        "source": "eda",
        "category": cat,
        "kind": kind,
        "message": message,
        "attributes": attributes,
        "limitations": list(finding.limitations),
    }


def eda_result_to_explainer_examples(eda_result: EDAResult) -> list[dict[str, Any]]:
    """Convert all findings from a canonical EDAResult to Explainer records."""
    return [finding_to_explainer_example(f) for f in eda_result.findings]


def cleaning_issues_to_explainer_examples(issues: Sequence[CleaningIssue]) -> list[dict[str, Any]]:
    """Convert cleaning issues to Explainer records with schema version."""
    examples = []
    for issue in issues:
        fid = issue.issue_id
        if fid.startswith("unusual:"):
            fid = f"outliers_{fid.split(':', 1)[1]}"
        examples.append(
            {
                "schema_version": SCHEMA_VERSION,
                "finding_id": fid,
                "source": "cleaning_issue",
                "category": issue.kind,
                "kind": "observation",
                "message": issue.message,
                "attributes": {},
                "limitations": list(issue.limitations),
            }
        )
    return examples
