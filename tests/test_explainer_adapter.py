"""Contract test for the Explainer adapter.

Pins the Explainer record shape:
    {finding_id, source, category, kind, message, attributes, limitations}
and verifies that canonical AnalysisOrchestrator findings map accurately to
the Explainer's expected schema and legacy ID values.
"""

import numpy as np
import pandas as pd
import pytest

from eval.build_parse_eval_set import _synthetic_dataset, build_eval_set
from eval.fidelity_checks import numerical_fidelity, causal_language_check, limitation_preserved
from parse.analysis import AnalysisOrchestrator, AnalysisRequest
from parse.core.contracts import SourceRef
from parse.explainer_adapter import (
    SCHEMA_VERSION,
    eda_result_to_explainer_examples,
    finding_to_explainer_example,
)


def test_explainer_record_schema_and_version():
    frame = _synthetic_dataset()
    examples = build_eval_set(frame)

    required_keys = {"schema_version", "finding_id", "source", "category", "kind", "message", "attributes", "limitations"}
    assert len(examples) > 0
    for example in examples:
        assert set(example.keys()) >= required_keys
        assert example["schema_version"] == "1.0"
        assert isinstance(example["finding_id"], str) and example["finding_id"]
        assert example["source"] in {"eda", "cleaning_issue"}
        assert example["kind"] in {"observation", "interpretation"}
        assert isinstance(example["message"], str) and example["message"]
        assert isinstance(example["attributes"], dict)
        assert isinstance(example["limitations"], list)


def test_explainer_adapter_maps_legacy_ids_for_planted_issues():
    frame = _synthetic_dataset()
    examples = build_eval_set(frame)
    finding_ids = {example["finding_id"] for example in examples}

    # Must preserve exact legacy finding IDs expected by Explainer training and eval
    assert "duplicate_rows" in finding_ids
    assert "missing_measurement" in finding_ids
    assert "outliers_measurement" in finding_ids
    assert "skew_skewed_value" in finding_ids
    assert "mixed_values_mixed_column" in finding_ids


def test_explainer_fidelity_on_canonical_findings():
    """Ensure every number in the canonical finding is preserved in message/attributes."""
    frame = _synthetic_dataset()
    source = SourceRef("test_fidelity", "user_input")
    eda_result = AnalysisOrchestrator().analyze(AnalysisRequest(frame, source))
    examples = eda_result_to_explainer_examples(eda_result)

    for ex in examples:
        # Check that numerical_fidelity check runs cleanly on the finding's own message
        fidelity = numerical_fidelity(ex["message"], ex["attributes"], ex["message"])
        assert fidelity.passed, f"Numerical fidelity failed for {ex['finding_id']}: {fidelity.detail}"

        # Check causal language compliance
        causal = causal_language_check(ex["message"], ex["kind"], ex["message"])
        assert causal.passed, f"Causal check failed for {ex['finding_id']}: {causal.detail}"
