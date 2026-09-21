"""Smoke test for eval/build_parse_eval_set.py -- confirms it runs against
the synthetic dataset and produces PARSE-shaped examples with the fields
the fidelity checkers and comparison runner expect.
"""

from eval.build_parse_eval_set import _synthetic_dataset, build_eval_set


def test_build_eval_set_produces_expected_shape():
    frame = _synthetic_dataset()
    examples = build_eval_set(frame)

    assert len(examples) > 0
    for example in examples:
        assert set(example.keys()) >= {"finding_id", "source", "category", "kind", "message", "attributes", "limitations"}
        assert isinstance(example["message"], str) and example["message"]
        assert isinstance(example["attributes"], dict)
        assert isinstance(example["limitations"], list)


def test_build_eval_set_catches_the_planted_issues():
    frame = _synthetic_dataset()
    examples = build_eval_set(frame)
    finding_ids = {example["finding_id"] for example in examples}

    # These correspond to the issues _synthetic_dataset() deliberately plants.
    assert "duplicate_rows" in finding_ids
    assert "missing_measurement" in finding_ids
    assert "outliers_measurement" in finding_ids
    assert "skew_skewed_value" in finding_ids
