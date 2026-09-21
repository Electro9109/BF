"""Unit tests for eval/fidelity_checks.py against hand-built pass/fail
pairs -- these are the ground truth the whole eval harness rests on,
per Task 24's testing requirement.
"""

from eval.fidelity_checks import (
    numerical_fidelity,
    causal_language_check,
    limitation_preserved,
    run_all_checks,
)


# ── numerical_fidelity ──────────────────────────────────────────────────

def test_numerical_fidelity_passes_when_numbers_preserved():
    result = numerical_fidelity(
        "Column 'x' contains 3 IQR-based statistical outlier(s).",
        {"column": "x", "count": 3},
        "The x column has 3 values that are statistical outliers.",
    )
    assert result.passed


def test_numerical_fidelity_fails_when_number_changed():
    result = numerical_fidelity(
        "Column 'x' contains 3 IQR-based statistical outlier(s).",
        {"column": "x", "count": 3},
        "The x column has 5 values that are statistical outliers.",
    )
    assert not result.passed
    assert "3" in result.detail


def test_numerical_fidelity_fails_when_number_dropped():
    result = numerical_fidelity(
        "Column 'x' has 5 missing value(s).",
        {"column": "x", "missing_count": 5},
        "The x column has some missing values.",
    )
    assert not result.passed


def test_numerical_fidelity_passes_trivially_with_no_source_numbers():
    result = numerical_fidelity("Column 'x' may be a target variable.", {"column": "x"}, "x looks like an outcome.")
    assert result.passed


def test_numerical_fidelity_handles_percentage_values():
    result = numerical_fidelity(
        "Column 'x' mixes numeric-like and non-numeric values (80% parse as numbers).",
        {"numeric_parse_fraction": 0.8},
        "About 80% of the values in x parse as numbers.",
    )
    assert result.passed


# ── causal_language_check ───────────────────────────────────────────────

def test_causal_language_passes_with_no_causal_phrasing():
    result = causal_language_check(
        "Column 'temp' may be a target or outcome variable.", "interpretation",
        "The temperature column appears to be an outcome variable.",
    )
    assert result.passed


def test_causal_language_fails_when_introduced_for_interpretation_finding():
    result = causal_language_check(
        "Column 'temp' may be a target or outcome variable.", "interpretation",
        "The temperature column causes the outcome to change.",
    )
    assert not result.passed
    assert "causal" in result.detail.lower()


def test_causal_language_fails_when_introduced_for_associational_message():
    result = causal_language_check(
        "Feature X is associated with output Y.", "observation",
        "Feature X leads to changes in output Y.",
    )
    assert not result.passed


def test_causal_language_passes_when_source_already_states_causality():
    result = causal_language_check(
        "The controlled experiment shows X causes Y.", "observation",
        "X causes Y, as shown by the controlled experiment.",
    )
    assert result.passed


# ── limitation_preserved ────────────────────────────────────────────────

def test_limitation_preserved_passes_with_no_limitations_to_check():
    result = limitation_preserved((), "Any explanation text.")
    assert result.passed


def test_limitation_preserved_passes_when_content_carries_over():
    result = limitation_preserved(
        ("A statistical outlier is not automatically a data error.",),
        "These are statistical outliers, but that doesn't automatically mean there's an error in the data.",
    )
    assert result.passed


def test_limitation_preserved_fails_when_dropped():
    result = limitation_preserved(
        ("A statistical outlier is not automatically a data error.",),
        "There are some unusual values in this column.",
    )
    assert not result.passed


# ── run_all_checks ───────────────────────────────────────────────────────

def test_run_all_checks_returns_all_three():
    finding = {
        "message": "Column 'x' contains 3 IQR-based statistical outlier(s).",
        "attributes": {"column": "x", "count": 3},
        "limitations": ["A statistical outlier is not automatically a data error."],
        "kind": "observation",
    }
    explanation = (
        "The x column has 3 unusual values that stand out statistically. "
        "That doesn't automatically mean there's an error in the data."
    )
    results = run_all_checks(finding, explanation)
    assert set(results.keys()) == {"numerical_fidelity", "causal_language", "limitation_preserved"}
    assert all(r.passed for r in results.values())
