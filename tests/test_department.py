"""Regression tests for the Department protocol and its BF implementation.

These exist to prevent the exact bug found by independent verification of
Task 15: BlastFurnaceDepartment initially duplicated feature_processing.py's
logic instead of delegating to it, which would have let the two silently
drift apart. Every assertion here ties BlastFurnaceDepartment's output back
to feature_processing's real functions/constants, not to a re-typed literal.
"""

import pandas as pd

from departments.blast_furnace import feature_processing as fp
from departments.blast_furnace.department import BlastFurnaceDepartment
from parse.core.department import Department


def test_blast_furnace_department_satisfies_protocol():
    assert isinstance(BlastFurnaceDepartment(), Department)


def test_feature_and_target_columns_match_feature_processing_constants():
    department = BlastFurnaceDepartment()
    assert department.feature_columns == fp.CHEM_COLS
    assert department.target_columns == fp.TARGET_COLS


def test_value_ranges_match_feature_processing_constants():
    department = BlastFurnaceDepartment()
    expected = {
        **fp.CHEM_RANGES,
        **fp.ATM_RANGES,
        **fp.BURDEN_RANGES,
        **fp.INTERACTION_RANGES,
    }
    assert department.value_ranges == expected


def test_parse_custom_fields_matches_feature_processing_parsers():
    department = BlastFurnaceDepartment()
    df = pd.DataFrame({
        "Test Condition": ["CO= 36%, H2=4% & N2=60%", "CO=40%, N2=60%"],
        "Burden compositon": ["70% S + 30% O", "100% P"],
    })

    result = department.parse_custom_fields(df)

    pd.testing.assert_frame_equal(
        result["atmosphere"], fp._parse_atmosphere(df["Test Condition"])
    )
    pd.testing.assert_frame_equal(
        result["burden"], fp._encode_burden_numeric(df["Burden compositon"])
    )


def test_parse_custom_fields_omits_groups_for_missing_columns():
    department = BlastFurnaceDepartment()
    df = pd.DataFrame({"Test Condition": ["CO=40%, N2=60%"]})

    result = department.parse_custom_fields(df)

    assert "atmosphere" in result
    assert "burden" not in result