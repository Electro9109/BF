"""Unit tests for cleaning_impact module (distribution shifts and information loss)."""

import numpy as np
import pandas as pd
import pytest

from parse.cleaning import ChangeRecord
from parse.cleaning_impact import (
    CATEGORICAL_TVD_THRESHOLD,
    RELATIVE_STD_SHIFT_THRESHOLD,
    compute_distribution_shifts,
    compute_ks_statistic,
    derive_information_loss_notes,
)


def test_compute_ks_statistic_identical():
    s1 = [1.0, 2.0, 3.0, 4.0, 5.0]
    s2 = [1.0, 2.0, 3.0, 4.0, 5.0]
    stat = compute_ks_statistic(s1, s2)
    assert stat == 0.0


def test_compute_ks_statistic_empty():
    assert compute_ks_statistic([], [1.0, 2.0]) == 0.0
    assert compute_ks_statistic([1.0], []) == 0.0


def test_compute_ks_statistic_hand_calculated():
    # s1 = [1, 2] -> ECDF1(1)=0.5, ECDF1(2)=1.0, ECDF1(3)=1.0, ECDF1(4)=1.0
    # s2 = [3, 4] -> ECDF2(1)=0.0, ECDF2(2)=0.0, ECDF2(3)=0.5, ECDF2(4)=1.0
    # At x=2: |0.5 - 0.0| = 0.5, at x=2.0 right search: cdf1=1.0, cdf2=0.0 -> diff=1.0!
    # Max diff is 1.0 because distributions are completely separated.
    s1 = [1.0, 2.0]
    s2 = [3.0, 4.0]
    assert compute_ks_statistic(s1, s2) == 1.0

    # s1 = [1, 2, 3, 4]
    # s2 = [1, 2, 2, 3]
    # Points: 1, 2, 3, 4
    # ECDF1: at 1 -> 1/4 = 0.25; at 2 -> 2/4 = 0.50; at 3 -> 3/4 = 0.75; at 4 -> 4/4 = 1.00
    # ECDF2: at 1 -> 1/4 = 0.25; at 2 -> 3/4 = 0.75; at 3 -> 4/4 = 1.00; at 4 -> 4/4 = 1.00
    # Differences:
    # at 1: |0.25 - 0.25| = 0.0
    # at 2: |0.50 - 0.75| = 0.25
    # at 3: |0.75 - 1.00| = 0.25
    # at 4: |1.00 - 1.00| = 0.0
    # Max diff = 0.25
    s3 = [1.0, 2.0, 3.0, 4.0]
    s4 = [1.0, 2.0, 2.0, 3.0]
    assert compute_ks_statistic(s3, s4) == 0.25


def test_compute_distribution_shifts_numeric():
    # Original: [10.0, 20.0, 30.0, np.nan] -> mean_before = 20.0, std_before = 10.0 (ddof=1)
    # Cleaned:  [10.0, 20.0, 30.0, 20.0]   -> mean_after = 20.0, std_after = 8.1650 (sqrt(200/3))
    df_orig = pd.DataFrame({"num": [10.0, 20.0, 30.0, np.nan]})
    df_clean = pd.DataFrame({"num": [10.0, 20.0, 30.0, 20.0]})

    shifts = compute_distribution_shifts(df_orig, df_clean)
    assert "num" in shifts
    shift = shifts["num"]
    assert shift["kind"] == "numeric"
    assert not shift["unchanged"]
    assert shift["sample_size_before"] == 3
    assert shift["sample_size_after"] == 4
    assert shift["mean_before"] == 20.0
    assert shift["mean_after"] == 20.0
    assert shift["mean_shift"] == 0.0
    assert shift["std_before"] == 10.0
    assert round(shift["std_after"], 4) == round(float(np.std([10.0, 20.0, 30.0, 20.0], ddof=1)), 4)
    assert "ks_statistic" in shift
    assert "ks_statistic_note" in shift


def test_compute_distribution_shifts_categorical():
    # Before: 2 'A', 2 'B' -> p(A)=0.5, p(B)=0.5
    # After:  3 'A', 1 'B' -> p(A)=0.75, p(B)=0.25
    # TVD = 0.5 * (|0.75 - 0.5| + |0.25 - 0.5|) = 0.5 * (0.25 + 0.25) = 0.25
    df_orig = pd.DataFrame({"cat": pd.Series(["A", "A", "B", "B"], dtype="category")})
    df_clean = pd.DataFrame({"cat": pd.Series(["A", "A", "A", "B"], dtype="category")})

    shifts = compute_distribution_shifts(df_orig, df_clean)
    assert "cat" in shifts
    shift = shifts["cat"]
    assert shift["kind"] == "categorical"
    assert not shift["unchanged"]
    assert shift["total_variation_distance"] == 0.25
    assert shift["categories_added"] == []
    assert shift["categories_removed"] == []
    assert shift["sample_size_before"] == 4
    assert shift["sample_size_after"] == 4


def test_compute_distribution_shifts_unchanged():
    df_orig = pd.DataFrame({"num": [1.0, 2.0, 3.0], "cat": ["X", "Y", "Z"]})
    df_clean = df_orig.copy()

    shifts = compute_distribution_shifts(df_orig, df_clean)
    assert shifts["num"]["unchanged"] is True
    assert shifts["num"]["mean_shift"] == 0.0
    assert shifts["num"]["std_shift"] == 0.0
    assert shifts["num"]["ks_statistic"] == 0.0

    assert shifts["cat"]["unchanged"] is True
    assert shifts["cat"]["total_variation_distance"] == 0.0


def test_compute_distribution_shifts_not_computed_text():
    # High cardinality free text
    long_strings = [f"Text comment number {i} with unique content" for i in range(50)]
    df_orig = pd.DataFrame({"text_col": long_strings})
    df_clean = df_orig.copy()

    shifts = compute_distribution_shifts(df_orig, df_clean)
    assert shifts["text_col"]["kind"] == "not_computed"
    assert "free-text" in shifts["text_col"]["reason"]
    assert shifts["text_col"]["unchanged"] is True


def test_derive_information_loss_notes_relative_std_shift():
    # std_before = 10.0, std_after = 8.0 -> std_shift = -2.0 -> abs(-2.0)/10.0 = 20% > 10%
    shifts = {
        "val": {
            "kind": "numeric",
            "std_before": 10.0,
            "std_after": 8.0,
            "std_shift": -2.0,
            "sample_size_before": 10,
            "sample_size_after": 10,
        }
    }
    changes = [ChangeRecord("p1", "i1", 0, "val", None, 20.0, "impute_missing")]
    notes = derive_information_loss_notes(shifts, changes)
    assert len(notes) == 1
    assert "standard deviation changed by 20.0%" in notes[0]


def test_derive_information_loss_notes_row_removal():
    # Column had 10 observations, now 9, and changes contain row-level removal (field is None)
    shifts = {
        "val": {
            "kind": "numeric",
            "std_before": 10.0,
            "std_after": 9.9,
            "std_shift": -0.1,  # 1% shift, below 10%
            "sample_size_before": 10,
            "sample_size_after": 9,
        }
    }
    # Row removal has field=None
    changes = [
        ChangeRecord("p1", "i1", 5, None, None, None, "remove_duplicates"),
        ChangeRecord("p2", "i2", 0, "val", None, 5.0, "impute_missing"),
    ]
    notes = derive_information_loss_notes(shifts, changes)
    assert len(notes) == 1
    assert "sample size reduced from 10 to 9 due to row removal" in notes[0]


def test_derive_information_loss_notes_categorical_tvd():
    shifts = {
        "cat": {
            "kind": "categorical",
            "total_variation_distance": 0.20,
            "sample_size_before": 10,
            "sample_size_after": 10,
        }
    }
    changes = [ChangeRecord("p1", "i1", 0, "cat", None, "A", "impute_missing")]
    notes = derive_information_loss_notes(shifts, changes)
    assert len(notes) == 1
    assert "Total Variation Distance 0.2000" in notes[0]


def test_derive_information_loss_notes_empty_when_under_threshold():
    shifts = {
        "val": {
            "kind": "numeric",
            "std_before": 10.0,
            "std_after": 9.8,
            "std_shift": -0.2,  # 2% shift (< 10%)
            "sample_size_before": 10,
            "sample_size_after": 10,
        },
        "cat": {
            "kind": "categorical",
            "total_variation_distance": 0.05,  # 0.05 (< 0.15)
            "sample_size_before": 10,
            "sample_size_after": 10,
        },
    }
    changes = [
        ChangeRecord("p1", "i1", 0, "val", None, 5.0, "impute_missing"),
        ChangeRecord("p2", "i2", 0, "cat", None, "A", "impute_missing"),
    ]
    notes = derive_information_loss_notes(shifts, changes)
    assert notes == []
