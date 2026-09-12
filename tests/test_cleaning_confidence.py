"""Unit tests for parse.cleaning_confidence scoring functions."""

import math
import numpy as np
import pandas as pd
import pytest

from parse.cleaning_confidence import (
    DUAL_AGREEMENT_BOOST,
    MAD_OUTLIER_THRESHOLD,
    MIN_GROUP_SIZE_FOR_ASSOCIATION,
    MIN_STABLE_SAMPLE_SIZE,
    NORMAL_SCALE_MAD,
    NUMERIC_EFFECT_SIZE_CAP,
    compute_mad_outliers,
    compute_missingness_association,
    score_duplicate_removal,
    score_imputation,
    score_outlier_flag,
)


def test_score_duplicate_removal_exact_match():
    frame = pd.DataFrame({
        "a": [1, 2, 1, 3],
        "b": ["x", "y", "x", "z"],
    })
    conf, basis = score_duplicate_removal(frame)
    assert conf == 1.0
    assert basis["match_type"] == "exact_row"
    assert basis["columns_compared"] == ["a", "b"]
    assert basis["duplicate_row_count"] == 1


def test_score_duplicate_removal_no_duplicates():
    frame = pd.DataFrame({
        "a": [1, 2, 3],
        "b": ["x", "y", "z"],
    })
    conf, basis = score_duplicate_removal(frame)
    assert conf == 1.0
    assert basis["match_type"] == "exact_row"
    assert basis["duplicate_row_count"] == 0


def test_score_imputation_numeric_zero_variance():
    # 30 non-null values all identical (std == 0.0), 10 missing (missing_rate = 10/40 = 0.25)
    # s_missing_rate = 1 - 0.25 = 0.75
    # s_sample_size = min(1.0, 30 / 30) = 1.0
    # s_dispersion = 1.0 (std == 0)
    # s_placeholder_independence = 1.0
    # expected conf = 0.30*0.75 + 0.25*1.0 + 0.25*1.0 + 0.20*1.0 = 0.225 + 0.25 + 0.25 + 0.20 = 0.925
    vals = [5.0] * 30 + [np.nan] * 10
    series = pd.Series(vals)
    conf, basis = score_imputation(series)

    assert conf == pytest.approx(0.925, abs=1e-3)
    assert basis["s_missing_rate"] == 0.75
    assert basis["s_sample_size"] == 1.0
    assert basis["s_dispersion"] == 1.0
    assert basis["s_placeholder_independence"] == 1.0
    assert basis["missing_rate"] == 0.25
    assert basis["non_null_count"] == 30
    assert basis["cv"] == 0.0
    assert basis["missingness_pattern_checked"] is False
    assert "MAR/MNAR" in basis["note"]


def test_score_imputation_numeric_mean_zero():
    # values symmetric around zero, std = 2.0, mean = 0.0
    # s_dispersion = 1 / (1 + std) = 1 / 3
    vals = [-2.0, 0.0, 2.0, np.nan]
    series = pd.Series(vals)
    conf, basis = score_imputation(series)
    assert "cv" not in basis
    assert basis["s_dispersion"] == pytest.approx(1.0 / (1.0 + float(np.std([-2.0, 0.0, 2.0], ddof=1))), abs=1e-3)


def test_score_imputation_single_non_null_edge_case():
    # 1 non-null value, 99 missing (missing_rate = 0.99)
    # s_missing_rate = max(0, 1 - 0.99) = 0.01
    # s_sample_size = 1 / 30 = 0.0333
    # s_dispersion = 1.0 (only 1 value)
    series = pd.Series([42.0] + [np.nan] * 99)
    conf, basis = score_imputation(series)
    assert basis["non_null_count"] == 1
    assert basis["s_dispersion"] == 1.0
    assert conf < 0.5


def test_score_imputation_empty_series():
    series = pd.Series([], dtype=float)
    conf, basis = score_imputation(series)
    assert conf == 0.0
    assert basis["total_count"] == 0


def test_score_imputation_categorical_dominant_vs_uniform():
    # Dominant category: 30 "A", 0 "B", plus 10 NaN
    series_dominant = pd.Series(["A"] * 30 + [np.nan] * 10)
    conf_dom, basis_dom = score_imputation(series_dominant)
    assert basis_dom["normalized_entropy"] == 0.0
    assert basis_dom["s_dispersion"] == 1.0

    # Perfectly uniform categories: 15 "A", 15 "B", plus 10 NaN
    series_uniform = pd.Series(["A"] * 15 + ["B"] * 15 + [np.nan] * 10)
    conf_uni, basis_uni = score_imputation(series_uniform)
    assert basis_uni["normalized_entropy"] == pytest.approx(1.0, abs=1e-3)
    assert basis_uni["s_dispersion"] == pytest.approx(0.0, abs=1e-3)

    assert conf_dom > conf_uni


def test_score_outlier_flag_known_distances():
    # Construct a dataset with known quantiles
    # [10, 10, 10, 20, 20, 20]:
    # q25 = 10, q75 = 20, iqr = 10
    # lower_fence = 10 - 15 = -5
    # upper_fence = 20 + 15 = 35
    # Value 45 is upper outlier: distance = (45 - 35) / 10 = 1.0
    # obs_confidence = 0.5 + 0.1 * 1.0 = 0.60
    vals = [10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 45.0]
    series = pd.Series(vals)
    conf, basis = score_outlier_flag(series, q1=10.0, q3=20.0, iqr=10.0)

    assert basis["flagged_count"] == 1
    assert basis["mean_iqr_distance"] == 1.0
    assert conf == pytest.approx(0.60, abs=1e-3)
    assert basis["per_row_distances"][6] == 1.0
    assert basis["per_row_distances_capped"] is False


def test_score_outlier_flag_caps_at_095():
    # Outlier extremely far away: distance = 100
    # conf = min(0.95, 0.5 + 0.1 * 100) = 0.95
    series = pd.Series([10.0, 10.0, 20.0, 20.0, 10000.0])
    conf, basis = score_outlier_flag(series, q1=10.0, q3=20.0, iqr=10.0)
    assert conf == 0.95
    assert basis["flagged_count"] == 1


def test_score_outlier_flag_near_fence_floor():
    # Outlier just barely past fence: distance = 0.001
    # conf = 0.5 + 0.1 * 0.001 = 0.5001
    series = pd.Series([10.0, 10.0, 20.0, 20.0, 35.01])
    conf, basis = score_outlier_flag(series, q1=10.0, q3=20.0, iqr=10.0)
    assert conf >= 0.5
    assert conf < 0.51


def test_score_outlier_flag_no_outliers():
    series = pd.Series([10.0, 15.0, 20.0])
    conf, basis = score_outlier_flag(series)
    assert conf == 0.5
    assert basis["flagged_count"] == 0


def test_score_outlier_flag_zero_iqr():
    series = pd.Series([5.0, 5.0, 5.0, 5.0])
    conf, basis = score_outlier_flag(series)
    assert conf == 0.5
    assert basis["flagged_count"] == 0
    assert "IQR is zero" in basis["note"]


def test_score_outlier_flag_per_row_capping():
    # 25 outliers beyond upper fence 35
    normals = [10.0] * 10 + [20.0] * 10
    outliers = [40.0 + i for i in range(25)]
    series = pd.Series(normals + outliers)
    conf, basis = score_outlier_flag(series, q1=10.0, q3=20.0, iqr=10.0)
    assert basis["flagged_count"] == 25
    assert len(basis["per_row_distances"]) == 20
    assert basis["per_row_distances_capped"] is True
    assert len(basis["per_row_mad_distances"]) == 20
    assert len(basis["per_row_agreement"]) == 20


def test_compute_mad_outliers_known():
    # [10, 10, 10, 20, 20, 20, 100]
    # Median is 20.0
    # Absolute deviations: [10, 10, 10, 0, 0, 0, 80] -> sorted: [0, 0, 0, 10, 10, 10, 80]
    # Median deviation (raw_mad) = 10.0
    # scaled_mad = 1.4826 * 10.0 = 14.826
    # For 100: |100 - 20| / 14.826 = 80 / 14.826 = 5.3959 > 3.0 (MAD outlier!)
    series = pd.Series([10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 100.0])
    mad_dists, raw_mad, scaled_mad = compute_mad_outliers(series)

    assert raw_mad == 10.0
    assert scaled_mad == pytest.approx(14.826, abs=1e-3)
    assert 6 in mad_dists
    assert mad_dists[6] == pytest.approx(80.0 / 14.826, abs=1e-3)
    assert mad_dists[6] > MAD_OUTLIER_THRESHOLD


def test_compute_mad_outliers_zero_mad():
    # More than 50% values identical -> deviations median is 0
    series = pd.Series([5.0, 5.0, 5.0, 5.0, 10.0])
    mad_dists, raw_mad, scaled_mad = compute_mad_outliers(series)

    assert raw_mad == 0.0
    assert scaled_mad == 0.0
    assert mad_dists == {}


def test_compute_mad_outliers_too_few_observations():
    series = pd.Series([1.0, 2.0])
    mad_dists, raw_mad, scaled_mad = compute_mad_outliers(series)
    assert mad_dists == {}
    assert raw_mad == 0.0


def test_score_outlier_flag_dual_method_agreement_boost():
    # Construct a dataset where an observation is flagged by BOTH IQR and MAD.
    # [10, 10, 10, 20, 20, 20, 100]:
    # q25 = 10, q75 = 20, iqr = 10 -> upper_fence = 35.
    # 100 exceeds upper_fence: dist = (100 - 35) / 10 = 6.5.
    # Base confidence: min(0.95, 0.5 + 0.1 * 6.5) = min(0.95, 1.15) = 0.95 (capped).
    # To test the boost before capping, let's pick a distance where base_conf < 0.95:
    # E.g. target base_conf = 0.65 -> dist = 1.5 -> value = 35 + 15 = 50.
    # For value 50:
    # iqr_distance = (50 - 35) / 10 = 1.5 -> base_conf = 0.5 + 0.1 * 1.5 = 0.65.
    # Let's check MAD for 50 with [10, 10, 10, 20, 20, 20, 50]:
    # Med = 20, raw_mad = 10, scaled_mad = 14.826.
    # For 50: |50 - 20| / 14.826 = 30 / 14.826 = 2.023 < 3.0 (MAD doesn't flag).
    # What if scaled_mad was smaller?
    # Say [19, 20, 20, 20, 21, 21, 50]:
    # Med = 20. Deviations: [1, 0, 0, 0, 1, 1, 30] -> sorted: [0, 0, 0, 1, 1, 1, 30] -> raw_mad = 1.0.
    # scaled_mad = 1.4826.
    # q25 = 20.0, q75 = 21.0, iqr = 1.0.
    # upper_fence = 21 + 1.5 * 1.0 = 22.5.
    # For value 24.0:
    # iqr_distance = (24.0 - 22.5) / 1.0 = 1.5.
    # base_conf = 0.5 + 0.1 * 1.5 = 0.65.
    # MAD distance for 24.0: |24 - 20| / 1.4826 = 4 / 1.4826 = 2.698 < 3.0.
    # For value 25.0:
    # iqr_distance = (25.0 - 22.5) / 1.0 = 2.5 -> base_conf = 0.5 + 0.1 * 2.5 = 0.75.
    # MAD distance for 25.0: |25 - 20| / 1.4826 = 5 / 1.4826 = 3.3725 > 3.0 (MAD flags!).
    # Dual agreement holds!
    # Expected confidence = base_conf + 0.05 = 0.75 + 0.05 = 0.80!
    series = pd.Series([19.0, 20.0, 20.0, 20.0, 21.0, 21.0, 25.0])
    conf, basis = score_outlier_flag(series, q1=20.0, q3=21.0, iqr=1.0)

    assert basis["flagged_count"] == 1
    assert basis["agreement_count"] == 1
    assert basis["agreement_rate"] == 1.0
    assert basis["per_row_agreement"][6] is True
    assert conf == pytest.approx(0.80, abs=1e-3)
    assert basis["dual_method"] == "dual_IQR_MAD"


def test_score_outlier_flag_single_method_no_boost():
    # Value 24.0 in [19, 20, 20, 20, 21, 21, 24]:
    # iqr_distance = (24.0 - 22.5) / 1.0 = 1.5 -> base_conf = 0.65.
    # MAD distance = 4 / 1.4826 = 2.698 < 3.0 (MAD does NOT flag).
    # Agreement count = 0, no boost -> conf == 0.65 exactly!
    series = pd.Series([19.0, 20.0, 20.0, 20.0, 21.0, 21.0, 24.0])
    conf, basis = score_outlier_flag(series, q1=20.0, q3=21.0, iqr=1.0)

    assert basis["flagged_count"] == 1
    assert basis["agreement_count"] == 0
    assert basis["agreement_rate"] == 0.0
    assert basis["per_row_agreement"][6] is False
    assert conf == pytest.approx(0.65, abs=1e-3)


# --- Task 5: Missingness Association and Imputation Confidence Tests ---


def test_compute_missingness_association_numeric_known_effect():
    # Construct a dataset where missingness in 'target' strongly associates with 'num_col'.
    # Group missing (is_missing=True): 5 observations of 10.0 (mean=10.0, var=0.0)
    # Group non-missing (is_missing=False): 5 observations of 4.0 (mean=4.0, var=0.0)
    # df = 5 + 5 - 2 = 8, pooled_var = 0.0 -> pooled_std = 0.0
    # Guard with non-zero variance:
    # missing: [8.0, 10.0, 10.0, 10.0, 12.0] -> mean = 10.0, s^2 = 2.0 (ddof=1: (-2)^2 + 0 + 0 + 0 + 2^2 = 8 / 4 = 2.0)
    # nonmissing: [2.0, 4.0, 4.0, 4.0, 6.0] -> mean = 4.0, s^2 = 2.0 (ddof=1: (-2)^2 + 0 + 0 + 0 + 2^2 = 8 / 4 = 2.0)
    # pooled_var = (4*2.0 + 4*2.0) / 8 = 2.0 -> pooled_std = sqrt(2.0)
    # effect_size = |10.0 - 4.0| / sqrt(2.0) = 6.0 / 1.41421356 = 4.24264
    # NUMERIC_EFFECT_SIZE_CAP = 3.0 -> min(1.0, 4.24264 / 3.0) = 1.0.
    # Now let's craft an exact effect size < 3.0:
    # mean diff = 1.5, pooled_std = 1.0 -> effect_size = 1.5 -> assoc = 1.5 / 3.0 = 0.50.
    # missing (n=5): [4.0, 5.0, 5.5, 6.0, 7.0] -> sum=27.5, mean=5.5.
    # Let's use simple constant variance by calculation:
    # Group 1 (missing): [9, 10, 10, 10, 11] -> mean = 10, s1^2 = (1 + 0 + 0 + 0 + 1)/4 = 0.5
    # Group 2 (non-missing): [7.5, 8.5, 8.5, 8.5, 9.5] -> mean = 8.5, s2^2 = (1 + 0 + 0 + 0 + 1)/4 = 0.5
    # pooled_std = sqrt((4*0.5 + 4*0.5)/8) = sqrt(0.5) = 0.70710678
    # mean diff = 1.5
    # effect = 1.5 / sqrt(0.5) = 1.5 * sqrt(2) = 2.12132034
    # expected assoc = 2.12132034 / 3.0 = 0.70710678 = sqrt(0.5)
    df = pd.DataFrame({
        "target": [np.nan] * 5 + [1.0] * 5,
        "num_col": [9.0, 10.0, 10.0, 10.0, 11.0, 7.5, 8.5, 8.5, 8.5, 9.5],
    })

    assoc, basis = compute_missingness_association(df, "target")
    expected_assoc = (1.5 / math.sqrt(0.5)) / 3.0
    assert assoc == pytest.approx(expected_assoc, abs=1e-3)
    assert basis["max_association"] == pytest.approx(expected_assoc, abs=1e-3)
    assert basis["associated_column"] == "num_col"
    assert basis["association_method"] == "effect_size"
    assert basis["columns_compared"] == 1
    assert basis["columns_skipped_insufficient_data"] == 0


def test_compute_missingness_association_categorical_known_cramers_v():
    # Hand-built 2x2 contingency table:
    #           Cat_A   Cat_B   Total
    # Missing:    5       0       5
    # Non-miss:   0       5       5
    # Total:      5       5      10
    #
    # Expected:
    # E(Missing, A) = 5*5/10 = 2.5, E(Missing, B) = 2.5
    # E(Non-miss, A) = 2.5, E(Non-miss, B) = 2.5
    # Chi2 = 4 * ((5 - 2.5)^2 / 2.5) = 4 * (6.25 / 2.5) = 4 * 2.5 = 10.0.
    # Cramér's V = sqrt(Chi2 / grand_total) = sqrt(10.0 / 10.0) = 1.0.
    df = pd.DataFrame({
        "target": [np.nan] * 5 + [100.0] * 5,
        "cat_col": ["A"] * 5 + ["B"] * 5,
    })

    assoc, basis = compute_missingness_association(df, "target")
    assert assoc == pytest.approx(1.0, abs=1e-3)
    assert basis["associated_column"] == "cat_col"
    assert basis["association_method"] == "cramers_v"
    assert basis["columns_compared"] == 1


def test_compute_missingness_association_null_uncorrelated():
    # Completely balanced / independent:
    # Both missing and non-missing have identical distributions in num_col and cat_col.
    # num_col: identical values for missing and non-missing -> mean diff = 0 -> effect = 0.
    # cat_col: perfectly balanced A and B in both groups -> chi2 = 0 -> V = 0.
    df = pd.DataFrame({
        "target": [np.nan] * 6 + [1.0] * 6,
        "num_col": [10.0, 20.0, 30.0, 10.0, 20.0, 30.0, 10.0, 20.0, 30.0, 10.0, 20.0, 30.0],
        "cat_col": ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y"],
    })

    assoc, basis = compute_missingness_association(df, "target")
    assert assoc == pytest.approx(0.0, abs=1e-3)
    assert basis["max_association"] == 0.0
    assert basis["columns_compared"] == 2
    assert basis["columns_skipped_insufficient_data"] == 0


def test_compute_missingness_association_insufficient_data_skipped():
    # Target has 4 missing and 20 non-missing.
    # Since n_missing = 4 < MIN_GROUP_SIZE_FOR_ASSOCIATION (5), other column should be skipped!
    df = pd.DataFrame({
        "target": [np.nan] * 4 + [1.0] * 20,
        "other": list(range(24)),
    })

    assoc, basis = compute_missingness_association(df, "target")
    assert assoc == 0.0
    assert basis["columns_compared"] == 0
    assert basis["columns_skipped_insufficient_data"] == 1
    assert basis["associated_column"] is None


def test_compute_missingness_association_guards():
    # Missing column not in frame
    df = pd.DataFrame({"a": [1, 2, 3]})
    assoc, basis = compute_missingness_association(df, "nonexistent")
    assert assoc == 0.0
    assert "not found" in basis["note"]

    # Target has no missing values
    df_no_missing = pd.DataFrame({"target": [1, 2, 3, 4], "other": [5, 6, 7, 8]})
    assoc, basis = compute_missingness_association(df_no_missing, "target")
    assert assoc == 0.0
    assert "no missing values" in basis["note"]

    # Target is fully missing
    df_all_missing = pd.DataFrame({"target": [np.nan] * 5, "other": [1, 2, 3, 4, 5]})
    assoc, basis = compute_missingness_association(df_all_missing, "target")
    assert assoc == 0.0
    assert "no non-missing values" in basis["note"]


def test_score_imputation_with_frame_penalizes_correlated_missingness():
    # Construct a dataset where missingness in 'target' is strongly correlated with 'cat'.
    # When frame is passed, s_placeholder_independence drops, lowering confidence.
    df = pd.DataFrame({
        "target": [np.nan] * 10 + [5.0] * 30,
        "cat": ["A"] * 10 + ["B"] * 30,
    })

    # Call with frame and column_name
    conf_with_frame, basis_with_frame = score_imputation(df["target"], frame=df, column_name="target")
    assert basis_with_frame["missingness_pattern_checked"] is True
    assert basis_with_frame["max_association"] == pytest.approx(1.0, abs=1e-3)
    assert basis_with_frame["s_placeholder_independence"] == pytest.approx(0.0, abs=1e-3)

    # Call without frame (backward compatibility fallback)
    conf_without_frame, basis_without_frame = score_imputation(df["target"])
    assert basis_without_frame["missingness_pattern_checked"] is False
    assert basis_without_frame["s_placeholder_independence"] == 1.0

    # The confidence with detected cross-column association must be strictly lower
    # Difference should be 0.20 * (1.0 - 0.0) = 0.20
    assert conf_without_frame - conf_with_frame == pytest.approx(0.20, abs=1e-3)

