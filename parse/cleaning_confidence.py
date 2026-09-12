"""Confidence scoring for data cleaning transformation proposals.

Per the PARSE philosophy, automation scales with evidence and reversibility.
Human reviewers need a calibrated, explainable signal to prioritize review
across proposals. Every scorer returns a confidence float in [0.0, 1.0] and
an inspectable basis dictionary detailing the sub-scores and raw inputs.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

from parse.dtype_utils import is_text_like

# The 30-observation floor is a conventional rule-of-thumb for a stable
# median/mode estimate, not a mathematically proven threshold.
MIN_STABLE_SAMPLE_SIZE = 30  # rule-of-thumb, revisit

# Constants for MAD-based outlier detection and dual-method agreement
NORMAL_SCALE_MAD = 1.4826  # Normal-consistent scale factor (1 / Phi^-1(0.75))
MAD_OUTLIER_THRESHOLD = 3.0  # Chosen threshold for modified Z-score (|x - med| / (1.4826 * MAD))
DUAL_AGREEMENT_BOOST = 0.05  # Confidence boost when both IQR and MAD flag an observation


def score_duplicate_removal(
    frame: pd.DataFrame,
    duplicate_indices: Iterable[Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Score confidence for exact-row duplicate removal.

    Exact full-row duplicates are a deterministic, byte-identical match across all
    columns. There is no statistical estimation or uncertainty in exact duplicates,
    so confidence is 1.0.

    Args:
        frame: The DataFrame being analyzed.
        duplicate_indices: Precomputed duplicate row indices. If None, computes
            frame.index[frame.duplicated(keep="first")].

    Returns:
        (confidence, basis) where confidence is 1.0 and basis describes the match.
    """
    if duplicate_indices is None:
        dupe_indices = tuple(frame.index[frame.duplicated(keep="first")])
    else:
        dupe_indices = tuple(duplicate_indices)

    confidence = 1.0
    basis: dict[str, Any] = {
        "match_type": "exact_row",
        "columns_compared": [str(c) for c in frame.columns],
        "duplicate_row_count": len(dupe_indices),
    }
    return confidence, basis


# Constants for missingness association check (Task 5)
# Rule-of-thumb floor for group size in missing vs non-missing comparison
MIN_GROUP_SIZE_FOR_ASSOCIATION = 5  # rule-of-thumb, revisit
# Chosen upper cap for numeric effect size (Cohen's d): 3 pooled SDs corresponds to max association 1.0
NUMERIC_EFFECT_SIZE_CAP = 3.0  # chosen threshold, not derived


def compute_missingness_association(
    frame: pd.DataFrame,
    target_column: str,
    min_group_size: int = MIN_GROUP_SIZE_FOR_ASSOCIATION,
    effect_size_cap: float = NUMERIC_EFFECT_SIZE_CAP,
) -> tuple[float, dict[str, Any]]:
    """Compute association between target column missingness and all other columns.

    Checks whether missingness in `target_column` correlates with values in any other
    column. If missingness is correlated with other observed variables (MAR/MNAR),
    unconditional single-value imputation (median or mode) is less representative,
    so confidence in the imputation should decrease.

    Args:
        frame: The full DataFrame containing target_column and other columns.
        target_column: The name of the column whose missingness is being analyzed.
        min_group_size: Minimum non-null count required in BOTH missing and
            non-missing groups to compute an association (default: 5).
        effect_size_cap: Maximum Cohen's d divisor representing maximal separation (default: 3.0).

    Returns:
        (max_association, basis_dict) where max_association is in [0.0, 1.0].
    """
    if target_column not in frame.columns:
        return 0.0, {
            "max_association": 0.0,
            "associated_column": None,
            "association_method": None,
            "columns_compared": 0,
            "columns_skipped_insufficient_data": 0,
            "note": f"Column '{target_column}' not found in frame.",
        }

    is_missing = frame[target_column].isna()
    # Guard: if target column is completely non-missing or completely missing across frame
    if is_missing.nunique() < 2:
        return 0.0, {
            "max_association": 0.0,
            "associated_column": None,
            "association_method": None,
            "columns_compared": 0,
            "columns_skipped_insufficient_data": 0,
            "note": "Target column has no missing values or no non-missing values to compare against.",
        }

    columns_compared = 0
    columns_skipped_insufficient_data = 0
    max_association = 0.0
    associated_column: str | None = None
    association_method: str | None = None

    for col in frame.columns:
        if col == target_column:
            continue

        other_series = frame[col]

        # Partition other_series by target column missingness, dropping NaN in other_series
        group_missing = other_series[is_missing].dropna()
        group_nonmissing = other_series[~is_missing].dropna()

        n_miss = len(group_missing)
        n_nonmiss = len(group_nonmissing)

        if n_miss < min_group_size or n_nonmiss < min_group_size:
            columns_skipped_insufficient_data += 1
            continue

        is_numeric = pd.api.types.is_numeric_dtype(other_series) and not pd.api.types.is_bool_dtype(other_series)
        is_categorical = (
            isinstance(other_series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(other_series)
            or is_text_like(other_series)
        )

        if is_numeric:
            s_miss_vals = pd.to_numeric(group_missing, errors="coerce").dropna()
            s_nonmiss_vals = pd.to_numeric(group_nonmissing, errors="coerce").dropna()
            n1 = len(s_miss_vals)
            n2 = len(s_nonmiss_vals)
            if n1 < min_group_size or n2 < min_group_size:
                columns_skipped_insufficient_data += 1
                continue

            mean1 = float(s_miss_vals.mean())
            mean2 = float(s_nonmiss_vals.mean())
            var1 = float(s_miss_vals.var(ddof=1)) if n1 > 1 else 0.0
            var2 = float(s_nonmiss_vals.var(ddof=1)) if n2 > 1 else 0.0

            # Pooled standard deviation: sqrt(((n1 - 1)*s1^2 + (n2 - 1)*s2^2) / (n1 + n2 - 2))
            df = n1 + n2 - 2
            if df > 0:
                pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / df
                pooled_std = math.sqrt(max(0.0, pooled_var))
            else:
                pooled_std = 0.0

            if pooled_std == 0.0:
                if mean1 == mean2:
                    assoc = 0.0
                else:
                    assoc = 1.0
            else:
                effect_size = abs(mean1 - mean2) / pooled_std
                assoc = min(1.0, effect_size / float(effect_size_cap))

            columns_compared += 1
            if assoc > max_association:
                max_association = assoc
                associated_column = str(col)
                association_method = "effect_size"

        elif is_categorical:
            # Contingency table between is_missing (2 rows) and categories of other_series (k cols)
            # Combine is_missing and other_series dropna
            valid_mask = other_series.notna()
            miss_sub = is_missing[valid_mask]
            cat_sub = other_series[valid_mask]

            contingency = pd.crosstab(miss_sub, cat_sub)
            rows, cols = contingency.shape
            if rows < 2 or cols < 2:
                # No variation in categories across the combined subsets
                columns_skipped_insufficient_data += 1
                continue

            grand_total = contingency.values.sum()
            if grand_total == 0:
                columns_skipped_insufficient_data += 1
                continue

            row_sums = contingency.sum(axis=1).values
            col_sums = contingency.sum(axis=0).values

            # Compute chi-square: sum((O - E)^2 / E)
            chi2 = 0.0
            for i in range(rows):
                for j in range(cols):
                    expected = (row_sums[i] * col_sums[j]) / grand_total
                    if expected > 0:
                        observed = contingency.iloc[i, j]
                        chi2 += ((observed - expected) ** 2) / expected

            # Cramér's V: sqrt(chi2 / (n * min(rows - 1, cols - 1)))
            # Here rows = 2, so min(rows - 1, cols - 1) = min(1, cols - 1) = 1 (since cols >= 2).
            # Thus V = sqrt(chi2 / grand_total).
            v = math.sqrt(max(0.0, chi2 / grand_total))
            assoc = max(0.0, min(1.0, v))

            columns_compared += 1
            if assoc > max_association:
                max_association = assoc
                associated_column = str(col)
                association_method = "cramers_v"
        else:
            # Explicitly skip non-numeric and non-categorical/boolean types (e.g. datetime)
            # per non-negotiables, not counting as compared or zero association.
            continue

    # Note on absence of evidence vs evidence of absence:
    # If no column qualified or no association was detected, max_association is 0.0,
    # meaning s_placeholder_independence = 1.0. Absence of detectable association
    # does not prove independence, but defaults to not penalizing confidence.
    basis: dict[str, Any] = {
        "max_association": round(max_association, 4),
        "associated_column": associated_column,
        "association_method": association_method,
        "columns_compared": columns_compared,
        "columns_skipped_insufficient_data": columns_skipped_insufficient_data,
    }
    return max_association, basis


def score_imputation(
    series: pd.Series,
    frame: pd.DataFrame | None = None,
    column_name: str | None = None,
) -> tuple[float, dict[str, Any]]:
    """Score confidence for missing value imputation (median or mode).

    Confidence reflects how representative the proposed fill value is likely to be,
    given what is observable in the non-null data. It is a composite weighted average
    of four factors:
        confidence = 0.30 * s_missing_rate
                   + 0.25 * s_sample_size
                   + 0.25 * s_dispersion
                   + 0.20 * s_placeholder_independence

    Sub-scores:
    - s_missing_rate: 1 - missing_rate, clipped to [0.0, 0.95] (never claim full certainty
      since missingness is present).
    - s_sample_size: min(1.0, non_null_count / MIN_STABLE_SAMPLE_SIZE), where 30 observations
      is a conventional rule-of-thumb for sample stability.
    - s_dispersion:
        * numeric: 1 / (1 + CV) where CV = std / abs(mean). If std == 0, s_dispersion = 1.0.
          If mean == 0, falls back to 1 / (1 + std).
        * categorical/boolean: 1 - normalized_entropy of value frequencies (Shannon entropy /
          log(k)). 1.0 if single dominant value or uniform-free, 0.0 if perfectly uniform across
          multiple classes.
    - s_placeholder_independence: 1.0 - max_association if frame and column_name are provided,
      derived from cross-column missingness association check. Falls back to 1.0 with a disclaimer
      if frame or column_name is omitted for backward compatibility.

    Returns:
        (confidence, basis) with sub-scores and raw inputs.
    """
    total_count = len(series)
    if total_count == 0:
        return 0.0, {
            "s_missing_rate": 0.0,
            "s_sample_size": 0.0,
            "s_dispersion": 0.0,
            "s_placeholder_independence": 1.0,
            "missing_rate": 1.0,
            "non_null_count": 0,
            "total_count": 0,
            "missingness_pattern_checked": False,
            "note": "Empty series.",
        }

    non_null = series.dropna()
    non_null_count = len(non_null)
    missing_count = total_count - non_null_count
    missing_rate = missing_count / total_count

    # 1. Missing rate factor: clipped to [0, 0.95]
    s_missing_rate = max(0.0, min(0.95, 1.0 - missing_rate))

    # 2. Sample size factor
    s_sample_size = min(1.0, non_null_count / float(MIN_STABLE_SAMPLE_SIZE))

    # 3. Dispersion factor
    is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
    cv: float | None = None
    normalized_entropy: float | None = None

    if non_null_count == 0:
        s_dispersion = 0.0
    elif is_numeric:
        # Convert non_null to float numeric
        numeric_vals = pd.to_numeric(non_null, errors="coerce").dropna()
        if len(numeric_vals) <= 1:
            # Single value or all coerced
            std = 0.0
            mean = float(numeric_vals.iloc[0]) if len(numeric_vals) == 1 else 0.0
        else:
            std = float(numeric_vals.std(ddof=1))
            mean = float(numeric_vals.mean())

        if std == 0.0 or math.isnan(std):
            s_dispersion = 1.0
            cv = 0.0
        elif mean == 0.0:
            cv = None
            s_dispersion = 1.0 / (1.0 + std)
        else:
            cv = std / abs(mean)
            s_dispersion = 1.0 / (1.0 + cv)
    else:
        # Categorical or boolean: normalized entropy
        val_counts = non_null.value_counts()
        k = len(val_counts)
        if k <= 1:
            normalized_entropy = 0.0
            s_dispersion = 1.0
        else:
            probs = (val_counts / non_null_count).to_numpy()
            entropy = float(-np.sum(probs * np.log(probs)))
            max_entropy = math.log(k)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            # Clamp normalized_entropy to [0.0, 1.0]
            normalized_entropy = max(0.0, min(1.0, normalized_entropy))
            s_dispersion = 1.0 - normalized_entropy

    # 4. Missingness independence check (Task 5)
    if frame is not None and column_name is not None and column_name in frame.columns:
        max_assoc, assoc_basis = compute_missingness_association(frame, column_name)
        s_placeholder_independence = max(0.0, min(1.0, 1.0 - max_assoc))
        pattern_checked = True
    else:
        # Fallback when frame or column_name is omitted (backward compatibility)
        s_placeholder_independence = 1.0
        pattern_checked = False
        assoc_basis = None

    confidence = (
        0.30 * s_missing_rate
        + 0.25 * s_sample_size
        + 0.25 * s_dispersion
        + 0.20 * s_placeholder_independence
    )
    # Ensure clamped to [0.0, 1.0]
    confidence = max(0.0, min(1.0, float(confidence)))

    basis: dict[str, Any] = {
        "s_missing_rate": round(s_missing_rate, 4),
        "s_sample_size": round(s_sample_size, 4),
        "s_dispersion": round(s_dispersion, 4),
        "s_placeholder_independence": round(s_placeholder_independence, 4),
        "missing_rate": round(missing_rate, 4),
        "non_null_count": non_null_count,
        "total_count": total_count,
        "missingness_pattern_checked": pattern_checked,
    }

    if pattern_checked and assoc_basis is not None:
        basis.update(assoc_basis)
    else:
        basis["note"] = "Frame or column_name omitted; MAR/MNAR cross-column check skipped."

    if cv is not None:
        basis["cv"] = round(cv, 4)
    if normalized_entropy is not None:
        basis["normalized_entropy"] = round(normalized_entropy, 4)

    return confidence, basis


def compute_mad_outliers(
    series: pd.Series,
    threshold: float = MAD_OUTLIER_THRESHOLD,
    normal_scale: float = NORMAL_SCALE_MAD,
) -> tuple[dict[Any, float], float, float]:
    """Compute scaled MAD and modified Z-scores for numeric observations.

    Args:
        series: Numeric series.
        threshold: Modified Z-score cutoff (default: 3.0).
        normal_scale: Asymptotic normal consistency multiplier (default: 1.4826).

    Returns:
        (per_row_mad_distances, raw_mad, scaled_mad)
        where per_row_mad_distances maps row index to modified Z-score (|x - median| / scaled_mad).
    """
    clean_series = pd.to_numeric(series.dropna(), errors="coerce").dropna()
    if len(clean_series) < 3:
        return {}, 0.0, 0.0

    med = float(clean_series.median())
    deviations = (clean_series - med).abs()
    raw_mad = float(deviations.median())
    scaled_mad = raw_mad * normal_scale

    if scaled_mad <= 0.0 or math.isnan(scaled_mad):
        return {}, raw_mad, 0.0

    z_scores = deviations / scaled_mad
    return {idx: float(z_scores.loc[idx]) for idx in clean_series.index}, raw_mad, scaled_mad


def score_outlier_flag(
    series: pd.Series,
    q1: float | None = None,
    q3: float | None = None,
    iqr: float | None = None,
) -> tuple[float, dict[str, Any]]:
    """Score confidence that flagged values are genuinely statistically unusual.

    This proposal does not alter data (action is flag_for_review). Confidence reflects
    how far beyond the IQR fences the flagged observations sit, strengthened by dual-method
    agreement when observations are also flagged under the MAD (Median Absolute Deviation) rule.

    For each flagged observation:
        distance = (fence_violated - fence) / iqr
        base_confidence = min(0.95, max(0.5, 0.5 + 0.1 * distance))
        if MAD also flags observation (modified Z > 3.0):
            observation_confidence = min(0.95, base_confidence + 0.05)
        else:
            observation_confidence = base_confidence

    The proposal-level confidence is the mean across all flagged observations in the column.

    Args:
        series: The numeric series being evaluated.
        q1: 25th percentile (optional, computed if None).
        q3: 75th percentile (optional, computed if None).
        iqr: Interquartile range (optional, computed if None).

    Returns:
        (confidence, basis) with dual-method statistics and per-row details.
    """
    clean_series = pd.to_numeric(series.dropna(), errors="coerce").dropna()
    if len(clean_series) == 0:
        return 0.5, {
            "method": "IQR_1.5",
            "dual_method": "dual_IQR_MAD",
            "flagged_count": 0,
            "mean_iqr_distance": 0.0,
            "per_row_distances": {},
            "per_row_distances_capped": False,
            "note": "No valid numeric observations to evaluate.",
        }

    mad_distances, raw_mad, scaled_mad = compute_mad_outliers(clean_series)

    if q1 is None or q3 is None or iqr is None:
        q_25 = float(clean_series.quantile(0.25))
        q_75 = float(clean_series.quantile(0.75))
        calc_iqr = q_75 - q_25
    else:
        q_25, q_75, calc_iqr = float(q1), float(q3), float(iqr)

    if calc_iqr <= 0:
        # IQR is zero (e.g. constant or low-variance discrete distribution)
        return 0.5, {
            "method": "IQR_1.5",
            "dual_method": "dual_IQR_MAD",
            "flagged_count": 0,
            "mean_iqr_distance": 0.0,
            "per_row_distances": {},
            "per_row_distances_capped": False,
            "raw_mad": round(raw_mad, 4),
            "scaled_mad": round(scaled_mad, 4),
            "note": "IQR is zero; no fence can be constructed.",
        }

    lower_fence = q_25 - 1.5 * calc_iqr
    upper_fence = q_75 + 1.5 * calc_iqr

    # Identify outliers and compute distances
    row_distances: dict[Any, float] = {}
    row_confidences: list[float] = []

    for idx, val in clean_series.items():
        val_float = float(val)
        if val_float < lower_fence:
            dist = (lower_fence - val_float) / calc_iqr
        elif val_float > upper_fence:
            dist = (val_float - upper_fence) / calc_iqr
        else:
            continue

        # Base confidence from IQR distance
        base_obs_conf = min(0.95, max(0.5, 0.5 + 0.1 * dist))
        # Dual-method agreement check: does MAD also flag this observation?
        z_mad = mad_distances.get(idx, 0.0)
        mad_agrees = z_mad > MAD_OUTLIER_THRESHOLD
        if mad_agrees:
            obs_conf = min(0.95, base_obs_conf + DUAL_AGREEMENT_BOOST)
        else:
            obs_conf = base_obs_conf

        row_distances[idx] = round(dist, 4)
        row_confidences.append(obs_conf)

    flagged_count = len(row_confidences)
    mad_flagged_count = sum(1 for z in mad_distances.values() if z > MAD_OUTLIER_THRESHOLD)
    agreement_count = sum(1 for idx in row_distances if mad_distances.get(idx, 0.0) > MAD_OUTLIER_THRESHOLD)
    agreement_rate = (agreement_count / flagged_count) if flagged_count > 0 else 0.0

    if flagged_count == 0:
        # No observations fell outside the fences
        return 0.5, {
            "method": "IQR_1.5",
            "dual_method": "dual_IQR_MAD",
            "flagged_count": 0,
            "mean_iqr_distance": 0.0,
            "per_row_distances": {},
            "per_row_distances_capped": False,
            "raw_mad": round(raw_mad, 4),
            "scaled_mad": round(scaled_mad, 4),
            "mad_flagged_count": mad_flagged_count,
            "agreement_count": 0,
            "agreement_rate": 0.0,
            "note": "No values exceeded IQR fences.",
        }

    mean_confidence = float(np.mean(row_confidences))
    mean_distance = float(np.mean(list(row_distances.values())))

    # Cap per_row_distances and per_row_mad_distances to first 20 entries
    capped = len(row_distances) > 20
    capped_distances = {k: v for i, (k, v) in enumerate(row_distances.items()) if i < 20}
    capped_mad_distances = {k: round(mad_distances.get(k, 0.0), 4) for i, k in enumerate(row_distances.keys()) if i < 20}
    capped_agreements = {k: bool(mad_distances.get(k, 0.0) > MAD_OUTLIER_THRESHOLD) for i, k in enumerate(row_distances.keys()) if i < 20}

    basis: dict[str, Any] = {
        "method": "IQR_1.5",
        "dual_method": "dual_IQR_MAD",
        "flagged_count": flagged_count,
        "mean_iqr_distance": round(mean_distance, 4),
        "per_row_distances": capped_distances,
        "per_row_distances_capped": capped,
        "q1": round(q_25, 4),
        "q3": round(q_75, 4),
        "iqr": round(calc_iqr, 4),
        "lower_fence": round(lower_fence, 4),
        "upper_fence": round(upper_fence, 4),
        "raw_mad": round(raw_mad, 4),
        "scaled_mad": round(scaled_mad, 4),
        "median": round(float(clean_series.median()), 4) if len(clean_series) else 0.0,
        "mad_flagged_count": mad_flagged_count,
        "agreement_count": agreement_count,
        "agreement_rate": round(agreement_rate, 4),
        "per_row_mad_distances": capped_mad_distances,
        "per_row_agreement": capped_agreements,
    }

    return round(mean_confidence, 4), basis
