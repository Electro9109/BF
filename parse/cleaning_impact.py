"""Downstream impact and distribution shift analysis for data cleaning.

Computes distribution shifts and potential information loss notes comparing
the original and cleaned DataFrames after transformations are applied.
All calculations are pure and self-contained (no scipy dependency).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from parse.cleaning import ChangeRecord

# Chosen rule-of-thumb thresholds for human reviewer warnings.
# These are selected policy heuristics, not mathematically derived constants.
RELATIVE_STD_SHIFT_THRESHOLD = 0.10  # 10% relative standard deviation shift threshold (chosen rule-of-thumb)
CATEGORICAL_TVD_THRESHOLD = 0.15  # 0.15 Total Variation Distance threshold (chosen rule-of-thumb)


def compute_ks_statistic(
    sample_before: np.ndarray | pd.Series | list[float],
    sample_after: np.ndarray | pd.Series | list[float],
) -> float:
    """Compute the two-sample Kolmogorov-Smirnov statistic without scipy.

    The KS statistic is defined as:
        D = max |F_before(x) - F_after(x)|
    over the sorted union of observation values from both samples.

    Args:
        sample_before: Numeric observations before transformation.
        sample_after: Numeric observations after transformation.

    Returns:
        The KS statistic in [0.0, 1.0].
    """
    arr_before = np.asarray(sample_before, dtype=float)
    arr_after = np.asarray(sample_after, dtype=float)

    arr_before = arr_before[~np.isnan(arr_before)]
    arr_after = arr_after[~np.isnan(arr_after)]

    n1 = len(arr_before)
    n2 = len(arr_after)

    if n1 == 0 or n2 == 0:
        return 0.0

    # If both samples are identical in content
    if n1 == n2 and np.array_equal(np.sort(arr_before), np.sort(arr_after)):
        return 0.0

    s1 = np.sort(arr_before)
    s2 = np.sort(arr_after)

    # Union of evaluation points
    eval_points = np.union1d(s1, s2)

    # Compute empirical CDFs at each evaluation point using right search
    # np.searchsorted(side='right') gives count of elements <= x
    cdf1 = np.searchsorted(s1, eval_points, side="right") / n1
    cdf2 = np.searchsorted(s2, eval_points, side="right") / n2

    d_stat = float(np.max(np.abs(cdf1 - cdf2)))
    return round(d_stat, 4)


def compute_distribution_shifts(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
    columns: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Compute column-by-column distribution shifts between original and cleaned frames.

    Evaluates the intersection of columns present in both frames, optionally filtered by `columns`.

    Args:
        original: DataFrame before cleaning transformations.
        cleaned: DataFrame after cleaning transformations.
        columns: Optional list of columns to evaluate (defaults to all columns in original).

    Returns:
        Dictionary mapping column name to shift metrics dict.
    """
    if columns is None:
        target_columns = [c for c in original.columns if c in cleaned.columns]
    else:
        target_columns = [c for c in columns if c in original.columns and c in cleaned.columns]

    shifts: dict[str, Any] = {}

    for col in target_columns:
        s_before = original[col]
        s_after = cleaned[col]

        col_str = str(col)

        # Check if identical (including index and null alignment)
        # Note: even if unchanged, we report full metrics with unchanged: True
        is_identical = s_before.equals(s_after)

        # Classify column dtype
        is_numeric = pd.api.types.is_numeric_dtype(s_before) and not pd.api.types.is_bool_dtype(s_before)
        is_categorical_or_bool = (
            pd.api.types.is_bool_dtype(s_before)
            or isinstance(s_before.dtype, pd.CategoricalDtype)
            or (s_before.dtype == object and s_before.dropna().nunique() <= max(20, int(len(s_before) * 0.2)))
        )

        if is_numeric:
            clean_b = pd.to_numeric(s_before.dropna(), errors="coerce").dropna()
            clean_a = pd.to_numeric(s_after.dropna(), errors="coerce").dropna()

            n_before = int(len(clean_b))
            n_after = int(len(clean_a))

            mean_b = float(clean_b.mean()) if n_before > 0 else 0.0
            mean_a = float(clean_a.mean()) if n_after > 0 else 0.0
            std_b = float(clean_b.std(ddof=1)) if n_before > 1 else 0.0
            std_a = float(clean_a.std(ddof=1)) if n_after > 1 else 0.0

            if np.isnan(std_b):
                std_b = 0.0
            if np.isnan(std_a):
                std_a = 0.0

            ks_stat = compute_ks_statistic(clean_b.to_numpy(), clean_a.to_numpy())

            shifts[col_str] = {
                "kind": "numeric",
                "unchanged": is_identical,
                "mean_before": round(mean_b, 4),
                "mean_after": round(mean_a, 4),
                "mean_shift": round(mean_a - mean_b, 4),
                "std_before": round(std_b, 4),
                "std_after": round(std_a, 4),
                "std_shift": round(std_a - std_b, 4),
                "sample_size_before": n_before,
                "sample_size_after": n_after,
                "ks_statistic": ks_stat,
                "ks_statistic_note": (
                    "Two-sample KS statistic computed without a p-value (no scipy dependency). "
                    "Use as a relative shift indicator, not a hypothesis-test result."
                ),
            }

        elif is_categorical_or_bool:
            val_b = s_before.dropna().astype(str)
            val_a = s_after.dropna().astype(str)

            n_before = int(len(val_b))
            n_after = int(len(val_a))

            probs_b = val_b.value_counts(normalize=True).to_dict() if n_before > 0 else {}
            probs_a = val_a.value_counts(normalize=True).to_dict() if n_after > 0 else {}

            all_cats = set(probs_b.keys()) | set(probs_a.keys())
            tvd = 0.5 * sum(abs(probs_a.get(c, 0.0) - probs_b.get(c, 0.0)) for c in all_cats)

            cats_added = sorted(list(set(probs_a.keys()) - set(probs_b.keys())))
            cats_removed = sorted(list(set(probs_b.keys()) - set(probs_a.keys())))

            shifts[col_str] = {
                "kind": "categorical",
                "unchanged": is_identical,
                "total_variation_distance": round(float(tvd), 4),
                "categories_added": cats_added,
                "categories_removed": cats_removed,
                "sample_size_before": n_before,
                "sample_size_after": n_after,
            }

        else:
            # High cardinality text / IDs / unsupported
            shifts[col_str] = {
                "kind": "not_computed",
                "unchanged": is_identical,
                "reason": "free-text or unsupported column type, distribution comparison not meaningful",
            }

    return shifts


def derive_information_loss_notes(
    distribution_shifts: dict[str, Any],
    changes: list[ChangeRecord],
) -> list[str]:
    """Derive human-readable potential information loss notes from computed shifts and changes.

    Evaluates columns touched by cell-level changes against fixed policy thresholds.

    Args:
        distribution_shifts: Column distribution shifts computed by `compute_distribution_shifts`.
        changes: List of ChangeRecords applied during cleaning.

    Returns:
        List of explanatory note strings.
    """
    notes: list[str] = []

    # Get set of columns directly touched by changes
    touched_columns = {c.field for c in changes if c.field is not None}

    for col in sorted(touched_columns):
        if col not in distribution_shifts:
            continue

        shift_info = distribution_shifts[col]
        kind = shift_info.get("kind")

        if kind == "numeric":
            std_before = shift_info.get("std_before", 0.0)
            std_shift = shift_info.get("std_shift", 0.0)
            n_before = shift_info.get("sample_size_before", 0)
            n_after = shift_info.get("sample_size_after", 0)

            # 1. Variance change check
            if std_before > 0:
                rel_std_change = abs(std_shift) / std_before
                if rel_std_change > RELATIVE_STD_SHIFT_THRESHOLD:
                    pct = round(rel_std_change * 100, 1)
                    notes.append(
                        f"'{col}': standard deviation changed by {pct}% after cleaning — "
                        f"check whether this reflects real signal removal (e.g. imputation reducing spread) "
                        f"rather than noise removal."
                    )

            # 2. Row removal driven shift (sample size reduced on a column that had no missing values before)
            # Note: if n_after < n_before on a column where changes were made
            # check if there were row removals
            has_row_removals = any(c.field is None for c in changes)
            if has_row_removals and n_after < n_before:
                notes.append(
                    f"'{col}': sample size reduced from {n_before} to {n_after} due to row removal — "
                    f"verify that removed observations were uninformative or redundant."
                )

        elif kind == "categorical":
            tvd = shift_info.get("total_variation_distance", 0.0)
            if tvd > CATEGORICAL_TVD_THRESHOLD:
                notes.append(
                    f"'{col}': category distribution shifted with Total Variation Distance {tvd:.4f} "
                    f"(exceeds {CATEGORICAL_TVD_THRESHOLD:.2f} threshold) — verify category balance."
                )

    return notes
