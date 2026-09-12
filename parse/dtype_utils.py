"""Pandas version-robust dtype classification utilities.

Pandas 3.0 changed default string column dtypes from `object` to dedicated `str`
(or `StringDtype`). Fragile checks such as `series.dtype == object` or
`pd.api.types.is_object_dtype` evaluate to False for string columns under pandas 3.x.

This module provides exclusion-based classification primitives that do not rely on
fragile dtype name enumeration or pandas-version-dependent string heuristics.
"""

from __future__ import annotations

import pandas as pd


def is_numeric(series: pd.Series) -> bool:
    """True if series contains numeric values (excluding booleans)."""
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def is_boolean(series: pd.Series) -> bool:
    """True if series contains boolean values."""
    return pd.api.types.is_bool_dtype(series)


def is_datetime(series: pd.Series) -> bool:
    """True if series contains datetime values."""
    return pd.api.types.is_datetime64_any_dtype(series)


def is_categorical_dtype_(series: pd.Series) -> bool:
    """True if series explicitly uses pandas CategoricalDtype."""
    return isinstance(series.dtype, pd.CategoricalDtype)


def is_text_like(series: pd.Series) -> bool:
    """True for any series that is not numeric, boolean, datetime, or explicit CategoricalDtype.

    Exclusion-based: covers legacy `object` dtype, pandas' nullable StringDtype, and
    pandas 3.x's native `str` dtype uniformly without relying on specific dtype names.
    """
    return not (
        is_numeric(series)
        or is_boolean(series)
        or is_datetime(series)
        or is_categorical_dtype_(series)
    )


def is_empty_or_all_missing(series: pd.Series) -> bool:
    """True if series has 0 non-null observations."""
    return series.dropna().empty
