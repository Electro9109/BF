"""Unit tests for parse.dtype_utils classification primitives."""

import pandas as pd
import pytest

from parse.dtype_utils import (
    is_boolean,
    is_categorical_dtype_,
    is_datetime,
    is_empty_or_all_missing,
    is_numeric,
    is_text_like,
)


def test_text_like_classifies_both_object_and_string_dtypes():
    """Verify that is_text_like handles both legacy object and nullable string dtypes."""
    series_obj = pd.Series(["apple", "banana", "cherry"], dtype=object)
    series_str = pd.Series(["apple", "banana", "cherry"], dtype="string")

    assert is_text_like(series_obj) is True
    assert is_text_like(series_str) is True

    # And neither classifies as numeric, boolean, datetime, or explicit CategoricalDtype
    assert is_numeric(series_obj) is False
    assert is_numeric(series_str) is False
    assert is_boolean(series_obj) is False
    assert is_boolean(series_str) is False
    assert is_datetime(series_obj) is False
    assert is_datetime(series_str) is False
    assert is_categorical_dtype_(series_obj) is False
    assert is_categorical_dtype_(series_str) is False


def test_numeric_dtype_classification():
    series_int = pd.Series([1, 2, 3])
    series_float = pd.Series([1.5, 2.5, 3.5])

    assert is_numeric(series_int) is True
    assert is_numeric(series_float) is True
    assert is_text_like(series_int) is False
    assert is_text_like(series_float) is False
    assert is_boolean(series_int) is False
    assert is_datetime(series_int) is False
    assert is_categorical_dtype_(series_int) is False


def test_boolean_dtype_classification():
    series_bool = pd.Series([True, False, True])

    assert is_boolean(series_bool) is True
    assert is_numeric(series_bool) is False
    assert is_text_like(series_bool) is False
    assert is_datetime(series_bool) is False
    assert is_categorical_dtype_(series_bool) is False


def test_datetime_dtype_classification():
    series_dt = pd.to_datetime(pd.Series(["2025-01-01", "2025-01-02"]))

    assert is_datetime(series_dt) is True
    assert is_numeric(series_dt) is False
    assert is_boolean(series_dt) is False
    assert is_text_like(series_dt) is False
    assert is_categorical_dtype_(series_dt) is False


def test_categorical_dtype_classification():
    series_cat = pd.Series(pd.Categorical(["low", "medium", "high"]))

    assert is_categorical_dtype_(series_cat) is True
    assert is_text_like(series_cat) is False
    assert is_numeric(series_cat) is False
    assert is_boolean(series_cat) is False
    assert is_datetime(series_cat) is False


def test_empty_or_all_missing():
    series_empty = pd.Series([], dtype=object)
    series_all_na = pd.Series([None, float("nan"), None])
    series_with_val = pd.Series([None, "hello"])

    assert is_empty_or_all_missing(series_empty) is True
    assert is_empty_or_all_missing(series_all_na) is True
    assert is_empty_or_all_missing(series_with_val) is False
