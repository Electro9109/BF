"""Tests for structured-data querying module."""

import pytest
import pandas as pd
from parse.structured_query import StructuredQueryEngine, query_dataset, QueryResult


class TestStructuredQueryEngine:
    """Test the structured query engine."""

    def test_get_columns_blast_furnace(self):
        """Test getting column list for BF dataset."""
        engine = StructuredQueryEngine()
        result = engine.get_columns("blast_furnace")
        assert result.success
        assert isinstance(result.data, dict)
        assert "T Fe %" in result.data
        assert "Ts" in result.data
        assert result.metadata["column_count"] > 0

    def test_get_columns_sintering(self):
        """Test getting column list for sintering dataset."""
        engine = StructuredQueryEngine()
        result = engine.get_columns("sinter_productivity")
        assert result.success
        assert isinstance(result.data, dict)
        assert "Sinter Productivity, T/m2-hr" in result.data

    def test_get_columns_energy(self):
        """Test getting column list for energy dataset."""
        engine = StructuredQueryEngine()
        result = engine.get_columns("steel_energy")
        assert result.success
        assert isinstance(result.data, dict)
        assert "Usage_kWh" in result.data

    def test_get_columns_invalid_dataset(self):
        """Test error handling for invalid dataset ID."""
        engine = StructuredQueryEngine()
        result = engine.get_columns("invalid_dataset")
        assert not result.success
        assert result.error is not None

    def test_get_statistics_numeric_column(self):
        """Test getting statistics for a numeric column."""
        engine = StructuredQueryEngine()
        result = engine.get_statistics("sinter_productivity", "Sinter Productivity, T/m2-hr")
        assert result.success
        assert "mean" in result.data
        assert "median" in result.data
        assert "min" in result.data
        assert "max" in result.data
        assert result.data["count"] > 0

    def test_get_statistics_invalid_column(self):
        """Test error handling for invalid column."""
        engine = StructuredQueryEngine()
        result = engine.get_statistics("sinter_productivity", "invalid_column")
        assert not result.success
        assert result.error is not None

    def test_filter_rows_eq(self):
        """Test filtering rows with equality operator."""
        engine = StructuredQueryEngine()
        result = engine.filter_rows("steel_energy", "Load_Type", "eq", "Light_Load")
        assert result.success
        assert isinstance(result.data, pd.DataFrame)
        assert len(result.data) > 0
        assert (result.data["Load_Type"] == "Light_Load").all()

    def test_filter_rows_gt(self):
        """Test filtering rows with greater-than operator."""
        engine = StructuredQueryEngine()
        result = engine.filter_rows("sinter_productivity", "Sinter Productivity, T/m2-hr", "gt", 1.2)
        assert result.success
        assert isinstance(result.data, pd.DataFrame)
        assert (result.data["Sinter Productivity, T/m2-hr"] > 1.2).all()

    def test_filter_rows_invalid_operator(self):
        """Test error handling for invalid operator."""
        engine = StructuredQueryEngine()
        result = engine.filter_rows("sinter_productivity", "Sinter Productivity, T/m2-hr", "invalid_op", 1.0)
        assert not result.success
        assert result.error is not None

    def test_aggregate_mean_by_group(self):
        """Test aggregating mean by group."""
        engine = StructuredQueryEngine()
        result = engine.aggregate("steel_energy", group_by="Load_Type", agg_column="Usage_kWh", agg_func="mean")
        assert result.success
        assert isinstance(result.data, dict)
        assert "Light_Load" in result.data or "Medium_Load" in result.data

    def test_aggregate_count_per_group(self):
        """Test counting per group."""
        engine = StructuredQueryEngine()
        result = engine.aggregate("steel_energy", group_by="Load_Type", agg_func="count")
        assert result.success
        assert isinstance(result.data, dict)
        assert sum(result.data.values()) > 0

    def test_aggregate_overall_mean(self):
        """Test aggregating mean over entire dataset."""
        engine = StructuredQueryEngine()
        result = engine.aggregate("sinter_productivity", agg_column="Sinter Productivity, T/m2-hr", agg_func="mean")
        assert result.success
        assert "value" in result.data
        assert isinstance(result.data["value"], float)

    def test_get_top_rows(self):
        """Test getting top N rows by column."""
        engine = StructuredQueryEngine()
        result = engine.get_top_rows("sinter_productivity", "Sinter Productivity, T/m2-hr", n=3, ascending=False)
        assert result.success
        assert isinstance(result.data, pd.DataFrame)
        assert len(result.data) <= 3
        assert result.data["Sinter Productivity, T/m2-hr"].is_monotonic_decreasing

    def test_get_top_rows_numeric_only(self):
        """Test error handling for non-numeric column."""
        engine = StructuredQueryEngine()
        result = engine.get_top_rows("steel_energy", "Load_Type", n=5)
        assert not result.success
        assert result.error is not None


class TestQueryDatasetConvenience:
    """Test the convenience function."""

    def test_convenience_get_columns(self):
        """Test convenience function for get_columns."""
        result = query_dataset("blast_furnace", "get_columns")
        assert result.success
        assert isinstance(result.data, dict)

    def test_convenience_get_statistics(self):
        """Test convenience function for get_statistics."""
        result = query_dataset("sinter_productivity", "get_statistics", column="Sinter Productivity, T/m2-hr")
        assert result.success
        assert "mean" in result.data

    def test_convenience_filter_rows(self):
        """Test convenience function for filter_rows."""
        result = query_dataset("steel_energy", "filter_rows", column="Load_Type", operator="eq", value="Light_Load")
        assert result.success
        assert len(result.data) > 0

    def test_convenience_aggregate(self):
        """Test convenience function for aggregate."""
        result = query_dataset("steel_energy", "aggregate", group_by="Load_Type", agg_column="Usage_kWh", agg_func="mean")
        assert result.success
        assert isinstance(result.data, dict)

    def test_convenience_get_top_rows(self):
        """Test convenience function for get_top_rows."""
        result = query_dataset("sinter_productivity", "get_top_rows", column="Sinter Productivity, T/m2-hr", n=5)
        assert result.success
        assert len(result.data) <= 5

    def test_convenience_invalid_operation(self):
        """Test error handling for invalid operation."""
        result = query_dataset("blast_furnace", "invalid_operation")
        assert not result.success
        assert result.error is not None

    def test_convenience_get_statistics_overall(self):
        """Test getting overall dataset statistics without specifying a column."""
        result = query_dataset("blast_furnace", "get_statistics")
        assert result.success
        assert "row_count" in result.data
        assert "column_count" in result.data
        assert result.data["row_count"] > 0
