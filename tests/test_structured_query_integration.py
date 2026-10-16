"""Integration tests for structured-query application workflow."""

import pytest
import pandas as pd
from parse.structured_query import StructuredQueryEngine, query_dataset


class TestStructuredQueryApplicationIntegration:
    """Integration tests for the structured-query application workflow."""

    def test_app_level_get_columns_all_datasets(self):
        """Test that all registered datasets can be queried for columns."""
        engine = StructuredQueryEngine()
        
        for dataset_id in ["blast_furnace", "sinter_productivity", "steel_energy"]:
            result = engine.get_columns(dataset_id)
            assert result.success, f"get_columns failed for {dataset_id}: {result.error}"
            assert isinstance(result.data, dict)
            assert len(result.data) > 0
            assert result.metadata["column_count"] > 0

    def test_app_level_statistics_sintering(self):
        """Test getting statistics for sintering productivity column."""
        result = query_dataset("sinter_productivity", "get_statistics", column="Sinter Productivity, T/m2-hr")
        assert result.success
        assert "mean" in result.data
        assert "median" in result.data
        assert result.data["count"] > 0

    def test_app_level_filter_energy_light_load(self):
        """Test filtering energy dataset by load type."""
        result = query_dataset("steel_energy", "filter_rows", column="Load_Type", operator="eq", value="Light_Load")
        assert result.success
        assert result.row_count > 0
        assert (result.data["Load_Type"] == "Light_Load").all()

    def test_app_level_aggregate_energy_by_load_type(self):
        """Test aggregating energy usage by load type."""
        result = query_dataset("steel_energy", "aggregate", group_by="Load_Type", agg_column="Usage_kWh", agg_func="mean")
        assert result.success
        assert isinstance(result.data, dict)
        assert "Light_Load" in result.data or "Medium_Load" in result.data

    def test_app_level_top_rows_sintering(self):
        """Test getting top productivity rows."""
        result = query_dataset("sinter_productivity", "get_top_rows", column="Sinter Productivity, T/m2-hr", n=3, ascending=False)
        assert result.success
        assert result.row_count <= 3
        assert result.data["Sinter Productivity, T/m2-hr"].is_monotonic_decreasing

    def test_app_level_error_invalid_dataset(self):
        """Test error handling for invalid dataset ID."""
        result = query_dataset("invalid_dataset", "get_columns")
        assert not result.success
        assert result.error is not None

    def test_app_level_error_invalid_column(self):
        """Test error handling for invalid column."""
        result = query_dataset("sinter_productivity", "get_statistics", column="invalid_column")
        assert not result.success
        assert result.error is not None

    def test_app_level_error_invalid_operator(self):
        """Test error handling for invalid operator."""
        result = query_dataset("sinter_productivity", "filter_rows", column="Sinter Productivity, T/m2-hr", operator="invalid_op", value=1.0)
        assert not result.success
        assert result.error is not None

    def test_query_result_metadata_preservation(self):
        """Test that query results preserve operation metadata."""
        result = query_dataset("steel_energy", "filter_rows", column="Load_Type", operator="eq", value="Light_Load")
        assert result.success
        assert result.dataset_id == "steel_energy"
        assert result.operation == "filter_rows"
        assert result.metadata["filter_column"] == "Load_Type"
        assert result.metadata["filter_operator"] == "eq"
        assert result.metadata["filter_value"] == "Light_Load"

    def test_query_result_empty_handling(self):
        """Test handling of queries that return no results."""
        # Filter for a value that doesn't exist
        result = query_dataset("sinter_productivity", "filter_rows", column="Sinter Productivity, T/m2-hr", operator="gt", value=999.0)
        assert result.success
        assert result.row_count == 0
        assert isinstance(result.data, pd.DataFrame)
        # Empty DataFrame is acceptable for empty results
