"""Minimal structured-data querying for PARSE registered datasets.

This module provides safe, constrained operations on registered tabular datasets.
It does not execute arbitrary SQL or LLM-generated queries. Instead, it exposes
a small set of validated operations that can be invoked programmatically or through
a constrained UI.

Operations supported:
- Get column list and types
- Get basic statistics (count, mean, median, min, max, std)
- Filter by column value (equality, range)
- Aggregate by group (mean, sum, count)
- Get top/bottom N rows by column
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from parse.source_catalogue import SourceCatalogue

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result envelope for structured data queries."""
    dataset_id: str
    operation: str
    success: bool
    data: pd.DataFrame | dict[str, Any] | None
    error: str | None = None
    row_count: int = 0
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.data is not None and isinstance(self.data, pd.DataFrame):
            self.row_count = len(self.data)


class StructuredQueryEngine:
    """Safe, constrained query engine for registered tabular datasets."""

    def __init__(self, catalogue: SourceCatalogue | None = None):
        self.catalogue = catalogue or SourceCatalogue()

    def _load_dataset(self, dataset_id: str) -> pd.DataFrame:
        """Load a registered dataset by ID from the catalogue."""
        # For now, load from known paths. In future, use catalogue to resolve
        # normalized artifact location.
        if dataset_id == "sinter_productivity":
            return pd.read_csv("data/raw/sintering/SinterProductivityData.csv")
        elif dataset_id == "steel_energy":
            return pd.read_csv("data/raw/energy/Steel_industry_data.csv")
        elif dataset_id == "blast_furnace":
            from departments.blast_furnace.feature_processing import load_raw
            return load_raw()
        else:
            raise ValueError(f"Unknown dataset_id: {dataset_id}")

    def get_columns(self, dataset_id: str) -> QueryResult:
        """Get column names and types for a dataset."""
        try:
            df = self._load_dataset(dataset_id)
            columns = {
                col: str(dtype)
                for col, dtype in df.dtypes.items()
            }
            return QueryResult(
                dataset_id=dataset_id,
                operation="get_columns",
                success=True,
                data=columns,
                metadata={"column_count": len(columns)},
            )
        except Exception as e:
            return QueryResult(
                dataset_id=dataset_id,
                operation="get_columns",
                success=False,
                data=None,
                error=str(e),
            )

    def get_statistics(self, dataset_id: str, column: str | None = None) -> QueryResult:
        """Get basic statistics for a dataset or specific column."""
        try:
            df = self._load_dataset(dataset_id)
            if column is not None:
                if column not in df.columns:
                    raise ValueError(f"Column '{column}' not found in dataset")
                col_data = df[column]
                if pd.api.types.is_numeric_dtype(col_data):
                    stats = {
                        "count": int(col_data.count()),
                        "mean": float(col_data.mean()) if col_data.count() > 0 else None,
                        "median": float(col_data.median()) if col_data.count() > 0 else None,
                        "min": float(col_data.min()) if col_data.count() > 0 else None,
                        "max": float(col_data.max()) if col_data.count() > 0 else None,
                        "std": float(col_data.std()) if col_data.count() > 0 else None,
                    }
                else:
                    stats = {
                        "count": int(col_data.count()),
                        "unique": int(col_data.nunique()),
                        "top": str(col_data.mode()[0]) if col_data.count() > 0 else None,
                    }
            else:
                # Overall dataset statistics
                stats = {
                    "row_count": len(df),
                    "column_count": len(df.columns),
                    "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
                }
            return QueryResult(
                dataset_id=dataset_id,
                operation="get_statistics",
                success=True,
                data=stats,
            )
        except Exception as e:
            return QueryResult(
                dataset_id=dataset_id,
                operation="get_statistics",
                success=False,
                data=None,
                error=str(e),
            )

    def filter_rows(
        self,
        dataset_id: str,
        column: str,
        operator: str,
        value: Any,
    ) -> QueryResult:
        """Filter rows by column value using safe operators.

        Supported operators: "eq", "ne", "gt", "lt", "ge", "le", "in"
        """
        try:
            df = self._load_dataset(dataset_id)
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in dataset")

            if operator == "eq":
                filtered = df[df[column] == value]
            elif operator == "ne":
                filtered = df[df[column] != value]
            elif operator == "gt":
                filtered = df[df[column] > value]
            elif operator == "lt":
                filtered = df[df[column] < value]
            elif operator == "ge":
                filtered = df[df[column] >= value]
            elif operator == "le":
                filtered = df[df[column] <= value]
            elif operator == "in":
                if not isinstance(value, (list, tuple)):
                    raise ValueError("Operator 'in' requires a list or tuple value")
                filtered = df[df[column].isin(value)]
            else:
                raise ValueError(f"Unsupported operator: {operator}")

            return QueryResult(
                dataset_id=dataset_id,
                operation="filter_rows",
                success=True,
                data=filtered,
                metadata={
                    "filter_column": column,
                    "filter_operator": operator,
                    "filter_value": value,
                },
            )
        except Exception as e:
            return QueryResult(
                dataset_id=dataset_id,
                operation="filter_rows",
                success=False,
                data=None,
                error=str(e),
            )

    def aggregate(
        self,
        dataset_id: str,
        group_by: str | None = None,
        agg_column: str | None = None,
        agg_func: str = "mean",
    ) -> QueryResult:
        """Aggregate data by group (mean, sum, count)."""
        try:
            df = self._load_dataset(dataset_id)
            if group_by is not None and group_by not in df.columns:
                raise ValueError(f"Group column '{group_by}' not found in dataset")
            if agg_column is not None and agg_column not in df.columns:
                raise ValueError(f"Aggregate column '{agg_column}' not found in dataset")

            if group_by is None:
                # Aggregate entire dataset
                if agg_column is None:
                    raise ValueError("Must specify either group_by or agg_column")
                if agg_func == "mean":
                    result = df[agg_column].mean()
                elif agg_func == "sum":
                    result = df[agg_column].sum()
                elif agg_func == "count":
                    result = len(df)
                else:
                    raise ValueError(f"Unsupported aggregate function: {agg_func}")
                return QueryResult(
                    dataset_id=dataset_id,
                    operation="aggregate",
                    success=True,
                    data={"value": float(result)},
                    metadata={"agg_column": agg_column, "agg_func": agg_func},
                )
            else:
                # Group by column
                if agg_column is None:
                    # Count per group
                    result = df.groupby(group_by).size().to_dict()
                else:
                    if agg_func == "mean":
                        result = df.groupby(group_by)[agg_column].mean().to_dict()
                    elif agg_func == "sum":
                        result = df.groupby(group_by)[agg_column].sum().to_dict()
                    elif agg_func == "count":
                        result = df.groupby(group_by)[agg_column].count().to_dict()
                    else:
                        raise ValueError(f"Unsupported aggregate function: {agg_func}")
                return QueryResult(
                    dataset_id=dataset_id,
                    operation="aggregate",
                    success=True,
                    data=result,
                    metadata={
                        "group_by": group_by,
                        "agg_column": agg_column,
                        "agg_func": agg_func,
                    },
                )
        except Exception as e:
            return QueryResult(
                dataset_id=dataset_id,
                operation="aggregate",
                success=False,
                data=None,
                error=str(e),
            )

    def get_top_rows(
        self,
        dataset_id: str,
        column: str,
        n: int = 5,
        ascending: bool = False,
    ) -> QueryResult:
        """Get top N rows by column value."""
        try:
            df = self._load_dataset(dataset_id)
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in dataset")
            if not pd.api.types.is_numeric_dtype(df[column]):
                raise ValueError(f"Column '{column}' is not numeric")

            sorted_df = df.sort_values(column, ascending=ascending).head(n)
            return QueryResult(
                dataset_id=dataset_id,
                operation="get_top_rows",
                success=True,
                data=sorted_df,
                metadata={
                    "sort_column": column,
                    "n": n,
                    "ascending": ascending,
                },
            )
        except Exception as e:
            return QueryResult(
                dataset_id=dataset_id,
                operation="get_top_rows",
                success=False,
                data=None,
                error=str(e),
            )


# Convenience function for direct use
def query_dataset(
    dataset_id: str,
    operation: str,
    **kwargs,
) -> QueryResult:
    """Convenience function to execute a query on a dataset."""
    engine = StructuredQueryEngine()
    if operation == "get_columns":
        return engine.get_columns(dataset_id)
    elif operation == "get_statistics":
        return engine.get_statistics(dataset_id, kwargs.get("column"))
    elif operation == "filter_rows":
        return engine.filter_rows(
            dataset_id,
            kwargs["column"],
            kwargs["operator"],
            kwargs["value"],
        )
    elif operation == "aggregate":
        return engine.aggregate(
            dataset_id,
            kwargs.get("group_by"),
            kwargs.get("agg_column"),
            kwargs.get("agg_func", "mean"),
        )
    elif operation == "get_top_rows":
        return engine.get_top_rows(
            dataset_id,
            kwargs["column"],
            kwargs.get("n", 5),
            kwargs.get("ascending", False),
        )
    else:
        return QueryResult(
            dataset_id=dataset_id,
            operation=operation,
            success=False,
            data=None,
            error=f"Unsupported operation: {operation}",
        )
