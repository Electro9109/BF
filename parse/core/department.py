"""Department-specific feature schema and processing contracts.

PROVISIONAL/V1: This contract is expected to be revised once a second department
is implemented. Currently only Blast Furnace implements this contract.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class Department(Protocol):
    """Protocol for department-specific feature schema and processing.
    
    Separates "what varies by department" from generic pipeline logic:
    - Column/loader schema
    - Practical value ranges for scaling
    - String-format parsers specific to data recording conventions
    - Target variable definitions
    """
    
    @property
    def department_id(self) -> str:
        """Unique identifier for this department (e.g., 'blast_furnace')."""
        ...
    
    @property
    def feature_columns(self) -> list[str]:
        """Column names for raw features (e.g., chemistry columns)."""
        ...
    
    @property
    def target_columns(self) -> list[str]:
        """Column names for prediction targets (e.g., Ts, Tm, Tm-Ts)."""
        ...
    
    @property
    def value_ranges(self) -> dict[str, tuple[float, float]]:
        """Domain-knowledge practical ranges for scaling (column -> (min, max))."""
        ...
    
    def load_data(self, path: str) -> pd.DataFrame:
        """Load and preprocess department-specific raw data from file."""
        ...
    
    def parse_custom_fields(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Parse department-specific string fields into numeric columns.
        
        Returns a dict mapping field group names to parsed DataFrames.
        Example: {'atmosphere': DataFrame(CO_pct, H2_pct, N2_pct), ...}
        """
        ...
