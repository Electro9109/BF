"""Blast Furnace department implementation of the Department protocol."""

from __future__ import annotations

import pandas as pd

from departments.blast_furnace import feature_processing as fp
from parse.core.department import Department


class BlastFurnaceDepartment:
    """Blast Furnace department-specific feature schema and processing.

    Delegates to departments.blast_furnace.feature_processing, which remains
    the single source of truth for BF columns, ranges, and parsers. This
    class only adapts that module's existing functions/constants to the
    Department protocol's shape.
    """

    @property
    def department_id(self) -> str:
        return "blast_furnace"

    @property
    def feature_columns(self) -> list[str]:
        """BF chemistry feature columns."""
        return list(fp.CHEM_COLS)

    @property
    def target_columns(self) -> list[str]:
        """BF prediction target columns."""
        return list(fp.TARGET_COLS)

    @property
    def value_ranges(self) -> dict[str, tuple[float, float]]:
        """Domain-knowledge practical ranges for BF scaling."""
        return {
            **fp.CHEM_RANGES,
            **fp.ATM_RANGES,
            **fp.BURDEN_RANGES,
            **fp.INTERACTION_RANGES,
        }

    def load_data(self, path: str = fp.DATA_FILE) -> pd.DataFrame:
        """Load and preprocess BF raw data from Excel file."""
        return fp.load_raw(path)

    def parse_custom_fields(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Parse BF-specific string fields into numeric columns."""
        result = {}
        if "Test Condition" in df.columns:
            result["atmosphere"] = fp._parse_atmosphere(df["Test Condition"])
        if "Burden compositon" in df.columns:
            result["burden"] = fp._encode_burden_numeric(df["Burden compositon"])
        return result
