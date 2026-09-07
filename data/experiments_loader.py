"""
data/experiments_loader.py
─────────────────────────────
Reads the configured historical experiment CSV -> list[ExperimentRow], and performs the
physical-integrity repair identified in the analysis notebook
(Step 3: Tm-Ts must equal Tm - Ts).

Used by ml/similarity.py for nearest-experiment lookup.
"""

import warnings

import pandas as pd

from config.paths import EXPERIMENTS_CSV
from data.sinter_schemas import CSV_COLUMN_MAP, ExperimentRow


class ExperimentValidationWarning(UserWarning):
    """The source contains incomplete values that are retained with provenance."""


def _validate_experiment_frame(df: pd.DataFrame) -> list[str]:
    issues = []
    numeric_columns = list(CSV_COLUMN_MAP.values())
    for column in numeric_columns:
        converted = pd.to_numeric(df[column], errors="coerce")
        invalid = df[column].notna() & converted.isna()
        if invalid.any():
            issues.append(f"{column}: {int(invalid.sum())} non-numeric value(s)")
    incomplete = df[numeric_columns].isna().sum()
    for column, count in incomplete.items():
        if count:
            issues.append(f"{column}: {int(count)} missing value(s)")
    return issues


def load_experiments_df(csv_path: str | None = None, repair: bool = True) -> pd.DataFrame:
    """
    Load the historical experiment CSV, standardize column names, and (optionally) repair
    Tm-Ts integrity violations (Tm-Ts must equal Tm - Ts).

    Returns a DataFrame with renamed columns matching ExperimentRow fields,
    plus a 1-based "row_index" column for traceability back to the CSV.
    """
    csv_path = csv_path or str(EXPERIMENTS_CSV)

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.replace(" ", "_")

    missing = [c for c in CSV_COLUMN_MAP if c not in df.columns]
    if missing:
        raise ValueError(
            f"Experiment CSV is missing expected columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df.rename(columns=CSV_COLUMN_MAP)
    df["row_index"] = range(1, len(df) + 1)

    issues = _validate_experiment_frame(df)
    if issues:
        warnings.warn(
            "Incomplete or malformed experiment values retained: " + "; ".join(issues),
            ExperimentValidationWarning,
            stacklevel=2,
        )

    if repair and {"Ts", "Tm", "Tm_Ts"}.issubset(df.columns):
        calculated = df["Tm"] - df["Ts"]
        mismatch = (df["Tm_Ts"] - calculated).abs() > 1e-3
        if mismatch.any():
            df.loc[mismatch, "Tm_Ts"] = calculated[mismatch]

    return df


def load_experiments(csv_path: str | None = None, repair: bool = True) -> list[ExperimentRow]:
    """Return historical experiment rows as dataclasses."""
    df = load_experiments_df(csv_path, repair=repair)
    fields = set(ExperimentRow.__dataclass_fields__)
    records = []
    for _, row in df.iterrows():
        kwargs = {k: row[k] for k in fields if k in df.columns}
        records.append(ExperimentRow(**kwargs))
    return records