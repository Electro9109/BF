"""Energy department implementation of the Department protocol."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from config.paths import STEEL_ENERGY_CSV
from parse.core.department import Department


ENERGY_FEATURE_COLS = [
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM",
]

ENERGY_CATEGORICAL_COLS = [
    "WeekStatus",
    "Day_of_week",
    "Load_Type",
]

ENERGY_TARGET_COLS = ["Usage_kWh"]

ENERGY_RANGES = {
    "Usage_kWh": (0.0, 200.0),
    "Lagging_Current_Reactive.Power_kVarh": (0.0, 100.0),
    "Leading_Current_Reactive_Power_kVarh": (0.0, 30.0),
    "CO2(tCO2)": (0.0, 0.1),
    "Lagging_Current_Power_Factor": (0.0, 100.0),
    "Leading_Current_Power_Factor": (0.0, 100.0),
    "NSM": (0.0, 86400.0),
}


class EnergyDepartment:
    """Energy & Utilities department schema, ranges, and data loader."""

    @property
    def department_id(self) -> str:
        return "energy"

    @property
    def feature_columns(self) -> list[str]:
        return list(ENERGY_FEATURE_COLS)

    @property
    def target_columns(self) -> list[str]:
        return list(ENERGY_TARGET_COLS)

    @property
    def value_ranges(self) -> dict[str, tuple[float, float]]:
        return dict(ENERGY_RANGES)

    def load_data(self, path: str | Path = STEEL_ENERGY_CSV) -> pd.DataFrame:
        df = pd.read_csv(path)
        if "date" in df.columns:
            # Parse 15-minute timestamps format DD/MM/YYYY HH:MM
            df["datetime"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M")
            df = df.sort_values("datetime").reset_index(drop=True)
        return df

    def parse_custom_fields(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """One-hot encode categorical calendar and load type columns."""
        cats = [c for c in ENERGY_CATEGORICAL_COLS if c in df.columns]
        if not cats:
            return {}
        dummies = pd.get_dummies(df[cats], drop_first=True, dtype=float)
        return {"categoricals": dummies}
