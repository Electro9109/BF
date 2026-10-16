"""Sintering department implementation of the Department protocol."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from config.paths import SINTER_PRODUCTIVITY_CSV
from parse.core.department import Department


# Prospective input features available at prediction time (raw mix & crushing granulometry)
SINTER_PROSPECTIVE_FEATURES = [
    "I/O Fines Total Fe %",
    "I/O Fines SiO2 %",
    "I/O Fines Al2O3 %",
    "I/O Fines CaO %",
    "Flux CaO %",
    "Flux MgO %",
    "Flux Crushing Index",
    "Coke Crushing Index",
]

# Post-process finished sinter features (quality inspection after production)
SINTER_RETROSPECTIVE_FEATURES = [
    "Sinter Total Fe %",
    "Sinter FeO %",
    "Sinter SiO2 %",
    "Sinter Al2O3 %",
    "Sinter CaO %",
    "Sinter MgO %",
    "Sinter +40mm Size, %",
    "Drum Tumbling Index (DTI), %",
]

# All available chemistry/quality features
SINTER_CHEM_COLS = SINTER_PROSPECTIVE_FEATURES + SINTER_RETROSPECTIVE_FEATURES

SINTER_TARGET_COLS = ["Sinter Productivity, T/m2-hr"]

# Domain ranges based on industrial sintering knowledge and dataset bounds
SINTER_RANGES = {
    "I/O Fines Total Fe %": (58.0, 63.0),
    "I/O Fines SiO2 %": (3.0, 6.0),
    "I/O Fines Al2O3 %": (2.0, 4.0),
    "I/O Fines CaO %": (0.2, 1.5),
    "Flux CaO %": (30.0, 42.0),
    "Flux MgO %": (6.0, 12.0),
    "Flux Crushing Index": (65.0, 100.0),
    "Coke Crushing Index": (65.0, 90.0),
    "Sinter Total Fe %": (50.0, 60.0),
    "Sinter FeO %": (7.0, 10.0),
    "Sinter SiO2 %": (4.0, 8.0),
    "Sinter Al2O3 %": (1.8, 3.5),
    "Sinter CaO %": (7.0, 13.0),
    "Sinter MgO %": (2.0, 3.5),
    "Sinter +40mm Size, %": (0.0, 35.0),
    "Drum Tumbling Index (DTI), %": (60.0, 85.0),
    "Sinter Productivity, T/m2-hr": (0.8, 1.5),
}


class SinteringDepartment:
    """Sintering department schema, ranges, and data loader."""

    @property
    def department_id(self) -> str:
        return "sintering"

    @property
    def feature_columns(self) -> list[str]:
        """Prospective features available at prediction time (avoids finished sinter leakage)."""
        return list(SINTER_PROSPECTIVE_FEATURES)

    @property
    def retrospective_columns(self) -> list[str]:
        """Post-process features available only after sintering run is finished."""
        return list(SINTER_RETROSPECTIVE_FEATURES)

    @property
    def target_columns(self) -> list[str]:
        return list(SINTER_TARGET_COLS)

    @property
    def value_ranges(self) -> dict[str, tuple[float, float]]:
        return dict(SINTER_RANGES)

    def load_data(self, path: str | Path = SINTER_PRODUCTIVITY_CSV) -> pd.DataFrame:
        df = pd.read_csv(path)
        if "Sample No" in df.columns:
            # Keep as identifier or index, don't use as predictor
            df = df.set_index("Sample No", drop=False)
        return df

    def parse_custom_fields(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        return {}
