"""Sinter-specific structured experiment contracts."""

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class ExperimentRow:
    """One row of the Sinter historical experiment dataset."""

    Sinter: float
    Ore: float
    Pellet: float
    T_Fe_pct: float
    FeO_pct: float
    SiO2_pct: float
    CaO_pct: float
    Al2O3_pct: float
    MgO_pct: float
    Basicity: float
    Ts: Optional[float] = None
    Td: Optional[float] = None
    Tm: Optional[float] = None
    Tm_Ts: Optional[float] = None
    DelP: Optional[float] = None
    NDM: Optional[float] = None
    row_index: Optional[int] = None

    def as_dict(self) -> dict:
        return asdict(self)


CSV_COLUMN_MAP = {
    "Sinter": "Sinter",
    "Ore": "Ore",
    "Pellet": "Pellet",
    "T_Fe_%": "T_Fe_pct",
    "FeO_%": "FeO_pct",
    "SiO2_%": "SiO2_pct",
    "CaO_%": "CaO_pct",
    "Al2O3_%": "Al2O3_pct",
    "MgO%": "MgO_pct",
    "Basicity": "Basicity",
    "Ts": "Ts",
    "Td": "Td",
    "Tm": "Tm",
    "Tm-Ts": "Tm_Ts",
    "DelP": "DelP",
    "NDM": "NDM",
}