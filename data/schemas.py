"""
data/schemas.py
─────────────────
Data contracts shared across the project. Any module producing or
consuming chunks/experiment rows should conform to these shapes.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Chunk:
    """A single retrievable unit of theory text."""
    content: str        # clean text -> sent to LLM
    embed_text: str      # "TOPIC: <t>\nSECTION: <s>\n<content>" -> sent to embedder
    source: str           # path relative to docs_dir
    topic: str             # canonical topic name
    section: str            # normalised section label

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentRow:
    """One row of the SMRF.csv experimental dataset."""
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
    row_index: Optional[int] = None  # original row number (1-based) in SMRF.csv

    def as_dict(self) -> dict:
        return asdict(self)


# Mapping from raw SMRF.csv column names -> ExperimentRow field names.
# Centralised here so a column rename only needs to change this dict.
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
