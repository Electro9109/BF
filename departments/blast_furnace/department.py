"""Blast Furnace department implementation of the Department protocol."""

import re
from typing import Any

import pandas as pd

from config.paths import DATA_FILE
from parse.core.department import Department


class BlastFurnaceDepartment:
    """Blast Furnace department-specific feature schema and processing."""
    
    @property
    def department_id(self) -> str:
        return "blast_furnace"
    
    @property
    def feature_columns(self) -> list[str]:
        """BF chemistry feature columns."""
        return ["T Fe %", "FeO %", "SiO2 %", "CaO %", "Al2O3 %", "MgO%", "Basicity"]
    
    @property
    def target_columns(self) -> list[str]:
        """BF prediction target columns."""
        return ["Ts", "Tm", "Tm-Ts"]
    
    @property
    def value_ranges(self) -> dict[str, tuple[float, float]]:
        """Domain-knowledge practical ranges for BF scaling."""
        return {
            # Chemistry ranges
            "T Fe %": (45, 65),
            "FeO %": (5, 25),
            "SiO2 %": (3, 10),
            "CaO %": (5, 15),
            "Al2O3 %": (1, 5),
            "MgO%": (0.5, 3),
            "Basicity": (0.8, 1.6),
            # Atmosphere ranges
            "CO_pct": (20, 50),
            "H2_pct": (0, 12),
            "N2_pct": (40, 80),
            # Burden ranges
            "sinter_pct": (0, 100),
            "ore_pct": (0, 100),
            "pellet_pct": (0, 100),
            "other_pct": (0, 100),
            "num_components": (1, 4),
            # Interaction ranges
            "CO_x_Basicity": (20 * 0.8, 50 * 1.6),
            "Reducibility_Ratio": (0.0, (50 + 12) / 40),
            "Basicity_x_Sinter": (0.8 * 0, 1.6 * 100),
        }
    
    def load_data(self, path: str = DATA_FILE) -> pd.DataFrame:
        """Load and preprocess BF raw data from Excel file."""
        df = pd.read_excel(path, sheet_name="Data Analysis ", header=1)
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)
        
        # The second column (NaN name) contains test type: SO / SOP / P
        cols = df.columns.tolist()
        nan_idx = [i for i, c in enumerate(cols) if pd.isna(c) or str(c).strip() == "nan"]
        if nan_idx:
            cols[nan_idx[0]] = "Test Type"
        df.columns = cols
        
        required_columns = self.feature_columns + self.target_columns
        missing = [column for column in required_columns if column not in df.columns]
        if missing:
            raise ValueError(f"BF data is missing required columns: {missing}")
        
        # Coerce numeric columns
        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df = df.dropna(subset=self.feature_columns + self.target_columns).reset_index(drop=True)
        return df
    
    def parse_custom_fields(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Parse BF-specific string fields into numeric columns."""
        result = {}
        
        # Parse atmosphere from "Test Condition" column
        if "Test Condition" in df.columns:
            atmosphere_records = []
            for val in df["Test Condition"]:
                val = str(val)
                co = float(m.group(1)) if (m := re.search(r"CO=\s*([\d.]+)", val)) else 0.0
                h2 = float(m.group(1)) if (m := re.search(r"H2=\s*([\d.]+)", val)) else 0.0
                n2 = float(m.group(1)) if (m := re.search(r"N2=\s*([\d.]+)", val)) else 0.0
                atmosphere_records.append({"CO_pct": co, "H2_pct": h2, "N2_pct": n2})
            result["atmosphere"] = pd.DataFrame(atmosphere_records)
        
        # Parse burden composition from "Burden compositon" column
        if "Burden compositon" in df.columns:
            burden_records = []
            for val in df["Burden compositon"]:
                val = str(val).upper()
                res = {'sinter_pct': 0.0, 'ore_pct': 0.0, 'pellet_pct': 0.0, 'other_pct': 0.0, 'num_components': 0}
                
                m_100 = re.match(r'100%\s*([A-Z])', val)
                if m_100:
                    letter = m_100.group(1)
                    res['num_components'] = 1
                    if letter == 'S': res['sinter_pct'] = 100.0
                    elif letter == 'O': res['ore_pct'] = 100.0
                    elif letter == 'P': res['pellet_pct'] = 100.0
                    else: res['other_pct'] = 100.0
                else:
                    matches = re.findall(r'([A-Z])[A-Z0-9/]*[-=]?(\d+(?:\.\d+)?)%', val)
                    for letter, pct in matches:
                        pct = float(pct)
                        res['num_components'] += 1
                        if letter == 'S': res['sinter_pct'] += pct
                        elif letter == 'O': res['ore_pct'] += pct
                        elif letter == 'P': res['pellet_pct'] += pct
                        else: res['other_pct'] += pct
                burden_records.append(res)
            result["burden"] = pd.DataFrame(burden_records)
        
        return result
