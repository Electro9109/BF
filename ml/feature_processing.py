"""
feature_processing.py
---------------------
Handles all feature engineering for the ML prediction pipeline.

Input  : raw DataFrame from data_result.xlsx  (Data Analysis sheet)
Output : (X, y_dict, feature_names, scalers)

Feature groups
--------------
  chemistry   : T Fe %, FeO %, SiO2 %, CaO %, Al2O3 %, MgO%, Basicity
  atmosphere  : CO%, H2%, N2%  (parsed from Test Condition string)
  burden      : one-hot encoded burden component ratios
  test_type   : SO / SOP / P  one-hot

Targets
-------
  Ts, Tm, Tm-Ts  (all three; caller selects which to use)
"""

import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ── Column name constants ──────────────────────────────────────────────────
CHEM_COLS = ["T Fe %", "FeO %", "SiO2 %", "CaO %", "Al2O3 %", "MgO%", "Basicity"]
TARGET_COLS = ["Ts", "Tm", "Tm-Ts"]
CONDITION_COL = "Test Condition"
BURDEN_COL = "Burden compositon"
TEST_TYPE_COL = None   # derived from the unnamed second column (SO/SOP/P)

# ── Domain-knowledge practical ranges (fixed, not data-derived) ────────────
CHEM_RANGES = {
    "T Fe %":   (45, 65),
    "FeO %":    (5, 25),
    "SiO2 %":   (3, 10),
    "CaO %":    (5, 15),
    "Al2O3 %":  (1, 5),
    "MgO%":     (0.5, 3),
    "Basicity": (0.8, 1.6),
}

ATM_RANGES = {
    "CO_pct": (20, 50),
    "H2_pct": (0, 12),
    "N2_pct": (40, 80),
}

BURDEN_RANGES = {
    "sinter_pct": (0, 100),
    "ore_pct": (0, 100),
    "pellet_pct": (0, 100),
    "other_pct": (0, 100),
    "num_components": (1, 4),
}

INTERACTION_RANGES = {
    "CO_x_Basicity": (
        ATM_RANGES["CO_pct"][0] * CHEM_RANGES["Basicity"][0],
        ATM_RANGES["CO_pct"][1] * CHEM_RANGES["Basicity"][1],
    ),
    # H2 can be 0, so ratio min is 0; N2 max in denom -> use N2 min for ratio max bound
    "Reducibility_Ratio": (
        0.0,
        (ATM_RANGES["CO_pct"][1] + ATM_RANGES["H2_pct"][1]) / ATM_RANGES["N2_pct"][0],
    ),
    "Basicity_x_Sinter": (
        CHEM_RANGES["Basicity"][0] * BURDEN_RANGES["sinter_pct"][0],
        CHEM_RANGES["Basicity"][1] * BURDEN_RANGES["sinter_pct"][1],
    ),
}


class RangeScaler:
    """Drop-in replacement for StandardScaler using fixed domain ranges."""
    def __init__(self, ranges: dict, columns: list):
        self.lo = np.array([ranges[c][0] for c in columns], dtype=float)
        self.hi = np.array([ranges[c][1] for c in columns], dtype=float)

    def fit_transform(self, X):
        return self.transform(X)

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.lo) / (self.hi - self.lo)


# ── Loader ─────────────────────────────────────────────────────────────────
def load_raw(path: str = "data_files/data_result.xlsx") -> pd.DataFrame:
    """Load and return the cleaned raw DataFrame from the Data Analysis sheet."""
    df = pd.read_excel(path, sheet_name="Data Analysis ", header=1)
    df.columns = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)

    # The second column (NaN name) contains test type: SO / SOP / P
    cols = df.columns.tolist()
    # Rename the NaN column that holds test type
    nan_idx = [i for i, c in enumerate(cols) if pd.isna(c) or str(c).strip() == "nan"]
    if nan_idx:
        cols[nan_idx[0]] = "Test Type"
    df.columns = cols

    # Coerce numeric columns
    for col in CHEM_COLS + TARGET_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=CHEM_COLS + TARGET_COLS).reset_index(drop=True)
    return df


# ── Atmosphere parser ──────────────────────────────────────────────────────
def _parse_atmosphere(condition_series: pd.Series) -> pd.DataFrame:
    """
    Parse 'Test Condition' string into numeric CO%, H2%, N2% columns.
    Example: 'CO= 36%, H2=4% & N2=60%'  →  CO=36, H2=4, N2=60
    """
    records = []
    for val in condition_series:
        val = str(val)
        co = float(m.group(1)) if (m := re.search(r"CO=\s*([\d.]+)", val)) else 0.0
        h2 = float(m.group(1)) if (m := re.search(r"H2=\s*([\d.]+)", val)) else 0.0
        n2 = float(m.group(1)) if (m := re.search(r"N2=\s*([\d.]+)", val)) else 0.0
        records.append({"CO_pct": co, "H2_pct": h2, "N2_pct": n2})
    return pd.DataFrame(records)


# ── Burden encoder ─────────────────────────────────────────────────────────
def _encode_burden_numeric(burden_series: pd.Series) -> pd.DataFrame:
    """
    Parse burden string into numeric component fractions:
    sinter_pct, ore_pct, pellet_pct, other_pct, num_components
    """
    records = []
    for val in burden_series:
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
        records.append(res)
    return pd.DataFrame(records)


# ── Test-type encoder ──────────────────────────────────────────────────────
def _encode_test_type(test_type_series: pd.Series) -> pd.DataFrame:
    """One-hot encode SO / SOP / P."""
    cleaned = test_type_series.fillna("unknown").astype(str).str.strip()
    dummies = pd.get_dummies(cleaned, prefix="type", dtype=float)
    return dummies


# ── Main pipeline ──────────────────────────────────────────────────────────
def build_features(
    df: pd.DataFrame,
    use_atmosphere: bool = True,
    use_burden: bool = True,
    use_test_type: bool = True,
    fit_scalers: bool = True,
    scalers: dict = None,
) -> dict:
    """
    Build feature matrix X and target dict from a raw DataFrame.

    Parameters
    ----------
    df              : raw DataFrame from load_raw()
    use_atmosphere  : include parsed CO/H2/N2 columns
    use_burden      : include one-hot burden composition
    use_test_type   : include SO/SOP/P one-hot
    fit_scalers     : if True, fit new StandardScalers on chemistry + atmosphere
                      if False, transform using provided `scalers`
    scalers         : dict of pre-fitted scalers (required if fit_scalers=False)

    Returns
    -------
    dict with keys:
        X              : np.ndarray  — feature matrix
        y              : dict        — {"Ts": array, "Tm": array, "Tm-Ts": array}
        feature_names  : list[str]
        scalers        : dict        — {"chemistry": scaler, "atmosphere": scaler}
        df_features    : pd.DataFrame  — for inspection / similarity search
    """
    parts = []
    feature_names = []

    # 1. Chemistry (always included, always scaled)
    chem = df[CHEM_COLS].copy().astype(float)
    chem_scaler = RangeScaler(CHEM_RANGES, CHEM_COLS)
    chem_scaled = chem_scaler.transform(chem)
    parts.append(chem_scaled)
    feature_names += CHEM_COLS

    # 2. Atmosphere
    atm_scaler = None
    if use_atmosphere and CONDITION_COL in df.columns:
        atm = _parse_atmosphere(df[CONDITION_COL])
        atm_scaler = RangeScaler(ATM_RANGES, atm.columns.tolist())
        atm_scaled = atm_scaler.transform(atm)
        parts.append(atm_scaled)
        feature_names += atm.columns.tolist()

    # 3. Burden composition
    burden_scaler = None
    burden_df = None
    if use_burden and BURDEN_COL in df.columns:
        burden_df = _encode_burden_numeric(df[BURDEN_COL])
        burden_scaler = RangeScaler(BURDEN_RANGES, burden_df.columns.tolist())
        burden_scaled = burden_scaler.transform(burden_df)
        parts.append(burden_scaled)
        feature_names += burden_df.columns.tolist()

    # 3.5 Interaction features (reducibility and slag chemistry combinations)
    inter_scaler = None
    if use_atmosphere and use_burden and CONDITION_COL in df.columns and BURDEN_COL in df.columns:
        atm_unscaled = _parse_atmosphere(df[CONDITION_COL])
        burden_unscaled = burden_df if burden_df is not None else _encode_burden_numeric(df[BURDEN_COL])
        
        inter_df = pd.DataFrame()
        inter_df["CO_x_Basicity"] = atm_unscaled["CO_pct"] * chem["Basicity"]
        inter_df["Reducibility_Ratio"] = (atm_unscaled["CO_pct"] + atm_unscaled["H2_pct"]) / (atm_unscaled["N2_pct"] + 1e-5)
        inter_df["Basicity_x_Sinter"] = chem["Basicity"] * burden_unscaled["sinter_pct"]
        
        inter_scaler = RangeScaler(INTERACTION_RANGES, inter_df.columns.tolist())
        inter_scaled = inter_scaler.transform(inter_df)
        parts.append(inter_scaled)
        feature_names += inter_df.columns.tolist()

    # 4. Test type (one-hot, no scaling)
    if use_test_type and "Test Type" in df.columns:
        type_dummies = _encode_test_type(df["Test Type"])
        parts.append(type_dummies.values)
        feature_names += type_dummies.columns.tolist()

    X = np.hstack(parts).astype(float)

    y = {col: df[col].values.astype(float) for col in TARGET_COLS}

    scalers_out = {"chemistry": chem_scaler}
    if atm_scaler:
        scalers_out["atmosphere"] = atm_scaler
    if burden_scaler:
        scalers_out["burden"] = burden_scaler
    if inter_scaler:
        scalers_out["interaction"] = inter_scaler

    # Keep a DataFrame version for similarity search / inspection
    df_features = pd.DataFrame(X, columns=feature_names)

    return {
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "scalers": scalers_out,
        "df_features": df_features,
    }


# ── Convenience: build from path ───────────────────────────────────────────
def load_and_build(
    path: str = "data_files/data_result.xlsx",
    use_atmosphere: bool = True,
    use_burden: bool = True,
    use_test_type: bool = True,
) -> dict:
    df = load_raw(path)
    return build_features(
        df,
        use_atmosphere=use_atmosphere,
        use_burden=use_burden,
        use_test_type=use_test_type,
        fit_scalers=True,
    )
