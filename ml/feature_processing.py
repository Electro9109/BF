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
def _encode_burden(burden_series: pd.Series) -> pd.DataFrame:
    """
    One-hot encode burden string.
    Each unique burden composition becomes a binary column.
    """
    dummies = pd.get_dummies(burden_series, prefix="burden", dtype=float)
    return dummies


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
    if fit_scalers:
        chem_scaler = StandardScaler()
        chem_scaled = chem_scaler.fit_transform(chem)
    else:
        chem_scaler = scalers["chemistry"]
        chem_scaled = chem_scaler.transform(chem)
    parts.append(chem_scaled)
    feature_names += CHEM_COLS

    # 2. Atmosphere
    atm_scaler = None
    if use_atmosphere and CONDITION_COL in df.columns:
        atm = _parse_atmosphere(df[CONDITION_COL])
        if fit_scalers:
            atm_scaler = StandardScaler()
            atm_scaled = atm_scaler.fit_transform(atm)
        else:
            atm_scaler = scalers.get("atmosphere")
            atm_scaled = atm_scaler.transform(atm) if atm_scaler else atm.values
        parts.append(atm_scaled)
        feature_names += atm.columns.tolist()

    # 3. Burden composition (one-hot, no scaling)
    burden_dummies = None
    if use_burden and BURDEN_COL in df.columns:
        burden_dummies = _encode_burden(df[BURDEN_COL])
        parts.append(burden_dummies.values)
        feature_names += burden_dummies.columns.tolist()

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
    if burden_dummies is not None:
        scalers_out["burden_columns"] = burden_dummies.columns.tolist()

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
