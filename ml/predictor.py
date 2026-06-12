"""
predictor.py
------------
Loads trained models and predicts Ts / Tm / Tm-Ts for new inputs.

Designed to be called by prediction_pipeline.py or directly.

Usage (standalone)
------------------
  from ml.predictor import Predictor

  p = Predictor()
  result = p.predict(
      chemistry={
          "T Fe %": 57.6, "FeO %": 6.65, "SiO2 %": 4.23,
          "CaO %": 6.93, "Al2O3 %": 2.51, "MgO%": 1.89, "Basicity": 1.64
      },
      test_condition="CO= 40% & N2=60%",
      burden="S1-70%+O1-30%",
      test_type="SO",
  )
  print(result)
  # {"Ts": 1290, "Tm": 1471, "Tm-Ts": 181}
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ml.feature_processing import (
    CHEM_COLS, TARGET_COLS,
    _parse_atmosphere, _encode_burden, _encode_test_type,
)

warnings.filterwarnings("ignore")

MODEL_DIR = Path("MLModels")


class Predictor:
    """
    Loads saved models + scalers and provides predict() for new experiments.

    Parameters
    ----------
    model_dir  : path to the MLModels/ directory
    targets    : which targets to predict (default: all three)
    model_type : 'best' (auto-selects best per target), 'RandomForest', or 'XGBoost'
    """

    def __init__(
        self,
        model_dir: str = "MLModels",
        targets: list = None,
        model_type: str = "best",
    ):
        self.model_dir  = Path(model_dir)
        self.targets    = targets or TARGET_COLS
        self.model_type = model_type
        self._models    = {}
        self._scalers   = None
        self._feat_names = None
        self._burden_columns = None
        self._load()

    # ── Load ──────────────────────────────────────────────────────────────
    def _load(self):
        scalers_path = self.model_dir / "scalers.pkl"
        feat_path    = self.model_dir / "feature_names.pkl"

        if not scalers_path.exists():
            raise FileNotFoundError(
                f"Scalers not found at {scalers_path}. Run train.py first."
            )

        with open(scalers_path, "rb") as f:
            self._scalers = pickle.load(f)
        with open(feat_path, "rb") as f:
            self._feat_names = pickle.load(f)

        self._burden_columns = self._scalers.get("burden_columns", [])

        for target in self.targets:
            safe = target.replace("-", "_")
            if self.model_type == "best":
                model_path = self.model_dir / f"best_{safe}.pkl"
            else:
                model_path = self.model_dir / f"{self.model_type}_{safe}.pkl"

            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")

            with open(model_path, "rb") as f:
                self._models[target] = pickle.load(f)

    # ── Feature builder for single sample ────────────────────────────────
    def _build_row(
        self,
        chemistry: dict,
        test_condition: str = None,
        burden: str = None,
        test_type: str = None,
    ) -> np.ndarray:
        """Build one feature row from raw inputs."""
        parts = []

        # 1. Chemistry
        chem_vals = np.array([[chemistry.get(c, 0.0) for c in CHEM_COLS]])
        chem_scaled = self._scalers["chemistry"].transform(chem_vals)
        parts.append(chem_scaled)

        # 2. Atmosphere
        if "atmosphere" in self._scalers and test_condition:
            atm_df = _parse_atmosphere(pd.Series([test_condition]))
            atm_scaled = self._scalers["atmosphere"].transform(atm_df)
            parts.append(atm_scaled)

        # 3. Burden (one-hot against training columns)
        if self._burden_columns:
            burden_vec = np.zeros((1, len(self._burden_columns)))
            if burden:
                col_name = f"burden_{burden}"
                if col_name in self._burden_columns:
                    idx = self._burden_columns.index(col_name)
                    burden_vec[0, idx] = 1.0
            parts.append(burden_vec)

        # 4. Test type (one-hot)
        type_cols = [c for c in self._feat_names if c.startswith("type_")]
        if type_cols:
            type_vec = np.zeros((1, len(type_cols)))
            if test_type:
                col_name = f"type_{test_type}"
                if col_name in type_cols:
                    idx = type_cols.index(col_name)
                    type_vec[0, idx] = 1.0
            parts.append(type_vec)

        row = np.hstack(parts).astype(float)
        return row

    # ── Public predict ────────────────────────────────────────────────────
    def predict(
        self,
        chemistry: dict,
        test_condition: str = None,
        burden: str = None,
        test_type: str = None,
    ) -> dict:
        """
        Predict Ts, Tm, Tm-Ts for given inputs.

        Parameters
        ----------
        chemistry      : dict of {col_name: value} for the 7 chemistry columns
        test_condition : raw string e.g. 'CO= 40% & N2=60%'
        burden         : burden string e.g. 'S1-70%+O1-30%'
        test_type      : 'SO', 'SOP', or 'P'

        Returns
        -------
        dict  : {"Ts": float, "Tm": float, "Tm-Ts": float}
        """
        row = self._build_row(chemistry, test_condition, burden, test_type)
        predictions = {}
        for target in self.targets:
            val = float(self._models[target].predict(row)[0])
            predictions[target] = round(val, 1)
        return predictions

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict for a DataFrame of raw inputs.
        Expected columns: T Fe %, FeO %, SiO2 %, CaO %, Al2O3 %, MgO%,
                          Basicity, Test Condition, Burden compositon, Test Type
        """
        rows = []
        for _, row in df.iterrows():
            chem = {c: row.get(c, 0.0) for c in CHEM_COLS}
            feat_row = self._build_row(
                chemistry=chem,
                test_condition=row.get("Test Condition"),
                burden=row.get("Burden compositon"),
                test_type=row.get("Test Type"),
            )
            rows.append(feat_row)
        X = np.vstack(rows)

        result = {}
        for target in self.targets:
            result[target] = self._models[target].predict(X).round(1)
        return pd.DataFrame(result)


# ── Quick test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = Predictor()
    sample_chem = {
        "T Fe %": 57.62, "FeO %": 6.65, "SiO2 %": 4.23,
        "CaO %": 6.93, "Al2O3 %": 2.51, "MgO%": 1.89,
        "Basicity": 1.638298,
    }
    result = p.predict(
        chemistry=sample_chem,
    )
    print("Prediction:", result)
