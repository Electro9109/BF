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
    _parse_atmosphere, _encode_burden_numeric, _encode_test_type,
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

        # Load benchmark results for per-target best-model selection
        benchmark_path = self.model_dir / "benchmark_results.csv"
        self._benchmark = None
        if benchmark_path.exists():
            import pandas as _pd
            self._benchmark = _pd.read_csv(benchmark_path)

        self._burden_columns = self._scalers.get("burden_columns", [])

        for target in self.targets:
            safe = target.replace("-", "_")
            if self.model_type == "best" and self._benchmark is not None:
                # Independently select best model per target based on lowest RMSE
                subset = self._benchmark[self._benchmark["Target"] == target]
                if not subset.empty:
                    best_name = subset.loc[subset["RMSE"].idxmin(), "Model"]
                    model_path = self.model_dir / f"{best_name}_{safe}.pkl"
                else:
                    model_path = self.model_dir / f"best_{safe}.pkl"
            elif self.model_type == "best":
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
        parsed_condition: dict = None,
    ) -> np.ndarray:
        """Build one feature row from raw inputs or parsed conditions."""
        parts = []

        # 1. Chemistry
        chem_vals = np.array([[chemistry.get(c, 0.0) for c in CHEM_COLS]])
        chem_scaled = self._scalers["chemistry"].transform(chem_vals)
        parts.append(chem_scaled)

        # 2. Atmosphere
        if "atmosphere" in self._scalers:
            if parsed_condition:
                atm_df = pd.DataFrame([{
                    "CO_pct": parsed_condition.get("CO_pct", 0.0),
                    "H2_pct": parsed_condition.get("H2_pct", 0.0),
                    "N2_pct": parsed_condition.get("N2_pct", 0.0)
                }])
            else:
                from ml.feature_processing import _parse_atmosphere
                atm_df = _parse_atmosphere(pd.Series([test_condition or ""]))
            atm_scaled = self._scalers["atmosphere"].transform(atm_df)
            parts.append(atm_scaled)

        # 3. Burden
        if "burden" in self._scalers:
            if parsed_condition:
                burden_df = pd.DataFrame([{
                    "sinter_pct": parsed_condition.get("sinter_pct", 0.0),
                    "ore_pct": parsed_condition.get("ore_pct", 0.0),
                    "pellet_pct": parsed_condition.get("pellet_pct", 0.0),
                    "other_pct": parsed_condition.get("other_pct", 0.0),
                    "num_components": parsed_condition.get("num_components", 0)
                }])
            else:
                from ml.feature_processing import _encode_burden_numeric
                burden_df = _encode_burden_numeric(pd.Series([burden or ""]))
            burden_scaled = self._scalers["burden"].transform(burden_df)
            parts.append(burden_scaled)

        # 3.5 Interaction features
        if "interaction" in self._scalers:
            if parsed_condition:
                co_val = parsed_condition.get("CO_pct", 0.0)
                h2_val = parsed_condition.get("H2_pct", 0.0)
                n2_val = parsed_condition.get("N2_pct", 0.0)
                sinter_val = parsed_condition.get("sinter_pct", 0.0)
            else:
                from ml.feature_processing import _parse_atmosphere, _encode_burden_numeric
                atm_unscaled = _parse_atmosphere(pd.Series([test_condition or ""]))
                co_val = float(atm_unscaled["CO_pct"].iloc[0])
                h2_val = float(atm_unscaled["H2_pct"].iloc[0])
                n2_val = float(atm_unscaled["N2_pct"].iloc[0])
                
                burden_unscaled = _encode_burden_numeric(pd.Series([burden or ""]))
                sinter_val = float(burden_unscaled["sinter_pct"].iloc[0])
                
            basicity_val = chemistry.get("Basicity", 0.0)
            
            inter_df = pd.DataFrame([{
                "CO_x_Basicity": co_val * basicity_val,
                "Reducibility_Ratio": (co_val + h2_val) / (n2_val + 1e-5),
                "Basicity_x_Sinter": basicity_val * sinter_val
            }])
            inter_scaled = self._scalers["interaction"].transform(inter_df)
            parts.append(inter_scaled)

        # 4. Test type (one-hot)
        type_cols = [c for c in self._feat_names if c.startswith("type_")]
        if type_cols:
            type_vec = np.zeros((1, len(type_cols)))
            actual_type = test_type
            if parsed_condition and "test_type" in parsed_condition:
                actual_type = parsed_condition["test_type"]
                
            if actual_type:
                col_name = f"type_{actual_type}"
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
        parsed_condition: dict = None,
    ) -> dict:
        """
        Predict Ts, Tm, Tm-Ts for given inputs.

        Parameters
        ----------
        chemistry      : dict of {col_name: value} for the 7 chemistry columns
        test_condition : raw string e.g. 'CO= 40% & N2=60%'
        burden         : burden string e.g. 'S1-70%+O1-30%'
        test_type      : 'SO', 'SOP', or 'P'
        parsed_condition : optional dict of already parsed numeric conditions

        Returns
        -------
        dict  : {"Ts": float, "Tm": float, "Tm-Ts": float}
        """
        row = self._build_row(chemistry, test_condition, burden, test_type, parsed_condition)
        predictions = {}
        for target in self.targets:
            val = float(self._models[target].predict(row)[0])
            predictions[target] = round(val, 1)
            
        # Compute nearest neighbor distance for confidence indication
        try:
            from ml.similarity import nearest_neighbor_distance
            dist = nearest_neighbor_distance(row)
            if dist <= 1.05:
                confidence = "high"
            elif dist <= 1.82:
                confidence = "medium"
            else:
                confidence = "low"
        except Exception:
            confidence = "medium"
            dist = 999.0
            
        predictions["confidence"] = confidence
        predictions["distance"] = round(dist, 3)
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
