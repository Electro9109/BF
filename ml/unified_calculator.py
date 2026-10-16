"""Unified multi-department prediction registry and runner for PARSE.

Loads and executes verified models for:
- Blast Furnace: Ts, Tm, Tm−Ts softening-melting predictions
- Sintering: Prospective Sinter Productivity (T/m2-hr)
- Energy: Contemporaneous electrical load (Usage_kWh)
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.paths import ML_MODEL_DIR
from departments.blast_furnace.feature_processing import CHEM_COLS, TARGET_COLS
from departments.energy.department import ENERGY_CATEGORICAL_COLS, ENERGY_FEATURE_COLS
from departments.sintering.department import SINTER_PROSPECTIVE_FEATURES
from ml.predictor import PredictionInputError, Predictor

logger = logging.getLogger(__name__)


@dataclass
class DepartmentPredictionResult:
    department: str
    target: str
    predicted_value: float | dict[str, float]
    units: str
    model_name: str
    inputs: dict[str, Any]
    evaluation_metrics: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    confidence: str | None = None

    def to_structured_result(self) -> Any:
        from parse.core.translator_contract import StructuredModuleResult
        preds: dict[str, Any] = {}
        if isinstance(self.predicted_value, dict):
            preds = dict(self.predicted_value)
        else:
            preds[self.target] = self.predicted_value

        return StructuredModuleResult(
            module_type="regression_prediction",
            department=self.department,
            operation=f"{self.department}_{self.target}_prediction",
            dataset_id=f"{self.department}_production_model",
            summary=f"Model {self.model_name} predicted {self.target} = {self.predicted_value} {self.units}.",
            target=self.target,
            units=self.units,
            predictions=preds,
            evaluation_metrics=self.evaluation_metrics,
            validation_warnings=list(self.warnings),
            limitations=list(self.limitations),
            confidence=self.confidence or "medium",
            provenance={
                "model_name": self.model_name,
                "department": self.department,
                "input_features": list(self.inputs.keys()),
            },
        )


class UnifiedCalculator:
    """Manages models and prediction execution across all PARSE operational departments."""

    def __init__(self, model_dir: Path | str = ML_MODEL_DIR):
        self.model_dir = Path(model_dir)
        self._bf_predictor: Predictor | None = None
        self._sinter_pkg: dict[str, Any] | None = None
        self._energy_pkg: dict[str, Any] | None = None

    def get_available_tasks(self) -> list[str]:
        tasks = []
        # Check Blast Furnace models
        if (self.model_dir / "scalers.pkl").exists() and (self.model_dir / "feature_names.pkl").exists():
            tasks.append("Blast Furnace (Softening & Melting Ts / Tm)")
        if (self.model_dir / "sinter_productivity_pipeline.pkl").exists():
            tasks.append("Sinter Plant (Prospective Productivity)")
        if (self.model_dir / "steel_energy_pipeline.pkl").exists():
            tasks.append("Steel Energy (Active Load Usage_kWh)")
        return tasks

    def load_bf_predictor(self, model_type: str = "best") -> Predictor:
        if self._bf_predictor is None or self._bf_predictor.model_type != model_type:
            self._bf_predictor = Predictor(model_dir=str(self.model_dir), model_type=model_type)
        return self._bf_predictor

    def load_sinter_model(self) -> dict[str, Any]:
        if self._sinter_pkg is None:
            path = self.model_dir / "sinter_productivity_pipeline.pkl"
            if not path.exists():
                raise FileNotFoundError(f"Sinter model artifact missing at {path}")
            with open(path, "rb") as f:
                self._sinter_pkg = pickle.load(f)
        return self._sinter_pkg

    def load_energy_model(self) -> dict[str, Any]:
        if self._energy_pkg is None:
            path = self.model_dir / "steel_energy_pipeline.pkl"
            if not path.exists():
                raise FileNotFoundError(f"Energy model artifact missing at {path}")
            with open(path, "rb") as f:
                self._energy_pkg = pickle.load(f)
        return self._energy_pkg

    def predict_blast_furnace(
        self,
        features_dict: dict[str, Any],
        model_type: str = "best",
    ) -> DepartmentPredictionResult:
        predictor = self.load_bf_predictor(model_type=model_type)
        res = predictor.predict(features_dict)
        predictions = {
            "Ts": res.get("Ts"),
            "Tm": res.get("Tm"),
            "Tm-Ts": res.get("Tm-Ts"),
        }
        return DepartmentPredictionResult(
            department="blast_furnace",
            target="Ts / Tm / Tm-Ts",
            predicted_value=predictions,
            units="°C",
            model_name=model_type,
            inputs=features_dict,
            evaluation_metrics={
                "benchmark": "Held-out cross-validation from benchmark_results.csv",
            },
            warnings=res.get("warnings", []),
            limitations=[
                "Predictions valid for standard blast furnace slag and burden compositions.",
                "Estimated using multi-model ensemble (CatBoost/RandomForest).",
                "Range warnings indicate inputs outside training distribution; predictions may be less reliable.",
            ],
            confidence=res.get("confidence", "medium"),
        )

    def predict_sinter_productivity(self, raw_mix_inputs: dict[str, float]) -> DepartmentPredictionResult:
        pkg = self.load_sinter_model()
        features = pkg["features"]
        model = pkg["model"]
        ranges = pkg.get("ranges", {})
        warnings = []

        # Validate inputs
        missing = [f for f in features if f not in raw_mix_inputs]
        if missing:
            raise PredictionInputError(f"Missing required prospective features: {missing}")

        row_values = []
        for f in features:
            val = raw_mix_inputs[f]
            if val is None or not np.isfinite(val):
                raise PredictionInputError(f"Feature '{f}' has invalid non-finite value: {val}")
            # Domain range checking
            if f in ranges:
                low, high = ranges[f]
                if val < low or val > high:
                    warnings.append(f"Input '{f}'={val} is outside training distribution range [{low}, {high}].")
            row_values.append(float(val))

        X = np.array([row_values])
        pred_val = round(float(model.predict(X)[0]), 3)

        return DepartmentPredictionResult(
            department="sintering",
            target=pkg["target"],
            predicted_value=pred_val,
            units=pkg["units"],
            model_name=pkg["model_type"],
            inputs=dict(raw_mix_inputs),
            evaluation_metrics=pkg.get("cv_metrics", {}),
            warnings=warnings,
            limitations=[
                "Model uses prospective raw mix and crushing parameters to prevent finished sinter quality leakage.",
                "Explains ~27% of variance (R²=0.271); unmeasured bed suction, bed depth, and moisture also affect throughput.",
            ],
            confidence="medium" if not warnings else "low",
        )

    def predict_energy_consumption(
        self,
        numeric_inputs: dict[str, float],
        calendar_inputs: dict[str, str],
    ) -> DepartmentPredictionResult:
        pkg = self.load_energy_model()
        model = pkg["model"]
        raw_features = pkg["raw_features"]
        all_features = pkg["all_features"]
        ranges = pkg.get("ranges", {})
        warnings = []

        # Validate numerics
        missing_num = [f for f in raw_features if f not in numeric_inputs]
        if missing_num:
            raise PredictionInputError(f"Missing required numeric energy features: {missing_num}")

        row_dict = {}
        for f in raw_features:
            val = numeric_inputs[f]
            if val is None or not np.isfinite(val):
                raise PredictionInputError(f"Feature '{f}' has invalid non-finite value: {val}")
            if f in ranges:
                low, high = ranges[f]
                if val < low or val > high:
                    warnings.append(f"Feature '{f}'={val} is outside training distribution range [{low}, {high}].")
            row_dict[f] = float(val)

        # Build categorical dummy columns
        cat_df = pd.DataFrame([{
            "WeekStatus": calendar_inputs.get("WeekStatus", "Weekday"),
            "Day_of_week": calendar_inputs.get("Day_of_week", "Monday"),
            "Load_Type": calendar_inputs.get("Load_Type", "Medium_Load"),
        }])
        dummies = pd.get_dummies(cat_df, drop_first=True, dtype=float)

        full_df = pd.DataFrame([row_dict])
        for col in dummies.columns:
            full_df[col] = dummies[col]

        # Reindex to match trained model's feature order exactly
        for col in all_features:
            if col not in full_df.columns:
                full_df[col] = 0.0

        X = full_df[all_features].values
        pred_val = round(float(model.predict(X)[0]), 2)

        combined_inputs = dict(numeric_inputs)
        combined_inputs.update(calendar_inputs)

        return DepartmentPredictionResult(
            department="energy",
            target=pkg["target"],
            predicted_value=pred_val,
            units=pkg["units"],
            model_name=pkg["model_type"],
            inputs=combined_inputs,
            evaluation_metrics=pkg.get("cv_metrics", {}),
            warnings=warnings,
            limitations=[
                "Contemporaneous whole-facility electrical load estimation; not a multi-step future load forecast.",
                "Evaluated using chronological TimeSeriesSplit (MAE=0.73 kWh, R²=0.995).",
            ],
            confidence="high" if not warnings else "medium",
        )
