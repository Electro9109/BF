"""
import_trained_model.py
------------------------
Brings a model trained off-repo (e.g. on Kaggle, against the export from
ml/export_training_data.py) into MLModels/, in the exact form
ml/predictor.py's Predictor already knows how to load.

Predictor._load() builds model filenames as f"{model_type}_{target}.pkl"
and picks the "best" model per target purely from benchmark_results.csv's
RMSE column - it has no hardcoded knowledge of any specific model type.
This script is the other half of that contract: it validates a pickle
before trusting it, writes it to the right filename, and records its
metrics so "best" selection sees it.

Usage
-----
  python ml/import_trained_model.py \\
      --model-path /path/to/lightgbm_ts.pkl \\
      --target Ts \\
      --model-name LightGBM \\
      --mae 4.2 --rmse 5.8 --r2 0.94

Validation performed before anything is written
-------------------------------------------------
  - the pickle loads
  - it exposes .predict()
  - .predict() actually runs on a dummy row shaped like the real feature
    vector (from MLModels/kaggle_export/feature_names.json, or
    MLModels/feature_names.pkl if that export isn't present) and returns
    one number per row - a model that loads but can't predict on the real
    feature shape is worse than one that fails to import, since it would
    only fail later, silently, inside the live app.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config.paths import ML_MODEL_DIR
from departments.blast_furnace.feature_processing import TARGET_COLS


class ImportValidationError(ValueError):
    """Raised when a candidate model fails the pre-import checks."""


def _load_feature_names(model_dir: Path) -> list[str]:
    export_spec = model_dir / "kaggle_export" / "feature_names.json"
    if export_spec.exists():
        with open(export_spec) as f:
            return json.load(f)["feature_names"]

    local_spec = model_dir / "feature_names.pkl"
    if local_spec.exists():
        with open(local_spec, "rb") as f:
            return list(pickle.load(f))

    raise ImportValidationError(
        f"No feature_names found under {model_dir} (checked kaggle_export/feature_names.json "
        "and feature_names.pkl). Run ml/export_training_data.py or ml/train.py first."
    )


def _validate_model(model, feature_names: list[str]) -> None:
    if not hasattr(model, "predict"):
        raise ImportValidationError("Loaded object has no .predict() method.")

    dummy_row = np.ones((1, len(feature_names)), dtype=np.float64)
    try:
        prediction = model.predict(dummy_row)
    except Exception as exc:
        raise ImportValidationError(
            f"model.predict() raised on a dummy row shaped {dummy_row.shape} "
            f"(matching {len(feature_names)} exported feature columns): {exc}"
        ) from exc

    prediction = np.asarray(prediction)
    if prediction.shape[0] != 1:
        raise ImportValidationError(
            f"model.predict() on one row returned {prediction.shape[0]} outputs, expected 1."
        )


def import_trained_model(
    model_path: str,
    target: str,
    model_name: str,
    mae: float,
    rmse: float,
    r2: float,
    model_dir: str = ML_MODEL_DIR,
) -> Path:
    """Validate and install a trained model, updating benchmark_results.csv.

    Returns the path the model was written to.
    """
    if target not in TARGET_COLS:
        raise ImportValidationError(f"Unknown target {target!r}, expected one of {TARGET_COLS}.")

    model_dir = Path(model_dir)
    feature_names = _load_feature_names(model_dir)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    _validate_model(model, feature_names)

    safe_target = target.replace("-", "_")
    dest = model_dir / f"{model_name}_{safe_target}.pkl"
    with open(dest, "wb") as f:
        pickle.dump(model, f)

    benchmark_path = model_dir / "benchmark_results.csv"
    new_row = {
        "Target": target,
        "Model": model_name,
        "MAE": round(mae, 2),
        "MAE_std": 0.0,
        "RMSE": round(rmse, 2),
        "RMSE_std": 0.0,
        "R2": round(r2, 3),
        "R2_std": 0.0,
    }
    if benchmark_path.exists():
        df = pd.read_csv(benchmark_path)
        df = df[~((df["Target"] == target) & (df["Model"] == model_name))]
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])
    df.to_csv(benchmark_path, index=False)

    return dest


def main():
    parser = argparse.ArgumentParser(
        description="Import a model trained off-repo into MLModels/, validated against the real feature shape."
    )
    parser.add_argument("--model-path", required=True, help="Path to the trained model's .pkl file")
    parser.add_argument("--target", required=True, choices=TARGET_COLS)
    parser.add_argument("--model-name", required=True, help="Short name, e.g. LightGBM, CatBoost")
    parser.add_argument("--mae", type=float, required=True, help="Mean absolute error, computed on Kaggle")
    parser.add_argument("--rmse", type=float, required=True, help="RMSE, computed on Kaggle")
    parser.add_argument("--r2", type=float, required=True, help="R^2, computed on Kaggle")
    parser.add_argument("--model-dir", default=ML_MODEL_DIR)
    args = parser.parse_args()

    dest = import_trained_model(
        model_path=args.model_path,
        target=args.target,
        model_name=args.model_name,
        mae=args.mae,
        rmse=args.rmse,
        r2=args.r2,
        model_dir=args.model_dir,
    )
    print(f"Imported and validated: {dest}")
    print("benchmark_results.csv updated - Predictor(model_type='best') will now consider this model.")


if __name__ == "__main__":
    main()
