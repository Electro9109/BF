"""
export_training_data.py
------------------------
Exports the exact feature matrix/targets/scalers that ml/train.py trains
against, in a portable form a Kaggle notebook (or any external environment)
can consume without needing this repo.

This deliberately calls the same load_and_build() entry point train.py
uses, with the same defaults, so exported features can never silently
drift from what the live app actually builds.

Usage
-----
  python ml/export_training_data.py
  python ml/export_training_data.py --out MLModels/kaggle_export

Output (in --out, default MLModels/kaggle_export/)
----------------------------------------------------
  training_data.npz   X, and one array per target ("y_Ts", "y_Tm", "y_Tm_Ts")
  feature_names.json  ordered list of column names X's columns correspond to
  scalers.pkl         the fitted scaler objects (chemistry/atmosphere/burden/
                       interaction) - a model trained on X should be paired
                       with these at prediction time via ml/predictor.py

A model trained against training_data.npz must:
  - accept X in this exact column order (see feature_names.json)
  - expose .predict(X: np.ndarray) -> np.ndarray, scikit-learn style

See KAGGLE_TRAINING.md for the full contract and the import step
(ml/import_trained_model.py) that brings a trained model back in.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from config.paths import DATA_FILE, ML_MODEL_DIR
from departments.blast_furnace.department import BlastFurnaceDepartment
from departments.blast_furnace.feature_processing import load_and_build


def export_training_data(data_path: str = DATA_FILE, out_dir: str = None) -> Path:
    """Build features via load_and_build() and write a portable export.

    Returns the output directory path.
    """
    out = Path(out_dir) if out_dir else Path(ML_MODEL_DIR) / "kaggle_export"
    out.mkdir(parents=True, exist_ok=True)

    result = load_and_build(path=data_path, department=BlastFurnaceDepartment())
    X = result["X"]
    y_dict = result["y"]
    feature_names = result["feature_names"]
    scalers = result["scalers"]

    arrays = {"X": X}
    for target, y in y_dict.items():
        arrays[f"y_{target.replace('-', '_')}"] = y
    np.savez(out / "training_data.npz", **arrays)

    with open(out / "feature_names.json", "w") as f:
        json.dump(
            {"feature_names": feature_names, "targets": list(y_dict.keys())},
            f,
            indent=2,
        )

    with open(out / "scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Export the training feature matrix for off-repo (e.g. Kaggle) model training."
    )
    parser.add_argument("--data", default=DATA_FILE)
    parser.add_argument("--out", default=None, help="Output dir (default: MLModels/kaggle_export)")
    args = parser.parse_args()

    out = export_training_data(data_path=args.data, out_dir=args.out)
    print(f"Exported training data to: {out}")
    print(f"  {out / 'training_data.npz'}")
    print(f"  {out / 'feature_names.json'}")
    print(f"  {out / 'scalers.pkl'}")


if __name__ == "__main__":
    main()
