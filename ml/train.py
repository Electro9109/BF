"""
train.py
--------
Trains Random Forest and XGBoost regressors for Ts, Tm, and Tm-Ts.

Workflow
--------
  1. Load + featurise data  (feature_processing.py)
  2. 5-fold cross-validation for each model × each target
  3. Print benchmark table  (MAE, RMSE, R²)
  4. Refit winning model on full data  →  save to MLModels/

Usage
-----
  python ml/train.py
  python ml/train.py --targets Ts Tm          # subset of targets
  python ml/train.py --no-burden --no-type    # chemistry + atmosphere only
"""

import argparse
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# XGBoost is optional (not available in all envs)
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    warnings.warn("xgboost not installed — only Random Forest will be trained.")

from ml.feature_processing import load_and_build, TARGET_COLS

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH   = "data_files/data_result.xlsx"
MODEL_DIR   = Path("MLModels")
MODEL_DIR.mkdir(exist_ok=True)


# ── Model definitions ──────────────────────────────────────────────────────
def get_models() -> dict:
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
    return models


# ── Hyperparameter tuning ──────────────────────────────────────────────────
def tune_hyperparameters(model_name: str, X: np.ndarray, y: np.ndarray) -> dict:
    """Tune hyperparameters for a given model and target using RandomizedSearchCV."""
    if model_name == "RandomForest":
        base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        param_dist = {
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_leaf": [1, 2, 4],
            "min_samples_split": [2, 5, 10],
            "max_features": ["sqrt", "log2", None]
        }
    elif model_name == "XGBoost" and HAS_XGB:
        base_model = XGBRegressor(random_state=42, verbosity=0)
        param_dist = {
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9]
        }
    else:
        return {}

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=25,
        scoring="neg_root_mean_squared_error",
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    search.fit(X, y)
    return search.best_params_


# ── CV evaluation ──────────────────────────────────────────────────────────
def cross_validate(model, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mae_scores, rmse_scores, r2_scores = [], [], []

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        m = type(model)(**model.get_params())
        m.fit(X_tr, y_tr)
        preds = m.predict(X_val)

        mae_scores.append(mean_absolute_error(y_val, preds))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
        r2_scores.append(r2_score(y_val, preds))

    return {
        "MAE":  np.mean(mae_scores),
        "RMSE": np.mean(rmse_scores),
        "R2":   np.mean(r2_scores),
        "MAE_std":  np.std(mae_scores),
        "RMSE_std": np.std(rmse_scores),
        "R2_std":   np.std(r2_scores),
    }


# ── Benchmark ─────────────────────────────────────────────────────────────
def run_benchmark(
    X: np.ndarray,
    y_dict: dict,
    targets: list,
    models: dict,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, dict]:
    records = []
    tuned_models_dict = {}
    for target in targets:
        y = y_dict[target]
        for model_name, model in models.items():
            print(f"  Tuning [{model_name}] target={target} ...", end=" ", flush=True)
            best_params = tune_hyperparameters(model_name, X, y)
            
            # Create model instance with tuned parameters
            if model_name == "RandomForest":
                tuned_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
            else:
                tuned_model = XGBRegressor(**best_params, random_state=42, verbosity=0)
                
            scores = cross_validate(tuned_model, X, y, n_splits=n_splits)
            print(f"MAE={scores['MAE']:.2f}  RMSE={scores['RMSE']:.2f}  R²={scores['R2']:.3f}")
            records.append({
                "Target":    target,
                "Model":     model_name,
                "MAE":       round(scores["MAE"],  2),
                "MAE_std":   round(scores["MAE_std"],  2),
                "RMSE":      round(scores["RMSE"], 2),
                "RMSE_std":  round(scores["RMSE_std"], 2),
                "R2":        round(scores["R2"],   3),
                "R2_std":    round(scores["R2_std"],   3),
            })
            tuned_models_dict[(target, model_name)] = tuned_model
    return pd.DataFrame(records), tuned_models_dict


# ── Save models ────────────────────────────────────────────────────────────
def save_models(tuned_models: dict, X: np.ndarray, y_dict: dict, targets: list, results_df: pd.DataFrame):
    """
    Refit each tuned model on the full dataset and save to MLModels/.
    Also saves the best model per target based on lowest RMSE.
    """
    print("\nRefitting on full dataset and saving models...")
    for (target, model_name), model in tuned_models.items():
        model.fit(X, y_dict[target])
        fname = MODEL_DIR / f"{model_name}_{target.replace('-', '_')}.pkl"
        with open(fname, "wb") as f:
            pickle.dump(model, f)
        print(f"  Saved: {fname}")

    # Save best model per target
    for target in targets:
        subset = results_df[results_df["Target"] == target]
        best_row = subset.loc[subset["RMSE"].idxmin()]
        best_name = best_row["Model"]
        src = MODEL_DIR / f"{best_name}_{target.replace('-', '_')}.pkl"
        dst = MODEL_DIR / f"best_{target.replace('-', '_')}.pkl"
        with open(src, "rb") as f:
            obj = pickle.load(f)
        with open(dst, "wb") as f:
            pickle.dump(obj, f)
        print(f"  Best for {target}: {best_name}  ->  {dst}")


# ── Entry point ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train ML models for Ts / Tm / Tm-Ts prediction")
    parser.add_argument("--targets",    nargs="+", default=TARGET_COLS,
                        help="Which targets to train (default: all three)")
    parser.add_argument("--no-atmosphere", action="store_true")
    parser.add_argument("--no-burden",     action="store_true")
    parser.add_argument("--no-type",       action="store_true")
    parser.add_argument("--folds",         type=int, default=5)
    parser.add_argument("--data",          default=DATA_PATH)
    parser.add_argument("--no-save",       action="store_true",
                        help="Skip saving trained models to disk")
    args = parser.parse_args()

    print("=" * 60)
    print("Metallurgical ML — Baseline Benchmark")
    print("=" * 60)
    print(f"Targets       : {args.targets}")
    print(f"Atmosphere    : {not args.no_atmosphere}")
    print(f"Burden        : {not args.no_burden}")
    print(f"Test type     : {not args.no_type}")
    print(f"CV folds      : {args.folds}")
    print()

    # Build features
    print("Loading and featurising data...")
    result = load_and_build(
        path=args.data,
        use_atmosphere=not args.no_atmosphere,
        use_burden=not args.no_burden,
        use_test_type=not args.no_type,
    )
    X         = result["X"]
    y_dict    = result["y"]
    feat_names = result["feature_names"]
    scalers   = result["scalers"]

    print(f"Dataset : {X.shape[0]} samples × {X.shape[1]} features")
    print(f"Features: {feat_names}\n")

    models = get_models()
    print(f"Models  : {list(models.keys())}\n")
    print("Running 5-fold cross-validation...")
    print("-" * 60)

    results_df, tuned_models = run_benchmark(X, y_dict, args.targets, models, n_splits=args.folds)

    # Print table
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS  (5-fold CV)")
    print("=" * 60)
    for target in args.targets:
        print(f"\n  Target: {target}")
        subset = results_df[results_df["Target"] == target]
        print(f"  {'Model':<15} {'MAE':>8} {'±':>6} {'RMSE':>8} {'±':>6} {'R²':>7} {'±':>6}")
        print(f"  {'-'*15} {'-'*8} {'-'*6} {'-'*8} {'-'*6} {'-'*7} {'-'*6}")
        for _, row in subset.iterrows():
            print(
                f"  {row['Model']:<15} "
                f"{row['MAE']:>8.2f} {row['MAE_std']:>6.2f} "
                f"{row['RMSE']:>8.2f} {row['RMSE_std']:>6.2f} "
                f"{row['R2']:>7.3f} {row['R2_std']:>6.3f}"
            )
    print()

    # Save results CSV
    results_path = MODEL_DIR / "benchmark_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Benchmark saved to: {results_path}")

    # Save models and scalers
    if not args.no_save:
        save_models(tuned_models, X, y_dict, args.targets, results_df)

        scalers_path = MODEL_DIR / "scalers.pkl"
        with open(scalers_path, "wb") as f:
            pickle.dump(scalers, f)
        print(f"Scalers saved to : {scalers_path}")

        feat_path = MODEL_DIR / "feature_names.pkl"
        with open(feat_path, "wb") as f:
            pickle.dump(feat_names, f)
        print(f"Feature names    : {feat_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
