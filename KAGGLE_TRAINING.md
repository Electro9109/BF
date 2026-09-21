# Training models off-repo (e.g. on Kaggle)

`ml/train.py` trains RandomForest/XGBoost locally with modest
hyperparameter search. This document is the contract for training
richer models (LightGBM, CatBoost, gradient-boosted ensembles with a
wider/GPU-accelerated search, small neural nets, etc.) somewhere with
more compute — Kaggle notebooks included — and bringing the result
back in, with zero changes to `ml/predictor.py`.

This works because `Predictor._load()` already has no hardcoded
knowledge of RandomForest/XGBoost specifically: it loads
`{model_type}_{target}.pkl` where `model_type` is any string, and
picks "best" per target purely from `benchmark_results.csv`'s RMSE
column. Any pickled object exposing a scikit-learn-style
`.predict(X) -> y` slots in.

## 1. Export the exact features the live app builds

```bash
python ml/export_training_data.py --out MLModels/kaggle_export
```

This calls the same `load_and_build()` entry point `ml/train.py` uses,
with the same defaults — the export can never silently drift from what
the app actually builds. It writes:

| File | Contents |
|---|---|
| `training_data.npz` | `X` (feature matrix), plus `y_Ts`, `y_Tm`, `y_Tm_Ts` (targets) |
| `feature_names.json` | ordered column names for `X`, and the list of target names |
| `scalers.pkl` | fitted `StandardScaler` objects (`chemistry`, `atmosphere`, `burden`, `interaction`) — needed at prediction time, not training time |

Upload `training_data.npz` and `feature_names.json` to Kaggle (or
wherever training happens). `scalers.pkl` is not needed for training —
`X` is already scaled — but keep it; it must travel with whichever
model wins, since `ml/predictor.py` loads scalers separately from
`MLModels/scalers.pkl` at prediction time and expects them to match
the model that produced `X`.

## 2. Train, off-repo

Load `training_data.npz`, pick a target's `y_<target>` array, train
against `X` in that exact column order (see `feature_names.json`).
Whatever library — the only requirement is the saved model exposes
`.predict(X: np.ndarray) -> np.ndarray` matching scikit-learn's
convention (one row in, one number out). Compute MAE/RMSE/R² on a
held-out split or cross-validation, the same way `ml/train.py`'s
`cross_validate()` does, so the numbers are comparable to the existing
benchmark. Save the fitted model with `pickle.dump(model, f)` and
download the `.pkl`.

## 3. Import it back

```bash
python ml/import_trained_model.py \
    --model-path /path/to/downloaded_model.pkl \
    --target Ts \
    --model-name LightGBM \
    --mae 4.2 --rmse 5.8 --r2 0.94
```

Before writing anything, this validates the pickle actually loads,
exposes `.predict()`, and produces one prediction from a dummy row
shaped like the real feature vector — a model that loads but can't
predict on the real feature shape is worse than one that fails to
import outright, since it would otherwise only fail later, silently,
inside the live app. On success it:

- writes the model to `MLModels/{model_name}_{target}.pkl`
- adds/updates its row in `MLModels/benchmark_results.csv`

`Predictor(model_type="best")` (the app's default) will then consider
it automatically, alongside RandomForest and XGBoost, purely by
comparing RMSE.

## Repeat per target

The `Ts`, `Tm`, and `Tm-Ts` targets are trained and imported
independently — nothing requires the same model type to win for all
three. A LightGBM model could win `Ts` while XGBoost still wins `Tm`.

## Scope note

This bridge does not change feature engineering, scaling, or anything
in `departments/blast_furnace/feature_processing.py` — it's purely
export/train-elsewhere/import. If a new model needs features the
current pipeline doesn't build, that's a `feature_processing.py`
change first, which then flows through `load_and_build()` to both the
local and Kaggle paths automatically.
