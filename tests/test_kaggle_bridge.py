"""Direct tests for ml/export_training_data.py and ml/import_trained_model.py
-- Task 22 Part B.

export_training_data() calls load_and_build(), which needs data_result.xlsx
(not present in every environment, same constraint as Task 19's similarity
tests) -- so it's monkeypatched at its origin, same pattern as
tests/test_similarity.py.
"""

import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from ml.export_training_data import export_training_data
from ml.import_trained_model import ImportValidationError, import_trained_model


FEATURE_NAMES = ["T Fe %", "FeO %", "SiO2 %"]


def _fake_build_result():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, len(FEATURE_NAMES)))
    return {
        "X": X,
        "y": {
            "Ts": rng.normal(1300, 10, size=20),
            "Tm": rng.normal(1450, 10, size=20),
            "Tm-Ts": rng.normal(150, 5, size=20),
        },
        "feature_names": FEATURE_NAMES,
        "scalers": {"chemistry": "fake_scaler_object"},
    }


def test_export_training_data_writes_expected_files(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "ml.export_training_data.load_and_build",
        lambda *args, **kwargs: _fake_build_result(),
    )

    out = export_training_data(out_dir=str(tmp_path))

    assert (out / "training_data.npz").exists()
    assert (out / "feature_names.json").exists()
    assert (out / "scalers.pkl").exists()

    data = np.load(out / "training_data.npz")
    assert data["X"].shape == (20, 3)
    assert "y_Ts" in data
    assert "y_Tm_Ts" in data  # '-' sanitized to '_' in the array name

    spec = json.loads((out / "feature_names.json").read_text())
    assert spec["feature_names"] == FEATURE_NAMES
    assert set(spec["targets"]) == {"Ts", "Tm", "Tm-Ts"}


def _write_export(tmp_path):
    export_dir = tmp_path / "kaggle_export"
    export_dir.mkdir()
    (export_dir / "feature_names.json").write_text(
        json.dumps({"feature_names": FEATURE_NAMES, "targets": ["Ts", "Tm", "Tm-Ts"]})
    )
    return tmp_path


def test_import_trained_model_round_trip(tmp_path):
    model_dir = _write_export(tmp_path)

    X = np.random.default_rng(1).normal(size=(20, len(FEATURE_NAMES)))
    y = np.random.default_rng(2).normal(1300, 10, size=20)
    model = LinearRegression().fit(X, y)
    model_path = tmp_path / "candidate_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    dest = import_trained_model(
        model_path=str(model_path),
        target="Ts",
        model_name="LinearRegression",
        mae=4.2,
        rmse=5.8,
        r2=0.94,
        model_dir=str(model_dir),
    )

    assert dest.exists()
    with open(dest, "rb") as f:
        loaded = pickle.load(f)
    assert loaded.predict(X[:1]).shape == (1,)

    benchmark = pd.read_csv(model_dir / "benchmark_results.csv")
    row = benchmark[(benchmark["Target"] == "Ts") & (benchmark["Model"] == "LinearRegression")]
    assert len(row) == 1
    assert row.iloc[0]["RMSE"] == pytest.approx(5.8)


def test_import_trained_model_updates_existing_row_instead_of_duplicating(tmp_path):
    model_dir = _write_export(tmp_path)
    X = np.random.default_rng(1).normal(size=(20, len(FEATURE_NAMES)))
    y = np.random.default_rng(2).normal(1300, 10, size=20)
    model = LinearRegression().fit(X, y)
    model_path = tmp_path / "candidate_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    import_trained_model(str(model_path), "Ts", "LinearRegression", 4.2, 5.8, 0.94, str(model_dir))
    import_trained_model(str(model_path), "Ts", "LinearRegression", 3.0, 4.0, 0.97, str(model_dir))

    benchmark = pd.read_csv(model_dir / "benchmark_results.csv")
    row = benchmark[(benchmark["Target"] == "Ts") & (benchmark["Model"] == "LinearRegression")]
    assert len(row) == 1
    assert row.iloc[0]["RMSE"] == pytest.approx(4.0)


def test_import_trained_model_rejects_object_without_predict(tmp_path):
    model_dir = _write_export(tmp_path)
    model_path = tmp_path / "not_a_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"just": "a dict"}, f)

    with pytest.raises(ImportValidationError, match="no .predict"):
        import_trained_model(str(model_path), "Ts", "Bogus", 1.0, 1.0, 1.0, str(model_dir))


class _WrongShapeModel:
    """Module-level (picklable) stand-in for a model expecting the wrong feature count."""

    def predict(self, X):
        if X.shape[1] != len(FEATURE_NAMES) + 5:
            raise ValueError("expected more columns")
        return np.zeros(X.shape[0])


def test_import_trained_model_rejects_wrong_feature_shape(tmp_path):
    model_dir = _write_export(tmp_path)

    model_path = tmp_path / "wrong_shape.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(_WrongShapeModel(), f)

    with pytest.raises(ImportValidationError, match="predict.*raised"):
        import_trained_model(str(model_path), "Ts", "WrongShape", 1.0, 1.0, 1.0, str(model_dir))


def test_import_trained_model_rejects_unknown_target(tmp_path):
    model_dir = _write_export(tmp_path)
    model = LinearRegression().fit(np.ones((5, 3)), np.ones(5))
    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with pytest.raises(ImportValidationError, match="Unknown target"):
        import_trained_model(str(model_path), "NotATarget", "X", 1.0, 1.0, 1.0, str(model_dir))
