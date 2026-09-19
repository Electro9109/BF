"""Direct tests for ml/similarity.py -- Task 19.

Both functions in this module do their real data loading via deferred,
no-argument calls (load_and_build(), load_experiments_df()) to files that
aren't present in every environment (data_result.xlsx, ignore/SMRF.csv).
So these tests monkeypatch the data source at its origin module (the exact
attribute the deferred `from X import Y` inside each function resolves at
call time), using real config constants (ML_FEATURES/ML_TARGET) so the
tests exercise the actual similarity math, not a stand-in for it.
"""

import numpy as np
import pandas as pd
import pytest

from config.bf_ml import ML_FEATURES, ML_TARGET
from ml.similarity import find_nearest_experiments, nearest_neighbor_distance


def test_nearest_neighbor_distance_returns_min_euclidean_distance(monkeypatch):
    training_X = np.array([
        [0.0, 0.0],
        [3.0, 4.0],
        [10.0, 10.0],
    ])

    def fake_load_and_build(*args, **kwargs):
        return {"X": training_X}

    monkeypatch.setattr(
        "departments.blast_furnace.feature_processing.load_and_build",
        fake_load_and_build,
    )

    query = np.array([[3.0, 0.0]])  # distance to [0,0]=3, to [3,4]=4, to [10,10]=~12.2
    result = nearest_neighbor_distance(query)

    assert result == pytest.approx(3.0)


def test_find_nearest_experiments_returns_top_k_by_cosine_similarity(monkeypatch):
    target_col = ML_TARGET.replace("-", "_")
    rows = []
    # Row 1: identical to the query -> similarity ~1.0
    rows.append({**{f: 1.0 + i * 0.01 for i, f in enumerate(ML_FEATURES)}, target_col: 1300.0})
    # Row 2: scaled but same direction as the query -> also high cosine similarity
    rows.append({**{f: 2.0 + i * 0.02 for i, f in enumerate(ML_FEATURES)}, target_col: 1350.0})
    # Row 3: deliberately dissimilar direction -> lowest similarity
    rows.append({**{f: (10.0 - i) for i, f in enumerate(ML_FEATURES)}, target_col: 1500.0})
    frame = pd.DataFrame(rows)
    frame["row_index"] = range(1, len(frame) + 1)

    def fake_load_experiments_df(*args, **kwargs):
        return frame

    monkeypatch.setattr(
        "data.experiments_loader.load_experiments_df",
        fake_load_experiments_df,
    )

    query = {f: 1.0 + i * 0.01 for i, f in enumerate(ML_FEATURES)}
    results = find_nearest_experiments(query, k=2)

    assert len(results) == 2
    # Row 1 (identical direction to the query) must be the top match.
    assert results[0]["row_index"] == 1
    assert results[0]["similarity"] >= results[1]["similarity"]
    for entry in results:
        assert set(ML_FEATURES).issubset(entry.keys())
        assert target_col in entry
        assert "similarity" in entry


def test_find_nearest_experiments_drops_rows_missing_required_columns(monkeypatch):
    target_col = ML_TARGET.replace("-", "_")
    complete_row = {f: 1.0 for f in ML_FEATURES}
    complete_row[target_col] = 1300.0
    incomplete_row = {f: (1.0 if f != ML_FEATURES[0] else None) for f in ML_FEATURES}
    incomplete_row[target_col] = 1400.0
    frame = pd.DataFrame([complete_row, incomplete_row])
    frame["row_index"] = range(1, len(frame) + 1)

    monkeypatch.setattr(
        "data.experiments_loader.load_experiments_df",
        lambda *a, **k: frame,
    )

    query = {f: 1.0 for f in ML_FEATURES}
    results = find_nearest_experiments(query, k=5)

    # Only the complete row should survive dropna(subset=valid_cols).
    assert len(results) == 1
    assert results[0]["row_index"] == 1