"""
ml/similarity.py
───────────────────
Nearest-neighbour utilities for the ML predictor.

  nearest_neighbor_distance(query_row)
      Returns the minimum Euclidean distance from the scaled query feature
      row to the scaled training set — used to estimate prediction confidence.

  find_nearest_experiments(features, k)
      Returns the k most similar historical rows based on cosine similarity.
      Requires EXPERIMENTS_CSV to be configured; used by hybrid_pipeline.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def nearest_neighbor_distance(query_scaled: np.ndarray) -> float:
    """
    Compute the minimum Euclidean distance from a scaled query row to
    the scaled training set features.

    Parameters
    ----------
    query_scaled : np.ndarray, shape (1, n_features)
        Feature row already scaled by the same scalers used in training.

    Returns
    -------
    float  — minimum Euclidean distance; smaller == closer to training data.
    """
    from ml.feature_processing import load_and_build
    res = load_and_build()
    X = res["X"]   # shape (n_samples, n_features), already scaled
    dists = np.linalg.norm(X - query_scaled, axis=1)
    return float(np.min(dists))


def find_nearest_experiments(features: dict, k: int = 3) -> list:
    """
    Return the k most similar historical experiment rows to `features`,
    based on cosine similarity over standardised ML_FEATURES.

    Each result dict contains the original experiment's feature values,
    its ML_TARGET value, its 1-based row_index, and a 'similarity' score.

    NOTE: uses Sinter configuration from config.sinter and the historical
    experiment loader.
    """
    from sklearn.preprocessing import StandardScaler
    from config.sinter import ML_FEATURES, ML_TARGET
    from data.experiments_loader import load_experiments_df

    df = load_experiments_df()
    target_col = ML_TARGET.replace("-", "_")

    valid_cols = ML_FEATURES + [target_col]
    df = df.dropna(subset=valid_cols).reset_index(drop=True)

    X = df[ML_FEATURES].values.astype(np.float64)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    query = np.array([[float(features[f]) for f in ML_FEATURES]], dtype=np.float64)
    query_scaled = scaler.transform(query)

    sims = cosine_similarity(query_scaled, X_scaled)[0]
    top_idx = np.argsort(sims)[::-1][:k]

    results = []
    for idx in top_idx:
        row = df.iloc[idx]
        entry = {f: float(row[f]) for f in ML_FEATURES}
        entry[target_col] = float(row[target_col])
        entry["row_index"] = int(row["row_index"])
        entry["similarity"] = float(sims[idx])
        results.append(entry)

    return results

