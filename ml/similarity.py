"""
ml/similarity.py
───────────────────
Finds the nearest historical experiments (from SMRF.csv) to a given
feature set, using cosine similarity over the ML_FEATURES.

Used by pipeline/hybrid_pipeline.py to ground the LLM explanation in real
prior trials ("compare to nearest experiment row 23, which had ...").
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from config.models import ML_FEATURES, ML_TARGET
from data.experiments_loader import load_experiments_df


def find_nearest_experiments(features: dict, k: int = 3) -> list[dict]:
    """
    Return the k most similar historical experiment rows to `features`,
    based on cosine similarity over standardized ML_FEATURES.

    Each result dict contains the original experiment's feature values,
    its ML_TARGET value, its 1-based row_index, and a "similarity" score.
    """
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
