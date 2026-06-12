"""
pipeline/hybrid_pipeline.py
─────────────────────────────────
Integration layer: ML prediction + nearest-experiment lookup -> a
quantified natural-language query -> RAG retrieval + generation.

This is the "quantify test conditions and composition to improve the
prediction explanation" piece. It depends on both prediction_pipeline and
rag_pipeline being stable, so build/debug it last.

    result = explain_prediction(features, rag_pipeline)
    # -> {
    #      "prediction": {...},
    #      "nearest_experiments": [...],
    #      "explanation": str,
    #      "retrieved_sources": [...],
    #      "query": str,
    #    }
"""

from config.models import ML_TARGET
from ml.feature_processing import describe_features
from pipeline.prediction_pipeline import predict_and_compare
from pipeline.rag_pipeline import RAGPipeline


def _build_explanation_query(features: dict, prediction: dict, nearest: list[dict]) -> str:
    """
    Turn numeric prediction + input features into a natural-language
    retrieval query — the "quantification" step.
    """
    target = prediction["target"]
    value = prediction["value"]
    feature_desc = describe_features(features)

    query = (
        f"Why does a burden with {feature_desc} result in "
        f"{target} of approximately {value:.1f}? "
        f"Explain the metallurgical mechanism."
    )

    if nearest:
        closest = nearest[0]
        target_col = ML_TARGET.replace("-", "_")
        closest_val = closest.target_col
        closest_val_str = f"{closest_val:.1f}" if isinstance(closest_val, (int, float)) else "N/A"
        query += (
            f" Compare to a similar historical experiment (row {closest['row_index']}, "
            f"similarity {closest['similarity']:.2f}) which had "
            f"{target}={closest_val_str}."
        )

    return query


def explain_prediction(features: dict, rag: RAGPipeline, k: int = 3, top_k: int = 3) -> dict:
    """
    features -> predict -> nearest experiments -> build query -> retrieve -> generate.

    `rag` is a pre-initialized RAGPipeline (so its LLM/index aren't reloaded
    on every call).
    """
    pred_result = predict_and_compare(features, k=k)
    prediction = pred_result["prediction"]
    nearest = pred_result["nearest_experiments"]

    query = _build_explanation_query(features, prediction, nearest)

    chunks = rag.retrieve(query, top_k=top_k)
    explanation = (
        rag.generate(query, chunks)
        if chunks
        else "No relevant theory documents were found to explain this prediction."
    )

    return {
        "prediction": prediction,
        "nearest_experiments": nearest,
        "explanation": explanation,
        "retrieved_sources": chunks,
        "query": query,
        "warnings": pred_result["warnings"],
    }
