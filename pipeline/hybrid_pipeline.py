"""Prediction, historical comparison, retrieval, and synthesis workflow."""

from config.sinter import ML_TARGET
from config.paths import ML_MODEL_DIR
from config.retrieval import TOP_K
from ml.similarity import find_nearest_experiments
from pipeline.prediction_pipeline import PredictionPipeline
from pipeline.rag_pipeline import answer_question


def _build_explanation_query(result, nearest: list[dict]) -> str:
    query = result.to_rag_query()
    if nearest:
        target_column = ML_TARGET.replace("-", "_")
        closest = nearest[0]
        closest_value = closest.get(target_column, "N/A")
        query += (
            f" Compare this with historical experiment row {closest['row_index']}, "
            f"which had {ML_TARGET}={closest_value} and similarity "
            f"{closest['similarity']:.2f}."
        )
    return query


def explain_prediction(
    features: dict,
    engine,
    tokenizer,
    model,
    model_dir=ML_MODEL_DIR,
    k: int = 3,
    top_k: int = TOP_K,
) -> dict:
    """Run prediction, historical comparison, retrieval, and generation.

    ``features`` contains the same input keys accepted by
    :class:`PredictionPipeline`. ``engine`` is a configured
    :class:`retrieval.retriever.RetrievalEngine`.
    """
    if "chemistry" not in features:
        raise ValueError("Hybrid prediction requires chemistry inputs")

    pipeline = PredictionPipeline(model_dir=model_dir)
    result = pipeline.run(**features)
    nearest = find_nearest_experiments(features["chemistry"], k=k)
    query = _build_explanation_query(result, nearest)
    explanation, matches = answer_question(
        question=query,
        search_query=query,
        engine=engine,
        tokenizer=tokenizer,
        model=model,
        top_n=top_k,
    )

    return {
        "prediction": result.predictions,
        "nearest_experiments": nearest,
        "explanation": explanation,
        "retrieved_sources": matches,
        "query": query,
        "warnings": result.parser_warnings,
    }
