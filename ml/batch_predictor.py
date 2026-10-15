"""Batch prediction and PARSE provenance integration for tabular sources."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from config.paths import SOURCE_NORMALIZED_DIR
from departments.blast_furnace.feature_processing import CHEM_COLS
from ml.predictor import Predictor
from parse.adapters.bf import prediction_to_analysis
from parse.core.contracts import (
    AnalysisResult,
    Applicability,
    EvidenceRef,
    Issue,
    Provenance,
    SourceRef,
)
from parse.source_catalogue import CatalogueEntry, SourceCatalogue
from pipeline.prediction_pipeline import PredictionResult


class BatchPredictionError(ValueError):
    """Raised when a dataset cannot be featurised or predicted in batch."""


def get_model_metadata(predictor: Predictor) -> dict[str, Any]:
    """Extract model information, types, and benchmark metrics from the loaded predictor."""
    meta: dict[str, Any] = {
        "model_type_setting": predictor.model_type,
        "targets": predictor.targets,
        "models": {tgt: type(m).__name__ for tgt, m in predictor._models.items()},
    }
    if predictor._benchmark is not None and not predictor._benchmark.empty:
        benchmark_dict = {}
        for tgt in predictor.targets:
            subset = predictor._benchmark[predictor._benchmark["Target"] == tgt]
            if not subset.empty:
                best_row = subset.loc[subset["RMSE"].idxmin()]
                benchmark_dict[tgt] = {
                    "best_model": best_row["Model"],
                    "rmse": float(best_row["RMSE"]),
                    "mae": float(best_row["MAE"]),
                    "r2": float(best_row["R2"]),
                }
        meta["benchmark"] = benchmark_dict
    return meta


def predict_dataframe(
    df: pd.DataFrame,
    predictor: Predictor,
    source_ref: SourceRef | None = None,
    dataset_name: str = "dataset",
) -> tuple[pd.DataFrame, list[AnalysisResult]]:
    """Run row-by-row prediction over a DataFrame and return enhanced DataFrame + AnalysisResults.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least the required CHEM_COLS.
        Optionally contains 'Test Condition', 'Burden compositon', 'Test Type'.
    predictor : Predictor
        Fitted Predictor instance.
    source_ref : SourceRef | None
        Provenance source reference to attach to each result.
    dataset_name : str
        Name/label for the dataset.

    Returns
    -------
    tuple[pd.DataFrame, list[AnalysisResult]]
        Enhanced DataFrame with predicted columns ('Pred_Ts', 'Pred_Tm', 'Pred_Tm-Ts', 'Pred_Confidence', 'Pred_Distance')
        and list of PARSE AnalysisResult objects.
    """
    missing_chem = [col for col in CHEM_COLS if col not in df.columns]
    if missing_chem:
        raise BatchPredictionError(
            f"Dataset '{dataset_name}' is missing required chemistry columns: {missing_chem}"
        )

    model_meta = get_model_metadata(predictor)
    out_df = df.copy()
    results: list[AnalysisResult] = []

    pred_ts = []
    pred_tm = []
    pred_diff = []
    pred_conf = []
    pred_dist = []

    for idx, row in df.iterrows():
        chem = {col: float(row[col]) for col in CHEM_COLS}
        cond = str(row["Test Condition"]) if "Test Condition" in df.columns and pd.notna(row["Test Condition"]) else None
        burden = str(row["Burden compositon"]) if "Burden compositon" in df.columns and pd.notna(row["Burden compositon"]) else None
        test_type = str(row["Test Type"]) if "Test Type" in df.columns and pd.notna(row["Test Type"]) else None

        try:
            preds = predictor.predict(
                chemistry=chem,
                test_condition=cond,
                burden=burden,
                test_type=test_type,
            )
            raw_inputs = {
                "chemistry": chem,
                "test_condition": cond,
                "burden": burden,
                "test_type": test_type,
            }
            row_result = PredictionResult(
                predictions=preds,
                inputs=raw_inputs,
                feature_vector=list(predictor.last_feature_vector or []),
                parser_warnings=[],
                similar_experiments=[],
            )
            row_id = f"{dataset_name}-row-{idx}"
            s_ref = source_ref or SourceRef(f"{dataset_name}#row={idx}", "csv", locator=str(idx))
            in_refs = [EvidenceRef(s_ref.source_id, "observed_from", locator=f"row_{idx}")]

            analysis = prediction_to_analysis(
                result=row_result,
                result_id=row_id,
                source_ref=s_ref,
                input_refs=in_refs,
                model_metadata=model_meta,
            )
            results.append(analysis)

            pred_ts.append(preds.get("Ts"))
            pred_tm.append(preds.get("Tm"))
            pred_diff.append(preds.get("Tm-Ts"))
            pred_conf.append(preds.get("confidence", "unknown"))
            pred_dist.append(preds.get("distance", float("nan")))

        except Exception as exc:
            pred_ts.append(float("nan"))
            pred_tm.append(float("nan"))
            pred_diff.append(float("nan"))
            pred_conf.append("failed")
            pred_dist.append(float("nan"))
            fail_id = f"{dataset_name}-row-{idx}-failed"
            s_ref = source_ref or SourceRef(f"{dataset_name}#row={idx}", "csv", locator=str(idx))
            results.append(
                AnalysisResult(
                    result_id=fail_id,
                    outputs={"error": str(exc)},
                    input_refs=[EvidenceRef(s_ref.source_id, "observed_from", locator=f"row_{idx}")],
                    method="bf-sinter-predictor",
                    provenance=Provenance(
                        sources=[s_ref],
                        operation="bf_batch_prediction",
                        method="bf-sinter-predictor",
                        parent_ids=[fail_id],
                    ),
                    applicability=Applicability(limitations=[str(exc)]),
                    issues=[Issue("batch_row_failed", "error", f"Row {idx} prediction failed: {exc}", source=s_ref)],
                )
            )

    out_df["Pred_Ts"] = pred_ts
    out_df["Pred_Tm"] = pred_tm
    out_df["Pred_Tm-Ts"] = pred_diff
    out_df["Pred_Confidence"] = pred_conf
    out_df["Pred_Distance"] = pred_dist

    return out_df, results


def predict_catalogue_dataset(
    resource_id: str,
    catalogue: SourceCatalogue | None = None,
    predictor: Predictor | None = None,
) -> tuple[pd.DataFrame, list[AnalysisResult]]:
    """Load a registered normalised tabular dataset from the catalogue and run batch prediction."""
    cat = catalogue or SourceCatalogue()
    pred = predictor or Predictor()

    entry = cat.get_source(resource_id)
    if entry is None:
        raise KeyError(f"Unknown source resource: {resource_id}")

    dataset_artifacts = [
        art for art in entry.artifacts
        if art.attributes.get("artifact_type") == "normalized_dataset"
    ]
    if not dataset_artifacts:
        raise ValueError(f"Resource '{resource_id}' does not have a normalized tabular dataset artifact")

    art = dataset_artifacts[0]
    csv_rel_path = art.attributes.get("path")
    csv_path = cat.normalized_dir / csv_rel_path
    if not csv_path.is_file():
        raise FileNotFoundError(f"Normalized dataset artifact file missing: {csv_path}")

    df = pd.read_csv(csv_path)
    source_ref = SourceRef(
        source_id=entry.manifest.source.source_id,
        source_type="csv",
        locator=str(csv_rel_path),
        label=entry.manifest.title or entry.manifest.source.label,
    )

    return predict_dataframe(
        df=df,
        predictor=pred,
        source_ref=source_ref,
        dataset_name=entry.manifest.resource_id,
    )
