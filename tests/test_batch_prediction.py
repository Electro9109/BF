import pandas as pd
import pytest

from departments.blast_furnace.feature_processing import CHEM_COLS
from ml.batch_predictor import (
    BatchPredictionError,
    get_model_metadata,
    predict_catalogue_dataset,
    predict_dataframe,
)
from ml.predictor import Predictor
from parse.core.contracts import AnalysisResult
from parse.source_catalogue import SourceCatalogue


def test_get_model_metadata():
    pred = Predictor()
    meta = get_model_metadata(pred)
    assert "models" in meta
    assert "targets" in meta
    assert "Tm" in meta["models"]
    assert "best_model" in meta["benchmark"]["Tm"]


def test_predict_dataframe_success():
    pred = Predictor()
    test_df = pd.DataFrame([{
        "T Fe %": 57.6,
        "FeO %": 6.65,
        "SiO2 %": 4.23,
        "CaO %": 6.93,
        "Al2O3 %": 2.51,
        "MgO%": 1.89,
        "Basicity": 1.64,
        "Test Condition": "CO= 40% & N2=60%",
        "Burden compositon": "S1-70%+O1-30%",
        "Test Type": "SO",
    }])
    out_df, analyses = predict_dataframe(test_df, pred, dataset_name="test_batch")

    assert len(out_df) == 1
    assert "Pred_Ts" in out_df.columns
    assert "Pred_Tm" in out_df.columns
    assert "Pred_Tm-Ts" in out_df.columns
    assert pd.notna(out_df.loc[0, "Pred_Ts"])
    assert pd.notna(out_df.loc[0, "Pred_Tm"])

    assert len(analyses) == 1
    analysis = analyses[0]
    assert isinstance(analysis, AnalysisResult)
    assert analysis.method == "bf-sinter-predictor"
    assert "Ts" in analysis.outputs
    assert "model_metadata" in analysis.outputs
    assert analysis.outputs["model_metadata"]["models"]["Tm"] == "CatBoostRegressor"


def test_predict_dataframe_missing_chemistry_raises():
    pred = Predictor()
    test_df = pd.DataFrame([{"T Fe %": 57.6}])
    with pytest.raises(BatchPredictionError, match="missing required chemistry"):
        predict_dataframe(test_df, pred)


def test_predict_catalogue_dataset_roundtrip(tmp_path):
    cat = SourceCatalogue(
        database_path=tmp_path / "catalogue.db",
        originals_dir=tmp_path / "originals",
        public_originals_dir=tmp_path / "public_originals",
        normalized_dir=tmp_path / "normalized",
    )
    # Register a tabular dataset
    csv_bytes = (
        "T Fe %,FeO %,SiO2 %,CaO %,Al2O3 %,MgO%,Basicity,Test Condition,Burden compositon,Test Type\n"
        "57.6,6.65,4.23,6.93,2.51,1.89,1.64,CO= 40% & N2=60%,S1-70%+O1-30%,SO\n"
    ).encode("utf-8")

    entry = cat.register_upload("test_sinter.csv", csv_bytes, title="Test Sinter Batch")
    assert entry.manifest.resource_id

    pred = Predictor()
    out_df, analyses = predict_catalogue_dataset(entry.manifest.resource_id, catalogue=cat, predictor=pred)

    assert len(out_df) == 1
    assert "Pred_Ts" in out_df.columns
    assert len(analyses) == 1
    assert analyses[0].input_refs[0].ref_id == entry.manifest.source.source_id
