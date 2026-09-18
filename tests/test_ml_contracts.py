import numpy as np
import pandas as pd
import pytest

from ml.condition_parser import parse_condition_nl
from departments.blast_furnace.feature_processing import (
    CHEM_COLS,
    TARGET_COLS,
    build_features,
    load_and_build,
    RangeScaler,
    CHEM_RANGES,
)
from ml.predictor import PredictionInputError, Predictor
from pipeline.prediction_pipeline import PredictionPipeline


def chemistry_values():
    return {
        "T Fe %": 57.62,
        "FeO %": 6.65,
        "SiO2 %": 4.23,
        "CaO %": 6.93,
        "Al2O3 %": 2.51,
        "MgO%": 1.89,
        "Basicity": 1.4,
    }


def test_incomplete_condition_without_llm_returns_warning():
    result = parse_condition_nl("CO=40% with an unspecified burden", None, None)

    assert result["warnings"]
    assert any("no local LLM" in warning for warning in result["warnings"])


def test_feature_builder_reuses_supplied_scalers():
    frame = pd.DataFrame([{
        **chemistry_values(),
        "Ts": 1300,
        "Tm": 1500,
        "Tm-Ts": 200,
    }])
    fitted = build_features(frame, use_atmosphere=False, use_burden=False, use_test_type=False)
    reused = build_features(
        frame,
        use_atmosphere=False,
        use_burden=False,
        use_test_type=False,
        fit_scalers=False,
        scalers=fitted["scalers"],
    )

    np.testing.assert_allclose(fitted["X"], reused["X"])


def test_predictor_rejects_missing_chemistry():
    predictor = Predictor.__new__(Predictor)
    predictor._scalers = {
        "chemistry": RangeScaler(CHEM_RANGES, CHEM_COLS),
    }

    with pytest.raises(PredictionInputError, match="Missing chemistry"):
        predictor._build_row({})


def test_prediction_pipeline_preserves_feature_and_similarity_results():
    pipeline = PredictionPipeline.__new__(PredictionPipeline)

    class FakePredictor:
        last_feature_vector = [0.1, 0.2]

        def predict(self, **kwargs):
            return {"Ts": 1300.0, "Tm": 1500.0, "Tm-Ts": 200.0}

    pipeline.predictor = FakePredictor()
    pipeline.attach_similarity_retriever(lambda chemistry, k: [{"row_index": 1, "similarity": 0.9}])

    result = pipeline.run(
        chemistry=chemistry_values(),
        test_condition="CO=40%, N2=60%",
        burden="S1-70%+O1-30%",
        test_type="SO",
    )

    assert result.feature_vector == [0.1, 0.2]
    assert result.similar_experiments[0]["row_index"] == 1


def test_load_and_build_uses_department_load_data_when_supplied():
    """Task 18: load_and_build(department=...) must call department.load_data(path)
    instead of load_raw(path), and must not otherwise change build_features's output.
    """
    frame = pd.DataFrame([{
        **chemistry_values(),
        "Ts": 1300,
        "Tm": 1500,
        "Tm-Ts": 200,
    }])

    class FakeDepartment:
        def __init__(self):
            self.calls = []

        def load_data(self, path):
            self.calls.append(path)
            return frame

    fake_department = FakeDepartment()
    result = load_and_build(
        path="unused-path-should-be-passed-through",
        use_atmosphere=False,
        use_burden=False,
        use_test_type=False,
        department=fake_department,
    )

    assert fake_department.calls == ["unused-path-should-be-passed-through"]
    direct = build_features(frame, use_atmosphere=False, use_burden=False, use_test_type=False)
    np.testing.assert_allclose(result["X"], direct["X"])


def test_load_and_build_without_department_does_not_require_one():
    # department defaults to None; this must not raise for the signature itself
    # (actual load_raw() execution needs the real data file, tested separately
    # / covered by the pre-existing environment-dependent test).
    import inspect
    signature = inspect.signature(load_and_build)
    assert signature.parameters["department"].default is None