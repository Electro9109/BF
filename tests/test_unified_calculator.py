"""Tests for the Unified Multi-Department Calculator and Translator integration."""

import pytest
import numpy as np
from pathlib import Path

from config.paths import ML_MODEL_DIR
from ml.unified_calculator import UnifiedCalculator, DepartmentPredictionResult
from ml.predictor import PredictionInputError
from parse.translator_pipeline import ResultsTranslator


def test_unified_calculator_tasks():
    calc = UnifiedCalculator(ML_MODEL_DIR)
    tasks = calc.get_available_tasks()
    assert "Blast Furnace (Softening & Melting Ts / Tm)" in tasks
    assert "Sinter Plant (Prospective Productivity)" in tasks
    assert "Steel Energy (Active Load Usage_kWh)" in tasks


def test_sinter_prospective_prediction():
    calc = UnifiedCalculator(ML_MODEL_DIR)
    inputs = {
        "I/O Fines Total Fe %": 60.10,
        "I/O Fines SiO2 %": 4.50,
        "I/O Fines Al2O3 %": 2.80,
        "I/O Fines CaO %": 0.60,
        "Flux CaO %": 36.00,
        "Flux MgO %": 8.50,
        "Flux Crushing Index": 82.00,
        "Coke Crushing Index": 78.00,
    }
    result = calc.predict_sinter_productivity(inputs)
    assert isinstance(result, DepartmentPredictionResult)
    assert result.department == "sintering"
    assert result.target == "Sinter Productivity, T/m2-hr"
    assert result.units == "T/m2-hr"
    assert 0.5 < result.predicted_value < 2.5
    assert "r2" in result.evaluation_metrics
    assert len(result.limitations) > 0

    # Ensure to_structured_result builds a valid contract
    s_res = result.to_structured_result()
    assert s_res.module_type == "regression_prediction"
    assert s_res.department == "sintering"
    assert result.target in s_res.predictions


def test_sinter_prediction_validation():
    calc = UnifiedCalculator(ML_MODEL_DIR)
    # Missing feature
    with pytest.raises(PredictionInputError, match="Missing required prospective features"):
        calc.predict_sinter_productivity({"I/O Fines Total Fe %": 60.1})

    # Non-finite value
    with pytest.raises(PredictionInputError, match="invalid non-finite"):
        calc.predict_sinter_productivity({
            "I/O Fines Total Fe %": np.nan,
            "I/O Fines SiO2 %": 4.50,
            "I/O Fines Al2O3 %": 2.80,
            "I/O Fines CaO %": 0.60,
            "Flux CaO %": 36.00,
            "Flux MgO %": 8.50,
            "Flux Crushing Index": 82.00,
            "Coke Crushing Index": 78.00,
        })


def test_energy_prediction_and_structure():
    calc = UnifiedCalculator(ML_MODEL_DIR)
    num_inputs = {
        "Lagging_Current_Reactive.Power_kVarh": 12.50,
        "Leading_Current_Reactive_Power_kVarh": 0.00,
        "Lagging_Current_Power_Factor": 88.50,
        "Leading_Current_Power_Factor": 100.00,
        "NSM": 32400.0,
    }
    cal_inputs = {
        "WeekStatus": "Weekday",
        "Day_of_week": "Monday",
        "Load_Type": "Medium_Load",
    }
    result = calc.predict_energy_consumption(num_inputs, cal_inputs)
    assert isinstance(result, DepartmentPredictionResult)
    assert result.department == "energy"
    assert result.target == "Usage_kWh"
    assert result.units == "kWh"
    assert result.predicted_value > 0
    assert "r2" in result.evaluation_metrics

    # Test conversion to structured result
    s_res = result.to_structured_result()
    assert s_res.department == "energy"
    assert "Usage_kWh" in s_res.predictions


def test_calculator_prediction_translator_roundtrip():
    calc = UnifiedCalculator(ML_MODEL_DIR)
    inputs = {
        "I/O Fines Total Fe %": 60.10,
        "I/O Fines SiO2 %": 4.50,
        "I/O Fines Al2O3 %": 2.80,
        "I/O Fines CaO %": 0.60,
        "Flux CaO %": 36.00,
        "Flux MgO %": 8.50,
        "Flux Crushing Index": 82.00,
        "Coke Crushing Index": 78.00,
    }
    dept_res = calc.predict_sinter_productivity(inputs)
    structured_res = dept_res.to_structured_result()

    translator = ResultsTranslator()
    out = translator.translate(structured_res)
    assert out.text is not None
    assert str(dept_res.predicted_value) in out.text or f"{dept_res.predicted_value:.2f}" in out.text
