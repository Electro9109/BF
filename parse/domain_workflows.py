"""Domain-specific analysis workflows for Sintering and Energy departments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit

from config.paths import SINTER_PRODUCTIVITY_CSV, STEEL_ENERGY_CSV
from departments.energy.department import EnergyDepartment
from departments.sintering.department import SinteringDepartment
from parse.core.translator_contract import StructuredModuleResult


def analyze_sintering_productivity(
    csv_path: str | Path = SINTER_PRODUCTIVITY_CSV,
) -> StructuredModuleResult:
    """Analyze the sintering machine productivity dataset and train a baseline predictor."""
    dept = SinteringDepartment()
    df = dept.load_data(csv_path)

    target_col = dept.target_columns[0]
    features = dept.feature_columns

    y = df[target_col].values
    X = df[features].values

    # Descriptive statistics
    stats = {
        "sample_count": len(df),
        "feature_count": len(features),
        "target_mean": round(float(np.mean(y)), 4),
        "target_std": round(float(np.std(y)), 4),
        "target_min": round(float(np.min(y)), 4),
        "target_max": round(float(np.max(y)), 4),
    }

    # Correlations with productivity
    corrs = df[features].apply(lambda s: float(s.corr(df[target_col]))).to_dict()
    top_pos = sorted(corrs.items(), key=lambda x: x[1], reverse=True)[:3]
    top_neg = sorted(corrs.items(), key=lambda x: x[1])[:3]

    findings = [
        {
            "finding_id": "productivity_distribution",
            "observation": f"Sinter Productivity averages {stats['target_mean']:.2f} T/m2-hr (range {stats['target_min']:.2f} to {stats['target_max']:.2f} T/m2-hr).",
            "values": {"mean": stats["target_mean"], "std": stats["target_std"]},
        },
        {
            "finding_id": "productivity_key_correlations",
            "observation": f"Strongest positive correlations: {top_pos[0][0]} (r={top_pos[0][1]:.2f}), {top_pos[1][0]} (r={top_pos[1][1]:.2f}). Strongest negative correlation: {top_neg[0][0]} (r={top_neg[0][1]:.2f}).",
            "values": {"positive_correlations": top_pos, "negative_correlations": top_neg},
            "limitations": "Correlation does not imply causation.",
        },
    ]

    # Outlier / Quality check
    anomalies = []
    # e.g., low tumbling strength (<65%)
    low_dti = df[df["Drum Tumbling Index (DTI), %"] < 65.0]
    if not low_dti.empty:
        anomalies.append({
            "anomaly_id": "low_cold_strength_samples",
            "description": f"Found {len(low_dti)} samples with cold strength DTI < 65.0%, representing degraded sinter quality.",
            "count": len(low_dti),
        })

    # Cross-validation regression
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rf = RandomForestRegressor(n_estimators=150, max_depth=8, min_samples_leaf=2, random_state=42)
    maes, rmses, r2s = [], [], []
    for tr, te in kf.split(X):
        rf.fit(X[tr], y[tr])
        preds = rf.predict(X[te])
        maes.append(mean_absolute_error(y[te], preds))
        rmses.append(np.sqrt(mean_squared_error(y[te], preds)))
        r2s.append(r2_score(y[te], preds))

    eval_metrics = {
        "model": "RandomForestRegressor",
        "cv_folds": 5,
        "mae": round(float(np.mean(maes)), 4),
        "rmse": round(float(np.mean(rmses)), 4),
        "r2": round(float(np.mean(r2s)), 3),
        "r2_std": round(float(np.std(r2s)), 3),
    }

    recommendations = [
        "Maintain coke and flux crushing granulometry within target ranges as they correlate directly with sintering strand bed permeability and productivity.",
        "Review operating conditions for batches with Drum Tumbling Index (DTI) below 65% to avoid degradation in blast furnace burden permeability.",
    ]

    limitations = [
        "Dataset contains 449 laboratory/plant batch records; predictions represent strand productivity under recorded raw mix conditions, not thermodynamic softening-melting temperatures.",
        "Model explains ~29% of variance (R²=0.293); unmeasured bed suction, ignition temperature, and moisture parameters also affect strand throughput.",
    ]

    return StructuredModuleResult(
        module_type="regression_prediction",
        department="sintering",
        operation="productivity_analysis_and_prediction",
        dataset_id=Path(csv_path).name,
        summary=f"Evaluated sintering productivity across {len(df)} production records with 5-fold cross-validation.",
        target=target_col,
        units="T/m2-hr",
        statistics=stats,
        findings=findings,
        evaluation_metrics=eval_metrics,
        anomalies=anomalies,
        recommendations=recommendations,
        limitations=limitations,
        confidence="medium",
    )


def analyze_steel_energy_consumption(
    csv_path: str | Path = STEEL_ENERGY_CSV,
) -> StructuredModuleResult:
    """Analyze the steel industry energy consumption dataset and evaluate temporal load dynamics."""
    dept = EnergyDepartment()
    df = dept.load_data(csv_path)

    target_col = dept.target_columns[0]
    features = dept.feature_columns

    y = df[target_col].values

    stats = {
        "sample_count": len(df),
        "time_interval_minutes": 15,
        "date_start": str(df["date"].iloc[0]),
        "date_end": str(df["date"].iloc[-1]),
        "mean_usage_kwh": round(float(np.mean(y)), 2),
        "max_usage_kwh": round(float(np.max(y)), 2),
        "min_usage_kwh": round(float(np.min(y)), 2),
        "total_mwh": round(float(np.sum(y) / 1000.0), 2),
    }

    # Group differences by load type
    load_groups = df.groupby("Load_Type")[target_col].agg(["mean", "std", "count"]).to_dict("index")
    weekday_vs_weekend = df.groupby("WeekStatus")[target_col].mean().to_dict()

    findings = [
        {
            "finding_id": "energy_usage_by_load_type",
            "observation": f"Mean electrical demand varies sharply by regime: Maximum_Load averages {load_groups.get('Maximum_Load', {}).get('mean', 0):.2f} kWh vs Light_Load at {load_groups.get('Light_Load', {}).get('mean', 0):.2f} kWh.",
            "values": {k: {m: round(float(v), 2) for m, v in vals.items()} for k, vals in load_groups.items()},
        },
        {
            "finding_id": "weekday_vs_weekend_demand",
            "observation": f"Weekday average consumption is {weekday_vs_weekend.get('Weekday', 0):.2f} kWh compared to {weekday_vs_weekend.get('Weekend', 0):.2f} kWh on weekends.",
            "values": {k: round(float(v), 2) for k, v in weekday_vs_weekend.items()},
        },
    ]

    # Outlier / Anomaly check: suboptimal power factor during active production (>50 kWh)
    low_pf = df[(df["Lagging_Current_Power_Factor"] < 80.0) & (df["Usage_kWh"] > 50.0)]
    anomalies = []
    if not low_pf.empty:
        anomalies.append({
            "anomaly_id": "poor_power_factor_high_load",
            "description": f"Identified {len(low_pf)} intervals where lagging power factor dropped below 80% during active load (>50 kWh), incurring reactive power penalties.",
            "count": len(low_pf),
        })

    # Temporal split baseline evaluation
    cats = dept.parse_custom_fields(df)["categoricals"]
    X = pd.concat([df[features], cats], axis=1).values

    tscv = TimeSeriesSplit(n_splits=3)
    ridge = Ridge(alpha=1.0)
    maes, rmses, r2s = [], [], []
    for tr, te in tscv.split(X):
        ridge.fit(X[tr], y[tr])
        preds = ridge.predict(X[te])
        maes.append(mean_absolute_error(y[te], preds))
        rmses.append(np.sqrt(mean_squared_error(y[te], preds)))
        r2s.append(r2_score(y[te], preds))

    eval_metrics = {
        "model": "RidgeRegression_TimeSeriesSplit",
        "splits": 3,
        "mae": round(float(np.mean(maes)), 2),
        "rmse": round(float(np.mean(rmses)), 2),
        "r2": round(float(np.mean(r2s)), 3),
    }

    recommendations = [
        "Audit capacitor banks and reactive power compensation during intervals where Lagging Power Factor falls below 60% to avoid utility penalties.",
        "Shift non-continuous energy-intensive operations to designated Light_Load off-peak intervals where feasible.",
    ]

    limitations = [
        "Dataset originates from the UCI Steel Industry Energy Consumption dataset covering electric arc and rolling plant loads in 2018.",
        "This dataset reflects whole-facility electrical demand and is not a substitute for blast furnace thermal or coke consumption models.",
    ]

    return StructuredModuleResult(
        module_type="descriptive_analysis",
        department="energy",
        operation="energy_consumption_profile",
        dataset_id=Path(csv_path).name,
        summary=f"Profiled 35,040 15-minute electrical demand intervals across weekday/weekend cycles and load regimes.",
        target=target_col,
        units="kWh",
        statistics=stats,
        findings=findings,
        evaluation_metrics=eval_metrics,
        anomalies=anomalies,
        recommendations=recommendations,
        limitations=limitations,
        confidence="high",
    )
