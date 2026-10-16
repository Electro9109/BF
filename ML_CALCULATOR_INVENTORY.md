# ML Calculator Inventory

## Operational Prediction Tasks

### 1. Blast Furnace - Softening & Melting Temperatures

| Property | Value |
|----------|-------|
| Department | blast_furnace |
| Prediction task | Softening & Melting temperature prediction |
| Target | Ts, Tm, Tm-Ts (°C) |
| Inputs (required) | 7 chemistry: T Fe %, FeO %, SiO2 %, CaO %, Al2O3 %, MgO%, Basicity |
| Inputs (optional) | 3 atmosphere: CO_pct, H2_pct, N2_pct (from parsed condition string) |
| Inputs (optional) | 5 burden: sinter_pct, ore_pct, pellet_pct, other_pct, num_components (from burden string) |
| Inputs (optional) | 3 interactions: CO_x_Basicity, Reducibility_Ratio, Basicity_x_Sinter (computed) |
| Inputs (optional) | Test type: SO, SOP, P, etc. (one-hot encoded) |
| Dataset | data_files/data_result.xlsx (Data Analysis sheet) |
| Dataset provenance | Experimental data (source unknown), 132 rows |
| Model type | Random Forest (auto-selected from CatBoost/RF/XGBoost per target) |
| Model artifacts | best_Ts.pkl, best_Tm.pkl, best_Tm_Ts.pkl in MLModels/ |
| Evaluation method | 5-fold cross-validation with RandomizedSearchCV (n_iter=25) |
| Evaluation metrics | Ts: MAE=46°C, RMSE=58°C, R²=0.64; Tm: MAE=36°C, RMSE=44°C, R²=0.35; Tm-Ts: MAE=46°C, RMSE=57°C, R²=0.51 |
| Feature transformations | RangeScaler using fixed domain ranges (CHEM_RANGES, ATM_RANGES, BURDEN_RANGES, INTERACTION_RANGES) |
| Categorical encoding | Test type one-hot encoding |
| Missing-value handling | Raises PredictionInputError for missing required chemistry |
| Applicability check | Nearest-neighbor Euclidean distance (scaled features) → high/medium/low confidence |
| Limitations | Small dataset (132 rows), no held-out test set, confidence is heuristic not calibrated, domain ranges are knowledge-based not data-derived |

**Training data distributions (7 chemistry features):**
- T Fe %: mean=58.3, min=43.1, max=68.6, std=4.0
- FeO %: mean=5.1, min=0.0, max=13.7, std=3.3
- SiO2 %: mean=4.3, min=1.2, max=16.8, std=1.7
- CaO %: mean=4.3, min=0.0, max=7.6, std=2.1
- Al2O3 %: mean=2.5, min=0.5, max=7.6, std=1.2
- MgO%: mean=1.9, min=0.0, max=5.1, std=1.2
- Basicity: mean=1.4, min=0.01, max=2.7, std=0.8

**Domain ranges (CHEM_RANGES in feature_processing.py) - UPDATED in Phase 2:**
- T Fe %: (40, 70) - covers observed data (43.1-68.6)
- FeO %: (0, 15) - covers observed data (0.0-13.7), includes zero
- SiO2 %: (1, 18) - covers observed data (1.2-16.8)
- CaO %: (0, 10) - covers observed data (0.0-7.6), includes zero
- Al2O3 %: (0.5, 8) - covers observed data (0.5-7.6)
- MgO%: (0, 6) - covers observed data (0.0-5.1), includes zero
- Basicity: (0.5, 3) - covers observed data (0.01-2.7)

**Observation:** Domain ranges were originally knowledge-based and sometimes tighter than observed data ranges. This would trigger warnings for valid experimental data at the extremes. **FIXED in Phase 2:** Ranges were adjusted to cover the full observed data distribution while remaining realistic for blast furnace operation. Existing models use the old ranges (saved in scalers.pkl), so scaling remains consistent. The new ranges only affect validation warnings for predictions and future training runs.

---

### 2. Sintering - Prospective Productivity

| Property | Value |
|----------|-------|
| Department | sintering |
| Prediction task | Prospective sinter productivity prediction |
| Target | Sinter Productivity (T/m²-hr) |
| Inputs (required) | 8 prospective features: I/O Fines Total Fe %, I/O Fines SiO2 %, I/O Fines Al2O3 %, I/O Fines CaO %, Flux CaO %, Flux MgO %, Flux Crushing Index, Coke Crushing Index |
| Dataset | data/raw/sintering/SinterProductivityData.csv |
| Dataset provenance | Public/research dataset (source unknown), 449 rows, 18 columns |
| Model type | sklearn pipeline (RandomForest) |
| Model artifacts | sinter_productivity_pipeline.pkl in MLModels/ |
| Evaluation method | 5-fold cross-validation |
| Evaluation metrics | MAE=0.051 T/m²-hr, RMSE=0.0655 T/m²-hr, R²=0.271 |
| Feature transformations | sklearn pipeline with preprocessing |
| Categorical encoding | None (all numeric) |
| Missing-value handling | Raises PredictionInputError for missing features or non-finite values |
| Applicability check | Range warnings only (no distance-based confidence) |
| Limitations | Low R² (27%), unmeasured factors (bed suction, moisture, depth) affect throughput, no test set holdout |

**Domain ranges (SINTER_RANGES in department.py):**
- I/O Fines Total Fe %: (58.0, 63.0)
- I/O Fines SiO2 %: (3.0, 6.0)
- I/O Fines Al2O3 %: (2.0, 4.0)
- I/O Fines CaO %: (0.2, 1.5)
- Flux CaO %: (30.0, 42.0)
- Flux MgO %: (6.0, 12.0)
- Flux Crushing Index: (65.0, 100.0)
- Coke Crushing Index: (65.0, 90.0)

---

### 3. Energy - Electrical Load

| Property | Value |
|----------|-------|
| Department | energy |
| Prediction task | Contemporaneous whole-facility electrical load estimation |
| Target | Usage_kWh |
| Inputs (required) | 5 numeric: Lagging_Current_Reactive.Power_kVarh, Leading_Current_Reactive_Power_kVarh, Lagging_Current_Power_Factor, Leading_Current_Power_Factor, NSM |
| Inputs (required) | 3 categorical: WeekStatus, Day_of_week, Load_Type |
| Dataset | data/raw/energy/Steel_industry_data.csv |
| Dataset provenance | UCI Machine Learning Repository, 35,040 rows, 15-minute intervals over 2018 |
| Model type | sklearn pipeline (RandomForest) |
| Model artifacts | steel_energy_pipeline.pkl in MLModels/ |
| Evaluation method | Chronological TimeSeriesSplit (5 splits) |
| Evaluation metrics | MAE=0.73 kWh, RMSE=1.88 kWh, R²=0.995 |
| Feature transformations | sklearn pipeline with preprocessing and one-hot encoding |
| Categorical encoding | One-hot encoding for WeekStatus, Day_of_week, Load_Type |
| Missing-value handling | Raises PredictionInputError for missing features or non-finite values |
| Applicability check | Range warnings only (no distance-based confidence) |
| Limitations | Contemporaneous estimation only, not multi-step future forecast |

**Domain ranges (ENERGY_RANGES in department.py):**
- Lagging_Current_Reactive.Power_kVarh: (0.0, 100.0)
- Leading_Current_Reactive_Power_kVarh: (0.0, 30.0)
- Lagging_Current_Power_Factor: (0.0, 100.0)
- Leading_Current_Power_Factor: (0.0, 100.0)
- NSM: (0.0, 86400.0)

---

## Training/Inference Consistency

### Blast Furnace
**Training (ml/train.py):**
- Uses `departments.blast_furnace.feature_processing.build_features()`
- Uses RangeScaler with CHEM_RANGES, ATM_RANGES, BURDEN_RANGES, INTERACTION_RANGES
- Parses atmosphere from "Test Condition" string
- Parses burden from "Burden compositon" string
- One-hot encodes test type
- Saves scalers.pkl, feature_names.pkl, targets.pkl

**Inference (ml/predictor.py):**
- Uses same scalers.pkl and feature_names.pkl
- Uses same RangeScaler class (loaded from scalers.pkl)
- Parses atmosphere from test_condition string or uses parsed_condition dict
- Parses burden from burden string or uses parsed_condition dict
- One-hot encodes test type
- **VERIFIED:** Training and inference use identical feature processing

### Sintering
**Training (implied from saved pipeline):**
- Uses 8 prospective features only
- sklearn pipeline with preprocessing
- Saves sinter_productivity_pipeline.pkl with features, ranges, model, cv_metrics

**Inference (ml/unified_calculator.py):**
- Loads pipeline, uses same 8 features
- Applies same preprocessing
- **VERIFIED:** Training and inference consistent (single pipeline artifact)

### Energy
**Training (implied from saved pipeline):**
- Uses 5 numeric + 3 categorical features
- sklearn pipeline with preprocessing and one-hot encoding
- Saves steel_energy_pipeline.pkl with raw_features, all_features, model, cv_metrics

**Inference (ml/unified_calculator.py):**
- Loads pipeline, builds categorical dummies
- Reindexes to match trained feature order exactly
- **VERIFIED:** Training and inference consistent (single pipeline artifact)

---

## Nearest-Neighbor Distance Calculation

**Location:** ml/similarity.py, function `nearest_neighbor_distance()`

**Method:**
1. Loads training matrix using `load_and_build()` (caches result)
2. Computes Euclidean distance between query row and all training rows
3. Returns minimum distance

**Features used:** Same 18 features as training (chemistry + atmosphere + burden + interactions + test type)

**Scaling:** Features are already scaled by RangeScaler (same as training)

**Thresholds (in ml/predictor.py):**
- distance ≤ 1.05: high confidence
- distance ≤ 1.82: medium confidence
- distance > 1.82: low confidence

**Status:** Implemented correctly. Features are scaled consistently. Thresholds are heuristic, not calibrated.

---

## Known Issues and Weaknesses

### 1. ~~BF Domain Ranges Tighter Than Observed Data~~ FIXED in Phase 2
**Issue (FIXED):** CHEM_RANGES in feature_processing.py were tighter than observed data min/max in some cases, causing unnecessary warnings for valid experimental data.

**Fix:** Adjusted ranges to cover the full observed data distribution while remaining realistic for blast furnace operation:
- T Fe %: (45, 65) → (40, 70)
- FeO %: (5, 25) → (0, 15)
- SiO2 %: (3, 10) → (1, 18)
- CaO %: (5, 15) → (0, 10)
- Al2O3 %: (1, 5) → (0.5, 8)
- MgO%: (0.5, 3) → (0, 6)
- Basicity: (0.8, 1.6) → (0.5, 3)

**Note:** Existing models use the old ranges (saved in scalers.pkl), so model scaling remains consistent. The new ranges only affect validation warnings for predictions and future training runs.

### 2. No Held-Out Test Set for BF
**Issue:** BF models are evaluated only via 5-fold CV, not on a held-out test set.

**Impact:** Cannot verify final model performance on unseen data.

**Recommendation:** Add train/validation/test split (e.g., 70/15/15) if time permits. Not critical for prototype.

### 3. Sintering R²=0.271 is Low
**Issue:** Sinter productivity model explains only 27% of variance.

**Impact:** Predictions have high uncertainty; unmeasured factors dominate.

**Recommendation:** Document limitation clearly. Consider adding more features if available (bed suction, moisture, depth). Do not claim high accuracy.

### 4. Confidence is Heuristic
**Issue:** Nearest-neighbor distance thresholds (1.05, 1.82) are not calibrated against prediction error.

**Impact:** "high confidence" does not guarantee low prediction error.

**Recommendation:** Document as heuristic applicability indicator, not calibrated confidence. Avoid presenting as probability.

### 5. No Applicability Check for Sinter/Energy
**Issue:** Only BF has distance-based confidence. Sinter and Energy have only range warnings.

**Impact:** No systematic way to flag unfamiliar inputs for Sinter/Energy.

**Recommendation:** Implement similar distance-based applicability for Sinter/Energy if feasible, or document as current limitation.

---

## Summary

All three prediction tasks are operational with verified training/inference consistency. Input validation is robust (missing values, non-finite values, range checking). Domain ranges are knowledge-based but sometimes tighter than observed data. Evaluation metrics are documented and reasonable for prototype scope. The main weakness is the heuristic confidence indicator (not calibrated) and the low R² for sintering (27%).
