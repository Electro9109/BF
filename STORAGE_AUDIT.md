# PARSE Storage Audit and Classification

## Existing Reference Knowledge (docs/)

| File | Topic | Role | Size | Intended Use | Status |
|------|-------|------|------|--------------|--------|
| basicity.txt | Basicity (CaO/SiO2) | reference | 5KB | BF slag chemistry fundamentals | Indexed |
| cohesive_zone.txt | Cohesive zone behavior | reference | 5KB | BF furnace permeability | Indexed |
| coke_quality.txt | CRI, CSR, solution loss | reference | 2.6KB | Coke quality fundamentals | Indexed |
| FeO.txt | Ferrous oxide in slag | reference | 5KB | BF oxidation state | Indexed |
| gas_atmosphere.txt | CO, H2, N2 atmospheres | reference | 6KB | BF reduction chemistry | Indexed |
| hydrogen_reduction.txt | H2 reduction mechanisms | reference | 5KB | Alternative reducing agents | Indexed |
| ore.txt | Iron ore types | reference | 5KB | Burden materials | Indexed |
| pellet.txt | Pellet characteristics | reference | 5KB | Burden materials | Indexed |
| permeability.txt | Gas permeability | reference | 5KB | BF gas flow | Indexed |
| pressure_drop.txt | Pressure drop | reference | 5KB | BF gas dynamics | Indexed |
| reducibility.txt | Ore reducibility | reference | 5KB | Reduction kinetics | Indexed |
| sinter.txt | Sinter properties | reference | 5KB | Burden materials | Indexed |
| slag_formation.txt | Slag formation | reference | 6KB | BF slag chemistry | Indexed |
| softening_melting.txt | Ts, Tm, Tm-Ts | reference | 6KB | BF softening/melting | Indexed |
| Tm_minus_Ts.txt | Melting interval | reference | 5KB | BF melting behavior | Indexed |
| experiments/experiment_scenarios.txt | Qualitative scenarios | experiment_scenario | N/A | Scenario retrieval | Indexed |

**Total:** 16 files, 124 chunks
**Coverage:** Strong BF/sintering fundamentals, no coke ovens or SMS theory
**Quality:** All are concise technical summaries, suitable for retrieval

## Existing Structured Datasets

### Blast Furnace (departments/blast_furnace/)
| Dataset | Location | Rows | Columns | Target | Use | Status |
|---------|----------|------|---------|--------|-----|--------|
| Data Analysis sheet | data_files/data_result.xlsx | 132 | Chemistry + atmosphere + burden + Ts/Tm/Tm-Ts | Ts, Tm, Tm-Ts | BF ML training | Active |

**Status:** Working, trained models in MLModels/, used by unified calculator

### Sintering (data/raw/sintering/)
| Dataset | Location | Rows | Columns | Target | Use | Status |
|---------|----------|------|---------|--------|-----|--------|
| SinterProductivityData.csv | data/raw/sintering/SinterProductivityData.csv | 449 | 18 (prospective + retrospective + productivity) | Sinter Productivity, T/m2-hr | Sinter ML training | Active |

**Status:** Working, trained model in MLModels/, used by unified calculator
**Note:** Uses only 8 prospective features to avoid leakage

### Energy (data/raw/energy/)
| Dataset | Location | Rows | Columns | Target | Use | Status |
|---------|----------|------|---------|--------|-----|--------|
| Steel_industry_data.csv | data/raw/energy/Steel_industry_data.csv | 35,040 | 11 (time + electrical + calendar) | Usage_kWh | Energy ML training | Active |

**Status:** Working, trained model in MLModels/, used by unified calculator
**Note:** UCI dataset, 15-minute time series, chronological split used

## Registered Sources in Catalogue

| Resource ID | Title | Type | Class | Processing State | Chunks |
|-------------|-------|------|-------|------------------|--------|
| resource-01cc3f4d... | steelmaking_process_overview.txt | document | internal_technical_document | indexed | 19 |

**Status:** 1 registered document (steelmaking overview), 1 of 16 docs/ files registered
**Note:** Most usage is via direct docs/ loading, not catalogue registration

## Model Artifacts (MLModels/)

| Artifact | Target | Model Type | Dataset | Metrics | Status |
|----------|--------|------------|---------|---------|--------|
| best_Ts.pkl | Ts | Random Forest (auto-selected) | BF data (132 rows) | MAE=46°C, RMSE=58°C, R²=0.64 | Active |
| best_Tm.pkl | Tm | Random Forest (auto-selected) | BF data (132 rows) | MAE=36°C, RMSE=44°C, R²=0.35 | Active |
| best_Tm_Ts.pkl | Tm-Ts | Random Forest (auto-selected) | BF data (132 rows) | MAE=46°C, RMSE=57°C, R²=0.51 | Active |
| sinter_productivity_pipeline.pkl | Sinter Productivity | sklearn pipeline | Sinter data (449 rows) | MAE=0.051, RMSE=0.0655, R²=0.271 | Active |
| steel_energy_pipeline.pkl | Usage_kWh | sklearn pipeline | Energy data (35,040 rows) | MAE=0.73 kWh, RMSE=1.88 kWh, R²=0.995 | Active |
| scalers.pkl | Feature scalers | RangeScaler | BF data | N/A | Active |
| feature_names.pkl | Feature names | N/A | BF data | N/A | Active |
| benchmark_results.csv | BF CV results | N/A | BF data | N/A | Active |

**Total:** 18 artifacts (3 BF models, 2 multi-department pipelines, 3 metadata files, 9 alternate models)
**Quality:** All have documented evaluation metrics

## Missing Departments

### Coke Ovens
**Status:** No implementation
**Needed for:**
- Coke quality prediction (CRI, CSR)
- Coke oven temperature optimization
- By-product recovery modeling

**Potential datasets (researched):**
1. UCI datasets: No direct coke oven datasets found
2. Research publications: Most are behind paywalls or lack open data
3. Industry datasets: Typically proprietary, not publicly available

**Recommendation:** Defer. Requires domain-specific proprietary data or partnership with coke plant. Existing coke_quality.txt provides reference knowledge but no structured dataset for ML.

### Steel Melting Shop (SMS)
**Status:** No implementation
**Needed for:**
- Steelmaking endpoint prediction
- Ladle furnace optimization
- Alloy addition calculations

**Potential datasets (researched):**
1. Kaggle: Limited steelmaking datasets, mostly generic steel industry data
2. UCI: Steel industry energy dataset already used for Energy department
3. Research: Process-specific SMS data is typically proprietary

**Recommendation:** Defer. Similar to coke ovens, requires plant-specific data. Energy department already covers electrical load.

## Storage Gaps by Priority

### High Priority (Working, can be improved)
1. **Catalogue registration:** Only 1 of 16 docs/ files registered
   - Action: Register remaining reference documents to formalize storage
   - Benefit: Consistent provenance tracking, easier management
   - Effort: Low (automated registration script)

2. **Coke Ovens reference knowledge:** Only coke_quality.txt exists
   - Action: Add 2-3 concise coke oven fundamentals documents
   - Topics: Coke oven temperature profiles, by-product recovery, oven operation
   - Benefit: Better RAG coverage for coke-related queries
   - Effort: Medium (requires research + writing)

### Medium Priority (Future expansion)
3. **SMS reference knowledge:** No SMS-specific documents
   - Action: Add 2-3 SMS fundamentals documents
   - Topics: Basic oxygen furnace, electric arc furnace, ladle refining
   - Benefit: Broader metallurgical coverage
   - Effort: Medium (requires research + writing)

4. **Sintering experiment scenarios:** No scenario file in docs/experiments/
   - Action: Create sintering-specific experiment scenarios
   - Benefit: Scenario-aware retrieval for sintering queries
   - Effort: Low (follow experiment_scenarios.txt pattern)

### Low Priority (Nice to have)
5. **Additional BF datasets:** Only one small dataset (132 rows)
   - Action: Find larger BF datasets (hundreds/thousands of rows)
   - Challenge: BF operational data is typically proprietary
   - Benefit: Improved model generalization
   - Effort: High (data discovery + integration)

6. **Time-series forecasting:** Current models are contemporaneous, not predictive
   - Action: Add time-series forecasting for energy load, BF parameters
   - Benefit: Operational planning support
   - Effort: High (new model types, validation methodology)

## Storage Refresh Verification

### Document Indexing
**Current behavior:**
- docs/ scanned recursively (fixed with rglob)
- storage/normalized/documents/ scanned recursively
- Fingerprint now includes nested files
- Index rebuilds on fingerprint change

**Test plan:**
1. Add new file to docs/experiments/
2. Verify fingerprint changes
3. Click "Reload Documents"
4. Verify chunk count increases
5. Verify new document retrievable

### Source Registration
**Current behavior:**
- Original bytes preserved
- Normalized artifact created
- Manifest recorded in SQLite
- Text documents indexed after registration
- Tabular documents catalogued only (not indexed)

**Test plan:**
1. Upload new text document
2. Register it
3. Verify in catalogue
4. Verify in chunks
5. Verify retrievable

## Storage Classification Summary

| Category | Count | Purpose | Integration Status |
|----------|-------|---------|-------------------|
| Reference documents (docs/) | 16 | RAG retrieval | Indexed via direct load |
| Registered documents | 1 | RAG retrieval + provenance | Indexed via catalogue |
| Structured datasets | 3 | ML training + analysis | Active in unified calculator |
| Model artifacts | 18 | ML inference | Active, documented metrics |
| Experiment scenarios | 1 | Scenario-aware retrieval | Indexed with role=experiment_scenario |

**Overall:** Storage is functional but under-utilized. Catalogue registration is sparse. Missing departments lack both reference knowledge and structured data.
