# Dataset Candidates for Integration

## Blast Furnace Candidates

### 1. Process data collected by blast-furnace sensors
- **Source:** https://doi.org/10.17632/6d7jbc7tb5 (Mendeley Data)
- **Publisher:** Mendeley Data
- **License:** Unknown (research dataset)
- **File format:** Likely CSV/Excel
- **Size:** Unknown
- **Features:** Total and specific gas pressure differences, peripheral gas temperature, top gas temperature, CO2 content of top gas, H2 content of top gas, top gas pressure, ore-to-coke ratio, blast flowrate, blast oxygen flowrate, hot blast pressure, hot blast temperature, cold blast pressure, cold blast temperature, Si content in hot metal
- **Target:** Silicon content in hot metal ([Si])
- **Intended use:** Regression, forecasting, classification, time-series analysis
- **Data structure:** Time series from 26 sensors
- **Quality:** Raw and validated versions available
- **Recommendation:** **Defer** - Dataset is for Si content prediction, which is different from current BF task (Ts/Tm/Tm-Ts). Would require new model training and different target. Existing BF dataset is adequate for prototype scope.
- **Integration effort:** High (new model, new target, different schema)

### 2. ylkyc/ironmaking-zju
- **Source:** https://github.com/ylkyc/ironmaking-zju
- **Publisher:** GitHub
- **License:** Research only, non-commercial
- **File format:** Likely CSV
- **Size:** Unknown
- **Features:** Silicon content, molten iron quality parameters
- **Target:** Molten iron quality (MIQ)
- **Intended use:** Soft sensor, deep learning
- **Recommendation:** **Reject** - License restricts to research-only, non-commercial use. Not suitable for prototype deployment.
- **Integration effort:** High (license restriction)

### 3. yingtaoluo/Ironmaking-Sequence-Learning
- **Source:** https://github.com/yingtaoluo/Ironmaking-Sequence-Learning
- **Publisher:** GitHub
- **License:** Unknown
- **File format:** Likely CSV
- **Features:** Silicon content [Si], sulfur content [S], coal injection PML, blast volume FL
- **Target:** Silicon content [Si] time series prediction
- **Intended use:** Temperature prediction, time series forecasting
- **Recommendation:** **Defer** - Focuses on Si content prediction, different from current BF task. Would require new model.
- **Integration effort:** High (new model, new target)

## Sintering Candidates

### 1. Sinter machine productivity (Mendeley)
- **Source:** https://doi.org/10.17632/72hgnwmsvs
- **Publisher:** Mendeley Data
- **License:** Unknown
- **File format:** Likely CSV
- **Features:** 16 input parameters (same as existing dataset)
- **Target:** Sinter machine productivity
- **Intended use:** Prescriptive analytics
- **Recommendation:** **Reject** - Likely the same dataset as existing SinterProductivityData.csv. No added value.
- **Integration effort:** Low (if same), but unnecessary

### 2. fight46/Sintering-Process-data
- **Source:** https://github.com/fight46/Sintering-Process-data
- **Publisher:** GitHub
- **License:** Unknown
- **File format:** Likely CSV
- **Features:** Sintering process data from model 400 sintering machines
- **Intended use:** Pattern moving modeling verification
- **Recommendation:** **Defer** - Could provide additional sintering data, but current dataset is adequate for prototype. Would need to verify schema compatibility.
- **Integration effort:** Medium (schema verification, possible integration)

## Coke Ovens Candidates

### 1. Historical coke and breeze data (GOV.UK)
- **Source:** https://www.gov.uk/government/statistical-data-sets/historical-coke-and-breeze-data
- **Publisher:** UK Government
- **License:** Open government license
- **File format:** Excel/CSV
- **Features:** Coal carbonized, coke and breeze produced (aggregated annual statistics)
- **Target:** None (time series statistics)
- **Intended use:** Historical analysis, not process control
- **Recommendation:** **Reject** - Aggregated annual statistics, not process-level data for ML prediction. Cannot support CRI/CSR prediction.
- **Integration effort:** Low (data is available), but not useful for ML

### 2. CSIRO Coke Reactivity Test
- **Source:** https://data.csiro.au/ (various collections)
- **Publisher:** CSIRO
- **License:** Unknown
- **Features:** Coke reactivity test results, process data
- **Target:** CRI, CSR
- **Intended use:** Coke quality assessment
- **Recommendation:** **Defer** - Requires account/request for data access. May have usage restrictions. Not trivial to integrate within time constraints.
- **Integration effort:** High (access request, license review)

### 3. Research papers on coke quality
- **Source:** Various DOI links
- **Publisher:** Academic journals
- **License:** Paywalled
- **Recommendation:** **Reject** - Papers behind paywalls, no accessible data.

## Steel Melting Shop Candidates

### 1. Industrial data from the ladle-furnace unit (Kaggle)
- **Source:** https://www.kaggle.com/datasets/yuriykatser/industrial-data-from-the-ladlefurnace-unit
- **Publisher:** Kaggle
- **License:** Unknown (likely CC0 or similar)
- **File format:** CSV (multiple files: data_gas.csv, data_bulk.csv, data_bulk_time.csv, data_wire.csv, data_wire_time.csv, data_arc.csv, data_temp.csv)
- **Size:** Unknown
- **Features:** Gas purging, bulk elements, wire supply, electricity, temperature measurements
- **Target:** Temperature, chemical composition
- **Intended use:** Process optimization, temperature prediction
- **Quality:** Industrial data from real ladle furnace unit
- **Recommendation:** **Use** - This is the most suitable candidate for SMS integration. Provides process-level data for ladle furnace operations. Multiple related files allow for comprehensive analysis.
- **Integration effort:** Medium (multiple files to load, schema mapping, department implementation)
- **Benefit:** Enables SMS department with real industrial data

### 2. ac1esan/EAF-on-Metallurgy
- **Source:** https://github.com/ac1esan/EAF-on-Metallurgy
- **Publisher:** GitHub
- **License:** Unknown
- **File format:** CSV (eaf_metallurgy_dataset.csv, eaf_premium_75k.zip)
- **Features:** 72 columns including technical parameters, economic indicators, weather, logistics
- **Target:** Anomaly detection, predictive maintenance
- **Intended use:** EAF digital twin, causal swarm analysis
- **Recommendation:** **Defer** - Data appears to be simulated/synthetic (author describes it as "pain" and "virtual workshop"). For prototype, prefer real industrial data.
- **Integration effort:** Medium (large dataset, complex schema)
- **Note:** Simulated data may not represent real EAF behavior accurately

## Summary and Selection

### Selected for Integration
**Industrial data from the ladle-furnace unit (Kaggle)**
- **Reason:** Real industrial data, suitable for SMS department, reasonable integration effort
- **URL:** https://www.kaggle.com/datasets/yuriykatser/industrial-data-from-the-ladlefurnace-unit
- **Task:** SMS ladle furnace temperature and chemical composition prediction
- **Department:** Steel Melting Shop (new)
- **Integration effort:** Medium (requires new department implementation)

### Deferred
- BF sensor datasets (different target, would require new model)
- Additional sintering datasets (existing dataset adequate)
- CSIRO coke data (access restrictions)
- EAF simulated data (synthetic, prefer real data)

### Rejected
- License-restricted datasets (research-only, paywalled)
- Aggregated statistics (not process-level)
- Duplicate datasets (same as existing)

### Remaining Gaps
- **Coke Ovens:** No suitable public dataset found. Requires proprietary plant data.
- **BF Temperature:** Existing dataset adequate. Alternative datasets have different targets (Si content).
- **Sintering:** Existing dataset adequate. Alternative datasets similar or require verification.
