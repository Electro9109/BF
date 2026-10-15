# PARSE Dataset Schemas: Sintering Productivity & Steel Energy

## 1. Sintering Machine Productivity Dataset

- **Original Location**: `SinterProductivityData.csv`
- **Permanent Relocation**: [`data/raw/sintering/SinterProductivityData.csv`](file:///c:/Users/ASUS/Coding/Fwd_%20Tm-Ts/data/raw/sintering/SinterProductivityData.csv)
- **Department**: Sinter Plant (`sintering`)
- **Domain Scope**: Iron ore sintering machine operational and raw mix quality measurements.
- **Dimensions**: 449 rows × 18 columns.
- **Completeness**: 0 missing values across all columns; 0 duplicate rows.
- **Temporal Nature**: Sequential production sample records (`Sample No` 1 to 449).

### Columns & Engineering Units
| Column Name | Data Type | Units / Meaning | Min | Max | Mean | Role |
|---|---|---|---|---|---|---|
| `Sample No` | int64 | Sample identifier | 1 | 449 | 225.0 | Identifier |
| `I/O Fines Total Fe %` | float64 | Iron Ore Fines Total Iron % | 59.62 | 61.55 | 60.59 | Input Feature (Raw material chemistry) |
| `I/O Fines SiO2 %` | float64 | Iron Ore Fines Silica % | 3.59 | 5.68 | 4.88 | Input Feature |
| `I/O Fines Al2O3 %` | float64 | Iron Ore Fines Alumina % | 2.17 | 3.63 | 2.87 | Input Feature |
| `I/O Fines CaO %` | float64 | Iron Ore Fines Lime % | 0.39 | 1.10 | 0.63 | Input Feature |
| `Flux CaO %` | float64 | Limestone/flux Lime % | 34.66 | 37.80 | 36.31 | Input Feature (Flux quality) |
| `Flux MgO %` | float64 | Dolomite/flux Magnesia % | 8.33 | 10.11 | 9.38 | Input Feature |
| `Flux Crushing Index` | float64 | Flux grind size / crushing index % | 70.00 | 94.73 | 82.52 | Input Feature (Granulometry) |
| `Coke Crushing Index` | float64 | Solid fuel (coke breeze) sizing index % | 70.00 | 85.03 | 77.06 | Input Feature (Fuel sizing) |
| `Sinter Total Fe %` | float64 | Finished product total iron % | 53.52 | 56.98 | 55.48 | Input Feature (Product chemistry) |
| `Sinter FeO %` | float64 | Finished product ferrous oxide % | 8.36 | 8.90 | 8.64 | Input Feature (State of oxidation) |
| `Sinter SiO2 %` | float64 | Finished product silica % | 4.84 | 6.90 | 5.92 | Input Feature |
| `Sinter Al2O3 %` | float64 | Finished product alumina % | 2.16 | 3.06 | 2.58 | Input Feature |
| `Sinter CaO %` | float64 | Finished product lime % | 8.04 | 11.55 | 9.94 | Input Feature |
| `Sinter MgO %` | float64 | Finished product magnesia % | 2.31 | 3.01 | 2.65 | Input Feature |
| `Sinter +40mm Size, %` | float64 | Yield percentage of oversize lump (+40mm) | 0.00 | 30.00 | 12.38 | Input Feature (Physical quality) |
| `Drum Tumbling Index (DTI), %` | float64 | Cold strength / ISO tumble test index % | 62.70 | 78.70 | 69.45 | Input Feature (Strength indicator) |
| `Sinter Productivity, T/m2-hr` | float64 | Sinter strand specific productivity (Tonnes / m² area / hour) | 0.87 | 1.37 | 1.15 | **Primary Prediction Target** |

### Analytical & Predictive Use
- **Primary Task**: Supervised continuous regression to predict `Sinter Productivity, T/m2-hr` from raw mix properties, granulometry (crushing indices), and finished sinter chemistry.
- **Distinction from Blast Furnace Model**: The existing BF model predicts laboratory softening and melting temperatures ($Ts$, $Tm$). This dataset predicts strand yield/production rate per unit strand area.

---

## 2. Steel Industry Energy Consumption Dataset

- **Original Location**: `Steel_industry_data.csv`
- **Permanent Relocation**: [`data/raw/energy/Steel_industry_data.csv`](file:///c:/Users/ASUS/Coding/Fwd_%20Tm-Ts/data/raw/energy/Steel_industry_data.csv)
- **Department**: Energy & Utilities (`energy`)
- **Origin**: UCI Machine Learning Repository (Steel Industry Energy Consumption Dataset).
- **Dimensions**: 35,040 rows × 11 columns (full 365 days of 15-minute sensor intervals in 2018: $365 \times 24 \times 4 = 35,040$).
- **Completeness**: 0 missing values; 0 duplicate timestamps.
- **Temporal Nature**: Strict 15-minute time-series (`date` formatted as `DD/MM/YYYY HH:MM`).

### Columns & Engineering Units
| Column Name | Data Type | Units / Meaning | Role |
|---|---|---|---|
| `date` | object (DateTime) | Timestamp (15-min interval, 2018-01-01 00:00 to 2018-12-31 23:45) | Time Index |
| `Usage_kWh` | float64 | Active electrical power consumption in kWh | **Primary Target** |
| `Lagging_Current_Reactive.Power_kVarh` | float64 | Reactive electrical energy (lagging) in kVarh | Input Feature / Secondary target |
| `Leading_Current_Reactive_Power_kVarh` | float64 | Reactive electrical energy (leading) in kVarh | Input Feature |
| `CO2(tCO2)` | float64 | Carbon dioxide emission equivalent in tonnes | Output indicator (derived from kWh) |
| `Lagging_Current_Power_Factor` | float64 | Power factor (lagging current, % or ratio 0–100) | Electrical efficiency feature |
| `Leading_Current_Power_Factor` | float64 | Power factor (leading current, % or ratio 0–100) | Electrical efficiency feature |
| `NSM` | int64 | Number of Seconds from Midnight (0 to 85,500s) | Cyclical time feature |
| `WeekStatus` | object | Categorical: `Weekday` vs `Weekend` | Calendar feature |
| `Day_of_week` | object | Categorical: Monday through Sunday | Calendar feature |
| `Load_Type` | object | Operational regime: `Light_Load`, `Medium_Load`, `Maximum_Load` | Operational state feature |

### Analytical & Predictive Use
- **Analytical Task**: Descriptive statistics, diurnal demand profiles, power factor efficiency analysis, load-type peak demand distribution, and CO2 intensity tracking.
- **Predictive Task**: Regression for 15-minute electrical demand (`Usage_kWh`) respecting strict temporal splitting (no random shuffle leakage across time boundaries).
- **Distinction from Metallurgical Reaction Models**: This dataset covers utility-level electrical load dynamics, not chemical kinetics or thermodynamic melt intervals.
