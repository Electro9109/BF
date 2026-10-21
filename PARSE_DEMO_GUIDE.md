# PARSE Demonstration Guide

This guide provides a reproducible sequence for demonstrating the major PARSE workflows.

## Prerequisites

- Python 3.13.14 (or compatible version)
- All dependencies installed: `pip install -r requirements.txt`
- Local models present:
  - `LocalModels/` (Qwen-0.5B-Chat or compatible LLM)
  - `EmbedModels/` (all-MiniLM-L6-v2 or compatible embedding model)
  - `MLModels/` (trained model artifacts)
- Documentation files in `docs/` (metallurgical reference documents)

## Launch the Application

```bash
streamlit run app_web.py
```

The application will open at `http://localhost:8501` (default Streamlit port).

**Note:** First launch may take several minutes to load the local LLM (~1.2 GB) and build the document index.

---

## Workflow 1: Data Ingestion and Exploration

### Step 1: Upload a Dataset
1. Navigate to the **📥 Data Ingestion** tab
2. Click "Upload a file"
3. Select a CSV, Excel, JSON, or JSON Lines file (max 100 MiB)
4. Review the file information displayed

### Step 2: Run Exploratory Data Analysis
1. Click "Run Analysis" (automatic after upload)
2. Review the analysis results:
   - **Attributes tab:** Column names, data types, missingness, unique counts, and distribution summaries
   - **Relationships tab:** Correlations and associations between columns
   - **Quality tab:** Data quality findings (missing values, duplicates, outliers)
   - **Semantic Candidates tab:** Candidate attribute meanings (requires domain-specific context)
   - **Summary tab:** Overall dataset profile and key findings

### Step 3: Review Cleaning Proposals (if applicable)
1. If quality issues are detected, cleaning proposals will appear
2. Review each proposal with its impact estimate
3. Select proposals to apply
4. Click "Apply Selected Cleanings"
5. Review the before/after quality snapshots

### Step 4: Register the Dataset
1. Click "Register in Catalogue"
2. Confirm the registration
3. The dataset is now available for structured querying

**Verification:** The simplified view should show correct data types, missingness percentages, and summary statistics. Finding descriptions should not show "N/A".

---

## Workflow 2: Structured Data Querying

### Step 1: Select a Dataset
1. Navigate to the **📊 Data Query** tab
2. Select a dataset from the dropdown:
   - `blast_furnace` (data_files/data_result.xlsx)
   - `sinter_productivity` (data/raw/sintering/SinterProductivityData.csv)
   - `steel_energy` (data/raw/energy/Steel_industry_data.csv)

### Step 2: Choose an Operation
1. Select an operation:
   - **Get Columns:** View column names and data types
   - **Get Statistics:** View count, mean, median, min, max, std for a column
   - **Filter Rows:** Filter by column value (eq, ne, gt, lt, ge, le, in)
   - **Aggregate:** Aggregate by group (mean, sum, count)
   - **Get Top Rows:** Get top/bottom N rows by column

### Step 3: Execute the Query
1. Fill in the operation-specific parameters
2. Click "Execute Query"
3. Review the results with metadata (dataset_id, operation, row_count, filter/aggregate details)
4. Download results as CSV (for DataFrame results)

**Verification:** Results should match independent calculations. Units and sample sizes should be accurate. Correlations should not be presented as causation.

---

## Workflow 3: ML Prediction

### Step 1: Select a Department
1. Navigate to the **🧮 Predictor** tab
2. Select a task:
   - **Blast Furnace (Softening & Melting Ts / Tm)**
   - **Sinter Plant (Prospective Productivity)**
   - **Steel Energy (Active Load Usage_kWh)**

### Step 2: Enter Feature Values
1. **Blast Furnace:**
   - Enter chemistry values (T Fe %, FeO %, SiO2 %, CaO %, Al2O3 %, MgO%, Basicity)
   - Enter atmosphere (CO %, H2 %, N2 %) or use natural language description
   - Enter burden composition (sinter %, ore %, pellet %, other %)
   - Select test type (SO, SOP, or P)
2. **Sintering:**
   - Enter raw mix chemistry and crushing indices
3. **Steel Energy:**
   - Enter reactive power, power factors, time, and load type

### Step 3: Review Parsed Condition
1. Expand "Parsed Condition Preview" to verify extracted values
2. Check for warnings (out-of-range values, percentage sum violations)

### Step 4: Run Prediction
1. Click "Predict"
2. Review the results:
   - Predicted value with units
   - Model name and evaluation metrics
   - Confidence score (Blast Furnace only)
   - Warnings and limitations
3. View translated explanation (if LLM available)

**Verification:** Target names, units, and input requirements should be clear. Validation errors should be understandable and not crash the application. Confidence should be described as an applicability indicator, not calibrated uncertainty.

---

## Workflow 4: RAG Chat (Technical Document Query)

### Step 1: Ask a Technical Question
1. Navigate to the **🔍 RAG Chat** tab
2. Enter a question about metallurgy, e.g.:
   - "What is the basicity range for sinter?"
   - "Explain sinter reducibility"
   - "What factors affect softening temperature?"

### Step 2: Review Retrieved Passages
1. Review the retrieved passages with source attribution
2. Check source filenames, sections, and roles (reference vs experiment_scenario)
3. Verify relevance scores (ordering signal, not confidence)

### Step 3: Review Generated Answer
1. Read the generated answer
2. Verify citations correspond to retrieved evidence
3. Check if the response accurately represents cited material
4. Look for qualified language on uncertain claims

**Verification:** Citations should correspond to retrieved evidence. Unsupported claims should be qualified or declined. Model errors and empty retrieval results should be handled safely.

---

## Edge Cases to Test

### Empty Dataset
- Upload an empty file → Should show error message
- Run analysis on empty dataset → Should report empty dataset/schema

### Invalid Input
- Enter missing required features → Should show "Missing chemistry values: [...]"
- Enter out-of-range values → Should show warnings (⚠ icon)
- Enter invalid data types → Should show validation error

### Unanswerable Question
- Ask about sun temperature (not in corpus) → May return low-relevance passages, but should not invent an answer

---

## Known Limitations

1. **RAG generation not verified in headless environment:** End-to-end RAG generation requires interactive UI testing. This guide assumes the UI is launched and operated manually.

2. **Dual EDA engines:** The legacy engine (`parse/eda.py`) and modern engine (`parse/analysis.py`) are both maintained. The UI uses the modern engine.

3. **Confidence is heuristic:** The Blast Furnace confidence score is based on nearest-neighbor distance, not calibrated prediction uncertainty.

4. **No held-out test set for BF models:** Evaluation is based on 5-fold cross-validation only.

5. **Lexical overlap guard is heuristic:** The grounding guard uses lexical overlap, not semantic entailment.

---

## Troubleshooting

### Application won't launch
- Check that Python and dependencies are installed: `python setup_check.py`
- Verify LocalModels/ and EmbedModels/ directories exist

### Models not loading
- Check MLModels/ directory for model artifacts
- Verify model filenames match expected format

### RAG generation fails
- Check that LocalModels/ contains the LLM weights
- Verify documents are indexed (check docs/ directory)

### Tests failing
- Run setup check: `python setup_check.py`
- Run test suite: `python -m pytest tests/ -v`
