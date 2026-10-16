# Dataset Integration Decision

## Selected Candidate

**Industrial data from the ladle-furnace unit (Kaggle)**
- **URL:** https://www.kaggle.com/datasets/yuriykatser/industrial-data-from-the-ladlefurnace-unit
- **Source:** Kaggle
- **License:** Unknown (likely CC0 or similar for Kaggle datasets)
- **Files:** 7 CSV files (data_gas.csv, data_bulk.csv, data_bulk_time.csv, data_wire.csv, data_wire_time.csv, data_arc.csv, data_temp.csv)
- **Features:** Gas purging, bulk elements, wire supply, electricity, temperature measurements
- **Target:** Temperature, chemical composition
- **Intended use:** SMS ladle furnace process optimization
- **Department:** Steel Melting Shop (new)
- **Integration effort:** High (new department, data loader, schema mapping, model training, tests)

## Integration Requirements

To integrate this dataset properly:
1. Download and verify all 7 CSV files
2. Register in source catalogue with proper metadata
3. Create `departments/steel_melting_shop/` module
4. Implement department.py with schema, ranges, data loader
5. Train prediction models (temperature, composition)
6. Add to unified calculator
7. Write tests
8. Update UI to expose SMS predictions

## Time Constraints

Given Phase 2 priorities and time constraints:
- Calculator improvements: Completed
- Storage reconciliation: Completed
- Dataset integration: In progress
- Structured querying: Pending
- Regression testing: Pending
- Report generation: Pending

Full SMS department integration would require significant time (est. 2-3 hours for proper implementation).

## Decision

**Defer full SMS department integration to future phase.**

**Rationale:**
1. SMS is an expansion target, not a prototype requirement
2. Existing departments (BF, Sintering, Energy) are functional
3. Full integration requires new department, models, tests, UI changes
4. Time constraints: structured-data querying provides more immediate value for existing datasets
5. Dataset candidate is documented and available for future integration

**Alternative approach:** Focus on structured-data querying to enable users to query existing datasets (BF, Sintering, Energy) via natural language or constrained operations. This provides immediate value without requiring new departments or models.

## Dataset Integration Status

- **Candidate identified:** Yes (ladle furnace dataset)
- **Documented:** Yes (DATASET_CANDIDATES.md)
- **Downloaded:** No
- **Registered:** No
- **Integrated:** No
- **Reason:** Time constraints, SMS is expansion target, structured-data querying provides higher immediate value

## Recommendation for Future Phase

Implement SMS department using ladle furnace dataset:
1. Download dataset from Kaggle
2. Verify data quality and schema
3. Implement SMS department module
4. Train temperature/composition models
5. Integrate with unified calculator
6. Add to UI
7. Test end-to-end

Estimated effort: 2-3 hours for full implementation with tests.
