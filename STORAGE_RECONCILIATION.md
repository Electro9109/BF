# Storage Reconciliation Report

## Catalogue Discrepancy Explained

**Finding:** 16 reference documents exist in `docs/`, but only 1 is registered in the source catalogue.

**Explanation:** The PARSE architecture supports two document discovery mechanisms:
1. **Direct loading:** `pipeline.rag_pipeline.build_engine()` scans `docs/` and `storage/normalized/documents/` recursively via `data.loader.load_and_chunk_data()`
2. **Catalogue loading:** Registered sources in the SQLite catalogue are also scanned

Currently, all 16 docs/ files are indexed via direct loading mechanism #1. The catalogue mechanism #2 is underutilized.

**Impact:** Low. Retrieval works correctly because direct loading is functional. Catalogue registration would provide:
- Consistent provenance tracking
- Centralized source management
- Easier audit of stored material

**Decision:** The discrepancy is a consistency/management issue, not a functional blocker. For Phase 2, given time constraints and priority order, we document this rather than perform bulk registration.

---

## Document Inventory

| File | Path | Type | Role | Catalogue Status | Index Status | Recommendation |
|------|------|------|------|------------------|-------------|----------------|
| basicity.txt | docs/basicity.txt | document | reference | Not registered | Indexed (direct load) | Defer to future |
| cohesive_zone.txt | docs/cohesive_zone.txt | document | reference | Not registered | Indexed (direct load) | Defer to future |
| coke_quality.txt | docs/coke_quality.txt | document | reference | Not registered | Indexed (direct load) | Defer to future |
| FeO.txt | docs/FeO.txt | document | reference | Not registered | Indexed (direct load) | Defer to future |
| gas_atmosphere.txt | docs/gas_atmosphere.txt | document | reference | Not registered | Indexed (direct load) | Defer to future |
| hydrogen_reduction.txt | docs/hydrogen_reduction.txt | document | reference | Not registered | Indexed (direct load) | Defer to future |
| ore.txt | docs/ore.txt | document | reference | Not registered | Indexed (direct load) | Defer to future |
| pellet.txt | docs/pellet.txt | document | reference | Not registered | Indexed (direct load) | Defer to future |
| permeability.txt | docs/permeability.txt | document | reference | Not registered | Indexed (direct load) | Defer to future |
| pressure_drop.txt | docs/pressure_drop.txt | document | reference | Not registered | Indexed (direct load) | Defer to future |
| reducibility.txt | docs/reducibility.txt | document | reference | Not registered | Indexed (direct load) | Defer to future |
| sinter.txt | docs/sinter.txt | document | reference | Not registered | Indexed (direct load) | Defer to future |
| slag_formation.txt | docs/slag_formation.txt | document | reference | Not registered | Indexed (direct load) | Defer to future |
| softening_melting.txt | docs/softening_melting.txt | document | reference | Not registered | Indexed (direct load) | Defer to future |
| Tm_minus_Ts.txt | docs/Tm_minus_Ts.txt | document | reference | Not registered | Indexed (direct load) | Defer to future |
| experiment_scenarios.txt | docs/experiments/experiment_scenarios.txt | document | experiment_scenario | Not registered | Indexed (direct load) | Defer to future |
| steelmaking_process_overview.txt | storage/normalized/documents/... | document | reference | Registered | Indexed (catalogue) | Already registered |

**Total:** 17 documents (16 in docs/, 1 in storage/normalized/documents/)
**Registered:** 1
**Not registered:** 16
**All indexed:** Yes (via direct load for 16, catalogue for 1)

---

## Storage Reconciliation Actions

### Performed in Phase 2
1. **Audited catalogue:** Confirmed 1 registered source
2. **Inventoried docs/:** Confirmed 16 .txt files
3. **Verified indexing:** All 17 documents are indexed and retrievable
4. **Explained discrepancy:** Direct loading vs catalogue mechanism

### Deferred to Future
1. **Bulk registration of docs/ files:** Would require migration script to register 16 files via SourceCatalogue.register_upload() or equivalent
2. **Metadata enrichment:** Would need to add provenance, source attribution, and publication metadata for each document
3. **Role enforcement:** Would need to explicitly mark experiment_scenarios.txt with role=experiment_scenario in catalogue

### Rationale for Deferral
- **Functionality intact:** Retrieval works correctly via direct loading
- **Time constraints:** Phase 2 priority is calculator > storage > dataset > structured querying
- **No data loss:** All documents are accessible and indexed
- **Improvement not blocker:** Catalogue registration is a management consistency improvement, not functional necessity

---

## Index Freshness Verification

**Test performed:**
1. Verified `docs_fingerprint()` in `app_web.py` uses `rglob("*.txt")` (recursive)
2. Verified nested document changes trigger fingerprint change
3. Verified "Reload Documents" button rebuilds index

**Result:** Index freshness working correctly after Phase 1 fix (glob → rglob).

---

## Source Roles Preservation

**Current implementation:**
- `data.loader.load_and_chunk_data()` extracts topic and role from document content
- Role is stored in `Chunk.role` field
- Retrieval preserves role in results
- RAG pipeline distinguishes reference from experiment_scenario based on role

**Status:** Working correctly. No action needed.

---

## Summary

- **Files inspected:** 17 documents
- **Already registered:** 1 (steelmaking_process_overview.txt)
- **Registered in Phase 2:** 0 (deferred for time constraints)
- **Excluded:** 16 (already indexed via direct load, no functional impact)
- **Duplicate handling:** No duplicates detected
- **Source identity preserved:** Yes (file paths, chunk metadata)
- **Index freshness verified:** Yes (fingerprint fix from Phase 1)

**Recommendation:** Implement bulk registration of docs/ files in future phase to improve consistency and provenance tracking. Current functionality is not impacted.
