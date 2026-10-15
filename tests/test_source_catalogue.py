import sqlite3
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest

from data.loader import load_and_chunk_data
from parse.core.contracts import Issue
from parse.source_catalogue import (
    SourceCatalogue,
    SourceCategoryConflict,
    SourceIntegrityError,
    SourceIngestionError,
)


def make_catalogue(tmp_path):
    return SourceCatalogue(
        tmp_path / "catalogue" / "sources.sqlite3",
        originals_dir=tmp_path / "private" / "originals",
        public_originals_dir=tmp_path / "public" / "originals",
        normalized_dir=tmp_path / "normalized",
    )


def test_catalogue_ingests_text_with_stable_identity_and_provenance(tmp_path):
    catalogue = make_catalogue(tmp_path)
    payload = b"TOPIC: sinter\n\nSECTION: overview\nSinter supports ironmaking."

    first = catalogue.register_upload(
        "first-name.txt",
        payload,
        source_location="upload:first-name.txt",
    )
    repeated = catalogue.register_upload(
        "renamed.txt",
        payload,
        source_location="C:/incoming/renamed.txt",
    )
    restored = make_catalogue(tmp_path).get_source(first.manifest.resource_id)

    assert repeated.manifest.resource_id == first.manifest.resource_id
    assert restored is not None
    assert restored.manifest.resource_id == first.manifest.resource_id
    assert repeated.manifest.processing_state == "normalized"
    assert set(repeated.source_locations) == {
        "upload:first-name.txt",
        "C:/incoming/renamed.txt",
    }
    assert len(repeated.artifacts) == 1
    artifact = repeated.artifacts[0]
    assert artifact.kind == "document"
    assert artifact.provenance.sources[0].source_id == first.manifest.source.source_id
    assert artifact.provenance.parent_ids == [first.manifest.source.source_id]
    assert (tmp_path / "private" / "originals" / first.manifest.resource_id / "first-name.txt").read_bytes() == payload
    assert not (tmp_path / "private" / "originals" / first.manifest.resource_id / "renamed.txt").exists()
    assert "SECTION: overview" in (tmp_path / "normalized" / "documents" / f"{first.manifest.resource_id}.txt").read_text(encoding="utf-8")
    assert load_and_chunk_data(tmp_path / "normalized" / "documents")[0].content == "Sinter supports ironmaking."


def test_catalogue_does_not_conflate_different_content_with_same_filename(tmp_path):
    catalogue = make_catalogue(tmp_path)
    first = catalogue.register_upload("same.txt", b"SECTION: overview\nFirst document.")
    second = catalogue.register_upload("same.txt", b"SECTION: overview\nSecond document.")

    assert first.manifest.resource_id != second.manifest.resource_id
    assert len(catalogue.list_sources()) == 2
    assert (tmp_path / "private" / "originals" / first.manifest.resource_id / "same.txt").read_bytes().endswith(b"First document.")
    assert (tmp_path / "private" / "originals" / second.manifest.resource_id / "same.txt").read_bytes().endswith(b"Second document.")


def test_catalogue_stores_public_originals_separately(tmp_path):
    catalogue = make_catalogue(tmp_path)
    entry = catalogue.register_upload(
        "paper.txt",
        b"SECTION: overview\nPublicly available research note.",
        resource_class="public_research",
    )

    stored = tmp_path / "public" / "originals" / entry.manifest.resource_id / "paper.txt"
    assert stored.is_file()
    assert not (
        tmp_path / "private" / "originals" / entry.manifest.resource_id / "paper.txt"
    ).exists()
    assert entry.manifest.metadata["original_storage_scope"] == "public"


def test_catalogue_registers_local_file_and_keeps_its_location(tmp_path):
    catalogue = make_catalogue(tmp_path)
    local_source = tmp_path / "incoming" / "data.csv"
    local_source.parent.mkdir()
    payload = b"temperature\n1200\n"
    local_source.write_bytes(payload)

    entry = catalogue.register_file(local_source)

    assert str(local_source.resolve()) in entry.source_locations
    assert entry.manifest.source.locator == str(local_source.resolve())
    preserved = (
        tmp_path
        / "private"
        / "originals"
        / entry.manifest.resource_id
        / "data.csv"
    )
    assert preserved.read_bytes() == payload


def test_register_file_enforces_size_limit_before_unbounded_read(tmp_path, monkeypatch):
    import parse.eda_ui

    catalogue = make_catalogue(tmp_path)
    local_source = tmp_path / "incoming.csv"
    local_source.write_bytes(b"12345")
    monkeypatch.setattr(parse.eda_ui, "MAX_SOURCE_BYTES", 4)

    with pytest.raises(ValueError, match="Source files must be no larger"):
        catalogue.register_file(local_source)

    assert catalogue.list_sources() == []


def test_catalogue_normalizes_tabular_upload_without_storing_rows_in_metadata(tmp_path):
    catalogue = make_catalogue(tmp_path)
    payload = b"temperature,grade\n1200,0.62\n1300,0.65\n"

    entry = catalogue.register_upload("results.csv", payload)

    assert entry.manifest.processing_state == "normalized"
    assert entry.manifest.source.source_type == "csv"
    assert entry.artifacts[0].kind == "structured_record"
    assert entry.artifacts[0].attributes["row_count"] == 2
    assert entry.artifacts[0].attributes["columns"] == ["temperature", "grade"]
    assert entry.artifacts[0].attributes["inferred_types"] == {
        "temperature": "int64",
        "grade": "float64",
    }
    assert "temperature" not in entry.manifest.metadata
    normalized = tmp_path / "normalized" / "datasets" / f"{entry.manifest.resource_id}.csv"
    assert normalized.read_bytes() == payload
    assert pd.read_csv(normalized).to_dict(orient="records") == [
        {"temperature": 1200, "grade": 0.62},
        {"temperature": 1300, "grade": 0.65},
    ]


def test_excel_registration_documents_first_sheet_selection_and_preserves_workbook(tmp_path):
    catalogue = make_catalogue(tmp_path)
    workbook = BytesIO()
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame({"reading": [1, 2]}).to_excel(
            writer, sheet_name="First", index=False
        )
        pd.DataFrame({"reading": [3]}).to_excel(
            writer, sheet_name="Second", index=False
        )
    payload = workbook.getvalue()

    entry = catalogue.register_upload("measurements.xlsx", payload)

    assert entry.artifacts[0].attributes["sheet_names"] == ["First", "Second"]
    assert entry.artifacts[0].attributes["selected_sheet"] == "First"
    assert any(
        issue.code == "additional_excel_sheets_not_normalized"
        for issue in entry.manifest.issues
    )
    original = (
        tmp_path / "private" / "originals"
        / entry.manifest.metadata["preserved_original"]
    )
    assert original.read_bytes() == payload
    normalized = tmp_path / "normalized" / entry.artifacts[0].attributes["path"]
    assert pd.read_csv(normalized)["reading"].tolist() == [1, 2]


def test_catalogue_retains_and_reports_dataset_missingness(tmp_path):
    catalogue = make_catalogue(tmp_path)

    entry = catalogue.register_upload("missing.csv", b"reading,comment\n1,\n2,NA\n")

    assert entry.manifest.processing_state == "normalized"
    assert entry.artifacts[0].attributes["missing_value_counts"] == {"comment": 2}
    assert len(entry.manifest.issues) == 1
    issue = entry.manifest.issues[0]
    assert issue.code == "missing_values_present"
    assert issue.field == "comment"
    assert "2 missing value(s)" in issue.message


@pytest.mark.parametrize(
    ("filename", "payload"),
    [
        ("records.json", b'[{"reading": 1, "unit": "C"}, {"reading": 2, "unit": "C"}]'),
        ("records.jsonl", b'{"reading": 1, "unit": "C"}\n{"reading": 2, "unit": "C"}\n'),
    ],
)
def test_catalogue_validates_and_normalizes_json_records(tmp_path, filename, payload):
    catalogue = make_catalogue(tmp_path)

    entry = catalogue.register_upload(filename, payload)

    assert entry.manifest.processing_state == "normalized"
    assert entry.manifest.content_type == "json"
    assert entry.manifest.metadata["original_file_type"] in {".json", ".jsonl"}
    assert entry.artifacts[0].attributes["row_count"] == 2
    normalized = tmp_path / "normalized" / entry.artifacts[0].attributes["path"]
    assert pd.read_csv(normalized).to_dict(orient="records") == [
        {"reading": 1, "unit": "C"},
        {"reading": 2, "unit": "C"},
    ]
    original = (
        tmp_path / "private" / "originals"
        / entry.manifest.metadata["preserved_original"]
    )
    assert original.read_bytes() == payload


def test_malformed_json_is_not_registered_as_a_normalized_dataset(tmp_path):
    catalogue = make_catalogue(tmp_path)

    with pytest.raises(SourceIngestionError, match="could not be normalized"):
        catalogue.register_upload("broken.json", b'{"reading": 1}')

    entry = catalogue.list_sources()[0]
    assert entry.manifest.processing_state == "failed"
    assert entry.artifacts == []
    assert entry.manifest.issues[0].severity == "error"


def test_exact_duplicate_preflight_is_hash_based_and_reports_classification_conflict(tmp_path):
    catalogue = make_catalogue(tmp_path)
    payload = b"reading\n1\n"
    entry = catalogue.register_upload("first.csv", payload)

    duplicate = catalogue.find_duplicate("renamed.csv", payload)
    conflict = catalogue.find_duplicate(
        "renamed.csv",
        payload,
        resource_class="public_dataset",
    )
    different_content = catalogue.find_duplicate("first.csv", b"reading\n2\n")

    assert duplicate is not None
    assert duplicate.duplicate_type == "exact_file"
    assert duplicate.existing_resource_id == entry.manifest.resource_id
    assert duplicate.physical_storage_reused
    assert not duplicate.new_source_record_created
    assert duplicate.classification_conflict is None
    assert conflict is not None
    assert conflict.classification_conflict is not None
    assert conflict.physical_storage_reused
    assert different_content is None


def test_identical_bytes_cannot_be_registered_as_a_different_file_type(tmp_path):
    catalogue = make_catalogue(tmp_path)
    payload = b"reading\n1\n"
    existing = catalogue.register_upload("measurements.csv", payload)

    duplicate = catalogue.find_duplicate("notes.txt", payload)
    with pytest.raises(SourceCategoryConflict, match="already registered as 'csv'"):
        catalogue.register_upload("notes.txt", payload)

    assert duplicate is not None
    assert duplicate.existing_resource_id == existing.manifest.resource_id
    assert "not 'document'" in duplicate.classification_conflict
    assert len(catalogue.list_sources()) == 1


def test_concurrent_exact_uploads_create_one_catalogue_source(tmp_path):
    catalogue = make_catalogue(tmp_path)
    catalogue._initialize()
    payload = b"reading\n1\n2\n"

    with ThreadPoolExecutor(max_workers=4) as executor:
        entries = list(
            executor.map(
                lambda _: catalogue.register_upload("same.csv", payload),
                range(4),
            )
        )

    assert len({entry.manifest.resource_id for entry in entries}) == 1
    assert len(catalogue.list_sources()) == 1
    assert len(catalogue.list_sources()[0].artifacts) == 1


def test_registration_refuses_to_reuse_corrupted_preserved_original(tmp_path):
    catalogue = make_catalogue(tmp_path)
    payload = b"reading\n1\n"
    entry = catalogue.register_upload("data.csv", payload)
    original = (
        tmp_path / "private" / "originals"
        / entry.manifest.metadata["preserved_original"]
    )
    original.write_bytes(b"tampered")

    duplicate = catalogue.find_duplicate("renamed.csv", payload)
    with pytest.raises(SourceIntegrityError, match="does not match"):
        catalogue.register_upload("renamed.csv", payload)

    assert duplicate is not None
    assert not duplicate.artifact_verified
    assert len(catalogue.list_sources()) == 1


def test_reconciliation_reports_missing_artifacts_without_deleting_original(tmp_path):
    catalogue = make_catalogue(tmp_path)
    entry = catalogue.register_upload("data.csv", b"reading\n1\n")
    normalized = tmp_path / "normalized" / entry.artifacts[0].attributes["path"]
    normalized.unlink()

    problems = catalogue.reconcile()
    repaired = catalogue.get_source(entry.manifest.resource_id)
    original = (
        tmp_path / "private" / "originals"
        / entry.manifest.metadata["preserved_original"]
    )

    assert problems == [
        {
            "resource_id": entry.manifest.resource_id,
            "artifact": entry.artifacts[0].attributes["path"],
            "problem": "normalized artifact is missing",
        }
    ]
    assert repaired is not None
    assert repaired.manifest.processing_state == "failed"
    assert original.read_bytes() == b"reading\n1\n"


def test_reconciliation_reports_manifest_artifact_missing_catalogue_row(tmp_path):
    catalogue = make_catalogue(tmp_path)
    entry = catalogue.register_upload("data.csv", b"reading\n1\n")
    with sqlite3.connect(catalogue.database_path) as connection:
        connection.execute(
            "DELETE FROM artifacts WHERE resource_id = ?",
            (entry.manifest.resource_id,),
        )

    problems = catalogue.reconcile()

    assert problems == [
        {
            "resource_id": entry.manifest.resource_id,
            "artifact": entry.manifest.derived_artifacts[0],
            "problem": "manifest references an artifact without a catalogue artifact record",
        }
    ]


def test_catalogue_persists_processing_failure_and_original(tmp_path):
    catalogue = make_catalogue(tmp_path)
    payload = b"REFERENCES\nThis is deliberately excluded from retrieval."

    with pytest.raises(SourceIngestionError, match="registered but could not be normalized") as exc_info:
        catalogue.register_upload("no-content.txt", payload)

    entry = exc_info.value.entry
    assert entry.manifest.processing_state == "failed"
    assert entry.manifest.issues == [
        Issue(
            "source_normalization_failed",
            "error",
            "Text source contains no indexable sections",
            source=entry.manifest.source,
        )
    ]
    assert (tmp_path / "private" / "originals" / entry.manifest.resource_id / "no-content.txt").read_bytes() == payload
    assert catalogue.get_source(entry.manifest.resource_id).manifest.processing_state == "failed"


def test_catalogue_rejects_unsupported_files_without_registering(tmp_path):
    catalogue = make_catalogue(tmp_path)

    with pytest.raises(ValueError, match="Supported source files"):
        catalogue.register_upload("archive.zip", b"not supported")

    assert catalogue.list_sources() == []


def test_confirmed_upload_preserves_original_and_stores_selected_tabular_artifact(tmp_path):
    catalogue = make_catalogue(tmp_path)
    original = b"temperature,grade\n1200,0.62\n1300,0.65\n"
    selected = b"temperature,grade\n1200,0.62\n"
    workflow = {
        "selected_version": "Reviewed working copy",
        "approved_transformations": ["proposal-1"],
        "unresolved_issue_ids": ["issue-1"],
    }

    first = catalogue.register_confirmed_upload(
        "results.csv",
        original,
        final_content=selected,
        title="Reviewed results",
        resource_class="private_operational_data",
        contexts=["sinter"],
        workflow_metadata=workflow,
    )
    repeated = catalogue.register_confirmed_upload(
        "renamed.csv",
        original,
        final_content=selected,
        title="Reviewed results",
        resource_class="private_operational_data",
        contexts=["sinter"],
        workflow_metadata=workflow,
    )

    original_path = (
        tmp_path / "private" / "originals" / first.manifest.resource_id / "results.csv"
    )
    final_path = tmp_path / "normalized" / "datasets" / (
        f"{first.manifest.resource_id}-"
        f"{first.manifest.metadata['final_content_sha256'][:16]}.csv"
    )
    assert original_path.read_bytes() == original
    assert final_path.read_bytes() == selected
    assert first.manifest.metadata["final_content_sha256"] != (
        first.manifest.metadata["content_sha256"]
    )
    assert first.manifest.metadata["workflow"] == workflow
    assert first.manifest.contexts == ["sinter"]
    assert repeated.manifest.resource_id == first.manifest.resource_id
    assert len(repeated.artifacts) == 2


def test_duplicate_content_cannot_be_registered_under_a_different_category(tmp_path):
    catalogue = make_catalogue(tmp_path)
    payload = b"TOPIC: sinter\nSECTION: overview\nShared source content."
    catalogue.register_upload(
        "guide.txt",
        payload,
        resource_class="internal_technical_document",
    )

    with pytest.raises(SourceCategoryConflict, match="already registered as"):
        catalogue.register_confirmed_upload(
            "guide.txt",
            payload,
            title="Guide",
            resource_class="public_research",
        )

    assert len(catalogue.list_sources()) == 1


def test_index_refresh_state_is_inspectable_and_retryable(tmp_path):
    catalogue = make_catalogue(tmp_path)
    entry = catalogue.register_upload(
        "guide.txt",
        b"TOPIC: sinter\nSECTION: overview\nRetrieval evidence.",
    )

    failed = catalogue.mark_index_failed(entry.manifest.resource_id, "index offline")
    assert failed.manifest.processing_state == "failed"
    assert failed.manifest.metadata["processing_stage"] == "indexing"
    assert failed.manifest.metadata["index_error"] == "index offline"
    assert failed.manifest.issues[-1].code == "source_indexing_failed"

    retried = catalogue.mark_indexed(entry.manifest.resource_id, 1)
    assert retried.manifest.processing_state == "indexed"
    assert retried.manifest.metadata["processing_stage"] == "indexed"
    assert retried.manifest.metadata["indexed_chunk_count"] == 1
    assert retried.manifest.metadata["index_error"] is None
    assert all(issue.code != "source_indexing_failed" for issue in retried.manifest.issues)


def test_registration_failure_can_be_retried_without_replacing_original(tmp_path):
    catalogue = make_catalogue(tmp_path)
    original = b"value\n1\n2\n"
    entry = catalogue.register_upload("data.csv", original)
    failed = catalogue.mark_registration_failed(
        entry.manifest.resource_id, "final artifact write interrupted"
    )

    assert failed.manifest.processing_state == "failed"
    assert failed.manifest.metadata["processing_stage"] == "registration"
    assert failed.manifest.issues[-1].code == "source_registration_failed"

    retried = catalogue.register_confirmed_upload(
        "data.csv",
        original,
        final_content=b"value\n1\n",
        resource_class="private_operational_data",
    )
    preserved = (
        tmp_path / "private" / "originals" / entry.manifest.resource_id / "data.csv"
    )
    assert retried.manifest.processing_state == "normalized"
    assert retried.manifest.metadata["processing_stage"] == "normalized"
    assert preserved.read_bytes() == original
    assert all(
        issue.code != "source_registration_failed"
        for issue in retried.manifest.issues
    )


def test_failed_final_catalogue_update_keeps_original_and_can_be_retried(tmp_path, monkeypatch):
    catalogue = make_catalogue(tmp_path)
    original = b"reading\n1\n2\n"
    original_store = catalogue._store_artifact

    def fail_once(*args, **kwargs):
        monkeypatch.setattr(catalogue, "_store_artifact", original_store)
        raise sqlite3.OperationalError("simulated catalogue update failure")

    monkeypatch.setattr(catalogue, "_store_artifact", fail_once)
    with pytest.raises(sqlite3.OperationalError, match="simulated"):
        catalogue.register_confirmed_upload(
            "data.csv",
            original,
            final_content=b"reading\n1\n",
            resource_class="private_operational_data",
        )

    duplicate = catalogue.find_duplicate("data.csv", original)
    assert duplicate is not None
    resource_id = duplicate.existing_resource_id
    failed = catalogue.get_source(resource_id)
    preserved = (
        tmp_path / "private" / "originals" / resource_id / "data.csv"
    )
    assert failed is not None
    assert failed.manifest.processing_state == "failed"
    assert preserved.read_bytes() == original

    retried = catalogue.register_confirmed_upload(
        "data.csv",
        original,
        final_content=b"reading\n1\n",
        resource_class="private_operational_data",
    )

    assert retried.manifest.processing_state == "normalized"
    assert len(retried.artifacts) == 2
    assert catalogue.reconcile() == []


def test_existing_rag_engine_includes_catalogued_documents(tmp_path, monkeypatch):
    import pipeline.rag_pipeline as rag_pipeline

    catalogue = make_catalogue(tmp_path)
    catalogue.register_upload(
        "guide.txt",
        b"TOPIC: sinter\nSECTION: overview\nCatalogued retrieval evidence.",
    )
    project_docs = tmp_path / "project_docs"
    project_docs.mkdir()

    class FakeEngine:
        def __init__(self, chunks):
            self.chunks = chunks

    monkeypatch.setattr(rag_pipeline, "SOURCE_NORMALIZED_DIR", tmp_path / "normalized")
    monkeypatch.setattr(rag_pipeline, "RetrievalEngine", FakeEngine)

    engine, chunks = rag_pipeline.build_engine(project_docs)

    assert engine.chunks == chunks
    assert len(chunks) == 1
    assert chunks[0].content == "Catalogued retrieval evidence."
    assert chunks[0].source.startswith("resource-")
