import pandas as pd
import pytest

from data.loader import load_and_chunk_data
from parse.core.contracts import Issue
from parse.source_catalogue import SourceCatalogue, SourceIngestionError


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


def test_catalogue_normalizes_tabular_upload_without_storing_rows_in_metadata(tmp_path):
    catalogue = make_catalogue(tmp_path)
    payload = b"temperature,grade\n1200,0.62\n1300,0.65\n"

    entry = catalogue.register_upload("results.csv", payload)

    assert entry.manifest.processing_state == "normalized"
    assert entry.manifest.source.source_type == "csv"
    assert entry.artifacts[0].kind == "structured_record"
    assert entry.artifacts[0].attributes["row_count"] == 2
    assert entry.artifacts[0].attributes["columns"] == ["temperature", "grade"]
    assert "temperature" not in entry.manifest.metadata
    normalized = tmp_path / "normalized" / "datasets" / f"{entry.manifest.resource_id}.csv"
    assert pd.read_csv(normalized).to_dict(orient="records") == [
        {"temperature": 1200, "grade": 0.62},
        {"temperature": 1300, "grade": 0.65},
    ]


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
