from pathlib import Path

from parse.core.contracts import KnowledgeItem, Provenance, SourceRef
from parse.core.resource_manifest import ResourceManifest
from parse.source_ui import _catalogue_rows
from parse.source_catalogue import CatalogueEntry

APP_WEB_PATH = Path(__file__).resolve().parent.parent / "app_web.py"


def test_source_catalogue_table_exposes_type_location_status_and_issues():
    source = SourceRef("source-1", "document", "upload:guide.txt", "Guide")
    entry = CatalogueEntry(
        manifest=ResourceManifest(
            resource_id="resource-1",
            resource_class="internal_technical_document",
            content_type="document",
            title="Guide",
            source=source,
            processing_state="normalized",
            source_location="upload:guide.txt",
            derived_artifacts=["documents/guide.txt"],
        ),
        source_locations=["upload:guide.txt"],
        artifacts=[
            KnowledgeItem(
                item_id="artifact-1",
                kind="document",
                attributes={"path": "documents/guide.txt"},
                provenance=Provenance(sources=[source]),
            )
        ],
    )

    assert _catalogue_rows([entry]) == [
        {
            "Source": "Guide",
            "Type": "document",
            "Location": "upload:guide.txt",
            "Status": "normalized",
            "Stage": "",
            "Index error": "",
            "Artifacts": 1,
            "Issues": 0,
        }
    ]


def test_source_catalogue_is_integrated_in_data_ingestion_without_extra_tab():
    source = APP_WEB_PATH.read_text(encoding="utf-8")

    assert '"📥 Data Ingestion"' in source
    assert "with tab_ingestion:" in source
    assert "render_catalogue_inspection" in source
    assert "tab_sources" not in source
    assert '"📚 Sources"' not in source
    assert source.count("st.file_uploader(") == 1
    assert "dest.write_bytes" not in source
