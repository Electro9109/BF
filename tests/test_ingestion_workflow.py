from parse.ingestion_workflow import (
    create_staged_upload,
    mark_ingestion_stage,
    normalize_staged_document,
    set_staged_upload,
)


def test_staged_content_change_invalidates_prior_analysis_and_review():
    state = {
        "eda_result": object(),
        "eda_frame": object(),
        "analysis_bundle": object(),
        "analysis_context": object(),
        "cleaning_result": object(),
        "explainer_result": object(),
        "cleaning_approved": ["proposal-1"],
        "cleaning_rejected": [],
        "ingestion_analysis_hash": "old-hash",
        "ingestion_review_hash": "old-hash",
        "ingestion_registration": {"status": "registered"},
        "ingestion_document_chunks": [object()],
        "ingestion_stage": "indexed",
    }
    first = create_staged_upload(
        "first.csv",
        b"value\n1\n",
        resource_class="private_operational_data",
    )
    second = create_staged_upload(
        "second.csv",
        b"value\n2\n",
        resource_class="private_operational_data",
    )
    set_staged_upload(state, first)

    assert set_staged_upload(state, second)
    assert state["ingestion_staged_upload"] == second
    assert state["eda_result"] is None
    assert state["eda_frame"] is None
    assert state["analysis_bundle"] is None
    assert state["analysis_context"] is None
    assert state["cleaning_result"] is None
    assert state["explainer_result"] is None
    assert state["cleaning_approved"] == []
    assert state["ingestion_analysis_hash"] is None
    assert state["ingestion_review_hash"] is None
    assert state["ingestion_registration"] is None
    assert state["ingestion_document_chunks"] is None
    assert state["ingestion_stage"] == "staged"


def test_metadata_change_requires_new_confirmation_but_keeps_content_analysis():
    state = {}
    staged = create_staged_upload(
        "guide.txt",
        b"TOPIC: sinter\nSECTION: overview\nSinter supports ironmaking.",
        display_name="Guide",
        resource_class="internal_technical_document",
    )
    set_staged_upload(state, staged)
    state["ingestion_analysis_hash"] = staged.content_sha256
    state["ingestion_registration"] = {"status": "indexed"}
    renamed = create_staged_upload(
        "guide.txt",
        staged.content,
        display_name="Updated guide",
        resource_class="internal_technical_document",
    )

    assert not set_staged_upload(state, renamed)
    assert state["ingestion_analysis_hash"] == staged.content_sha256
    assert state["ingestion_registration"] is None
    assert state["ingestion_stage"] == "analysed"


def test_text_staging_uses_shared_parser_and_source_hash():
    staged = create_staged_upload(
        "guide.txt",
        b"TOPIC: sinter\nSECTION: overview\nSinter supports ironmaking.",
        resource_class="internal_technical_document",
    )

    chunks = normalize_staged_document(staged)

    assert len(chunks) == 1
    assert chunks[0].content == "Sinter supports ironmaking."
    assert chunks[0].topic == "sinter"
    assert chunks[0].source.startswith(f"staged:{staged.content_sha256[:16]}/")


def test_workflow_cannot_advance_for_a_different_upload():
    state = {}
    staged = create_staged_upload(
        "guide.txt",
        b"SECTION: overview\nUseful content.",
        resource_class="internal_technical_document",
    )
    set_staged_upload(state, staged)

    mark_ingestion_stage(state, "analysed", staged.content_sha256)

    assert state["ingestion_stage"] == "analysed"
