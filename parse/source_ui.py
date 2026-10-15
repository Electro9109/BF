"""Minimal Streamlit view for registering and inspecting PARSE sources."""

from __future__ import annotations

from config.paths import DOCS_DIR
from pipeline.rag_pipeline import build_engine
from parse.source_catalogue import CatalogueEntry, SourceCatalogue, SourceIngestionError


def _catalogue_rows(entries: list[CatalogueEntry]) -> list[dict[str, object]]:
    rows = []
    for entry in entries:
        manifest = entry.manifest
        rows.append(
            {
                "Source": manifest.title,
                "Type": manifest.content_type,
                "Location": ", ".join(entry.source_locations),
                "Status": manifest.processing_state,
                "Artifacts": len(entry.artifacts),
                "Issues": len(manifest.issues),
            }
        )
    return rows


def render_source_management() -> None:
    """Render upload, registration, status, artifact, and issue details."""
    import streamlit as st

    st.subheader("Source catalogue")
    st.caption(
        "Original files are retained separately from normalized artifacts. "
        "Repeated uploads of identical content reuse the same source identity."
    )
    uploaded = st.file_uploader(
        "Register a document or dataset",
        type=["txt", "csv", "xlsx", "xls"],
        key="parse_source_upload",
    )
    resource_classes = {
        "Internal technical document": "internal_technical_document",
        "Private operational data": "private_operational_data",
        "Public research": "public_research",
        "Public dataset": "public_dataset",
        "Standards reference": "standards_reference",
    }
    if uploaded is not None:
        default_class = (
            "Internal technical document"
            if uploaded.name.lower().endswith(".txt")
            else "Private operational data"
        )
        selected_class = st.selectbox(
            "Source category",
            options=list(resource_classes),
            index=list(resource_classes).index(default_class),
            key=f"parse_source_class_{uploaded.name}",
        )
        title = st.text_input(
            "Display name",
            value=uploaded.name,
            key=f"parse_source_title_{uploaded.name}",
        )
        if st.button(
            "Register and process source",
            type="primary",
            key="register_parse_source",
        ):
            catalogue = SourceCatalogue()
            try:
                entry = catalogue.register_upload(
                    uploaded.name,
                    uploaded.getvalue(),
                    title=title or uploaded.name,
                    resource_class=resource_classes[selected_class],
                    source_location=f"upload:{uploaded.name}",
                )
            except SourceIngestionError as exc:
                st.error(str(exc))
                entry = exc.entry
            else:
                st.success(
                    f"Registered {entry.manifest.title} "
                    f"({entry.manifest.processing_state})."
                )
            if (
                entry.manifest.processing_state == "normalized"
                and entry.manifest.source is not None
                and entry.manifest.source.source_type == "document"
            ):
                try:
                    with st.spinner("Refreshing retrieval index..."):
                        engine, chunks = build_engine(DOCS_DIR)
                    st.session_state.engine = engine
                    st.session_state.chunks = chunks
                    st.session_state.engine_error = None
                    st.success(f"Retrieval index refreshed with {len(chunks)} chunks.")
                except (OSError, ValueError, RuntimeError) as exc:
                    st.error(
                        "Source was registered, but the retrieval index could not "
                        f"be refreshed: {exc}"
                    )

    entries = SourceCatalogue().list_sources()
    if not entries:
        st.info("No sources have been registered yet.")
        return

    st.dataframe(_catalogue_rows(entries), use_container_width=True, hide_index=True)
    by_id = {entry.manifest.resource_id: entry for entry in entries}
    selected_id = st.selectbox(
        "Inspect source details",
        options=list(by_id),
        format_func=lambda resource_id: (
            f"{by_id[resource_id].manifest.title} "
            f"({by_id[resource_id].manifest.processing_state})"
        ),
        key="parse_source_inspection",
    )
    selected = by_id[selected_id]
    st.write("**Original locations**")
    for location in selected.source_locations:
        st.code(location)
    preserved_original = selected.manifest.metadata.get("preserved_original")
    if preserved_original:
        st.write("**Preserved original copy**")
        storage_scope = selected.manifest.metadata.get(
            "original_storage_scope",
            "private",
        )
        st.code(f"{storage_scope}/originals/{preserved_original}")
    if selected.artifacts:
        st.write("**Derived artifacts**")
        for artifact in selected.artifacts:
            st.json(artifact.to_dict())
    if selected.manifest.issues:
        st.write("**Processing issues**")
        for issue in selected.manifest.issues:
            st.warning(f"{issue.severity}: {issue.message}")
    else:
        st.caption("No processing issues recorded.")


__all__ = ["render_source_management"]
