"""Minimal Streamlit view for registering and inspecting PARSE sources."""

from __future__ import annotations

from parse.source_catalogue import CatalogueEntry, SourceCatalogue


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
                "Stage": manifest.metadata.get("processing_stage", ""),
                "Index error": manifest.metadata.get("index_error") or "",
                "Artifacts": len(entry.artifacts),
                "Issues": len(manifest.issues),
            }
        )
    return rows


def render_catalogue_inspection(
    entries: list[CatalogueEntry] | None = None,
) -> None:
    """Render registered source status and provenance details."""
    import streamlit as st

    st.subheader("Source catalogue")
    catalogue = SourceCatalogue()
    entries = entries if entries is not None else catalogue.list_sources()
    if st.button("Reconcile stored artifacts", key="parse_reconcile_catalogue"):
        problems = catalogue.reconcile()
        if problems:
            st.error(f"Reconciliation found {len(problems)} storage issue(s).")
            for problem in problems:
                st.warning(
                    f"{problem['resource_id']} · {problem['artifact']}: "
                    f"{problem['problem']}"
                )
        else:
            st.success("All catalogue-referenced originals and artifacts are valid.")
        entries = catalogue.list_sources()
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


__all__ = ["_catalogue_rows", "render_catalogue_inspection"]
