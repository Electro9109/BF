"""Small, content-bound state helpers for the Data Ingestion workflow."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, MutableMapping

from data.loader import decode_document_bytes
from data.parser import extract_topic, split_into_sections, strip_topic_header
from data.schemas import Chunk
from parse.core.contracts import SourceRef
from parse.eda_ui import MAX_SOURCE_BYTES

_SOURCE_TYPES = {
    ".txt": "document",
    ".csv": "csv",
    ".xlsx": "spreadsheet",
    ".xls": "spreadsheet",
    ".json": "json",
    ".jsonl": "json",
}
_WORKFLOW_INVALIDATION_KEYS = (
    "eda_result",
    "eda_frame",
    "analysis_bundle",
    "analysis_context",
    "cleaning_result",
    "eda_working_dataset",
    "explainer_result",
    "explainer_finding_key",
    "explainer_recommendation",
    "analysis_objective",
)


@dataclass(frozen=True)
class StagedUpload:
    filename: str
    content: bytes
    content_sha256: str
    source: SourceRef
    display_name: str
    resource_class: str
    contexts: tuple[str, ...] = ()


def create_staged_upload(
    filename: str,
    content: bytes,
    *,
    display_name: str | None = None,
    resource_class: str,
    contexts: tuple[str, ...] = (),
) -> StagedUpload:
    if not isinstance(content, bytes):
        raise TypeError("content must be bytes")
    if len(content) > MAX_SOURCE_BYTES:
        raise ValueError(
            f"Source files must be no larger than {MAX_SOURCE_BYTES // (1024 * 1024)} MiB."
        )
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("filename must be a non-empty string")
    source_type = _SOURCE_TYPES.get(Path(filename).suffix.lower())
    if source_type is None:
        raise ValueError(
            "Supported source files are .txt, .csv, .xlsx, .xls, .json, and .jsonl"
        )
    content_hash = hashlib.sha256(content).hexdigest()
    source_id = hashlib.sha256(source_type.encode("utf-8") + b"\0" + content).hexdigest()
    label = (display_name or filename).strip()
    if not label:
        raise ValueError("display_name must be a non-empty string")
    return StagedUpload(
        filename=filename,
        content=content,
        content_sha256=content_hash,
        source=SourceRef(
            source_id=f"sha256:{source_id}",
            source_type=source_type,
            locator=f"upload:{filename}",
            label=label,
        ),
        display_name=label,
        resource_class=resource_class,
        contexts=tuple(contexts),
    )


def set_staged_upload(
    state: MutableMapping[str, Any],
    staged: StagedUpload,
) -> bool:
    """Bind analysis state to an upload hash, clearing stale results on change."""
    previous = state.get("ingestion_staged_upload")
    metadata_changed = previous is not None and (
        previous.filename != staged.filename
        or previous.source.label != staged.source.label
        or previous.resource_class != staged.resource_class
        or previous.contexts != staged.contexts
        or previous.source.locator != staged.source.locator
    )
    changed = (
        previous is None
        or previous.content_sha256 != staged.content_sha256
        or previous.source.source_type != staged.source.source_type
    )
    if changed:
        for key in _WORKFLOW_INVALIDATION_KEYS:
            state[key] = None if key in {"eda_result", "eda_frame", "analysis_bundle", "analysis_context", "cleaning_result", "explainer_result"} else ""
        state["eda_working_dataset"] = "original_upload"
        state["cleaning_approved"] = []
        state["cleaning_rejected"] = []
        state["ingestion_analysis_hash"] = None
        state["ingestion_review_hash"] = None
        state["ingestion_registration"] = None
        state["ingestion_document_chunks"] = None
        state["ingestion_analysis_objective"] = None
        state["ingestion_stage"] = "staged"
    elif metadata_changed:
        state["ingestion_registration"] = None
        if state.get("ingestion_analysis_hash") == staged.content_sha256:
            state["ingestion_stage"] = "analysed"
    state["ingestion_staged_upload"] = staged
    return changed


def mark_ingestion_stage(
    state: MutableMapping[str, Any],
    stage: str,
    content_sha256: str,
) -> None:
    staged = state.get("ingestion_staged_upload")
    if staged is None or staged.content_sha256 != content_sha256:
        raise ValueError("Cannot advance a workflow for a different staged upload")
    state["ingestion_stage"] = stage


def normalize_staged_document(staged: StagedUpload) -> list[Chunk]:
    """Parse staged text in memory using the same parser as document loading."""
    if staged.source.source_type != "document":
        raise ValueError("Only text documents can be parsed as retrieval chunks")
    text = decode_document_bytes(staged.content)
    topic = extract_topic(text, Path(staged.filename).stem)
    sections = split_into_sections(strip_topic_header(text))
    if not sections:
        raise ValueError("Text source contains no indexable sections")
    rel_source = f"staged:{staged.content_sha256[:16]}/{Path(staged.filename).name}"
    return [
        Chunk(
            content=section["content"],
            embed_text=(
                f"TOPIC: {topic}\nSECTION: {section['section']}\n"
                f"{section['content']}"
            ),
            source=rel_source,
            topic=topic,
            section=section["section"],
        )
        for section in sections
    ]


__all__ = [
    "StagedUpload",
    "create_staged_upload",
    "mark_ingestion_stage",
    "normalize_staged_document",
    "set_staged_upload",
]
