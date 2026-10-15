"""Minimal JSON manifest support for PARSE resource registration.

This module intentionally stays narrow: it serializes and validates a traceable
resource manifest, but it does not implement ingestion, retrieval, indexing, or
web crawling. It reuses the existing PARSE SourceRef and Provenance contracts.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from parse.core.contracts import Issue, Provenance, SourceRef

_RESOURCE_CLASSES = {
    "private_operational_data",
    "internal_technical_document",
    "public_research",
    "public_dataset",
    "standards_reference",
    "derived_analysis_result",
}
_CONTENT_TYPES = {
    "document",
    "csv",
    "spreadsheet",
    "json",
    "text",
    "model",
    "dataset",
    "derived_artifact",
    "artifact",
}
_PROCESSING_STATES = {"registered", "normalized", "indexed", "derived", "failed"}


def _require_text(value: Any, field_name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _coerce_source(value: Any) -> SourceRef | None:
    if value is None:
        return None
    if isinstance(value, SourceRef):
        return value
    if isinstance(value, Mapping):
        return SourceRef.from_dict(value)
    raise TypeError("source must be a SourceRef or a mapping")


def _coerce_provenance(value: Any) -> Provenance | None:
    if value is None:
        return None
    if isinstance(value, Provenance):
        return value
    if isinstance(value, Mapping):
        return Provenance.from_dict(value)
    raise TypeError("provenance must be a Provenance or a mapping")


def _as_text_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple, set)):
        raise TypeError(f"{field_name} must be a list of strings")
    entries: list[str] = []
    for entry in value:
        if not isinstance(entry, str):
            raise TypeError(f"{field_name} must contain only strings")
        text = entry.strip()
        if text:
            entries.append(text)
    return entries


def _stable_resource_id(
    *,
    resource_class: str,
    title: str,
    content_type: str,
    source: SourceRef | None,
    source_location: str | None,
    contexts: list[str],
) -> str:
    seed = "|".join(
        [
            resource_class,
            title,
            content_type,
            source.source_id if source is not None else "unknown_source",
            source_location or "",
            ",".join(sorted(contexts)),
        ]
    )
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"{resource_class}-{digest}"


@dataclass
class ResourceManifest:
    """Persisted JSON description of a PARSE resource.

    The manifest is intentionally a thin, JSON-friendly representation of the
    existing SourceRef and Provenance contracts. It does not replace the in-memory
    contracts or the domain-specific record classes.
    """

    resource_id: str
    resource_class: str
    content_type: str
    title: str
    source: SourceRef | None = None
    contexts: list[str] = field(default_factory=list)
    provenance: Provenance | None = None
    processing_state: str = "registered"
    source_location: str | None = None
    access_restrictions: str | None = None
    license: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    normalized_content: str | None = None
    derived_artifacts: list[str] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    created_at: str | None = None
    manifest_version: str = "1.0"

    def __post_init__(self) -> None:
        self.resource_id = _require_text(self.resource_id, "resource_id")
        self.resource_class = _require_text(self.resource_class, "resource_class")
        if self.resource_class not in _RESOURCE_CLASSES:
            raise ValueError(
                f"Unsupported resource_class: {self.resource_class}. "
                f"Allowed values: {sorted(_RESOURCE_CLASSES)}"
            )
        self.content_type = _require_text(self.content_type, "content_type")
        if self.content_type not in _CONTENT_TYPES:
            raise ValueError(
                f"Unsupported content_type: {self.content_type}. "
                f"Allowed values: {sorted(_CONTENT_TYPES)}"
            )
        self.title = _require_text(self.title, "title")
        self.contexts = _as_text_list(self.contexts, "contexts")
        self.derived_artifacts = _as_text_list(self.derived_artifacts, "derived_artifacts")
        self.issues = list(self.issues)
        if self.source is not None and not isinstance(self.source, SourceRef):
            raise TypeError("source must be a SourceRef when supplied")
        if self.provenance is not None and not isinstance(self.provenance, Provenance):
            raise TypeError("provenance must be a Provenance when supplied")
        if any(not isinstance(issue, Issue) for issue in self.issues):
            raise TypeError("issues must contain Issue values")
        if self.metadata is None:
            self.metadata = {}
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a mapping")
        if self.processing_state not in _PROCESSING_STATES:
            raise ValueError(
                f"Unsupported processing_state: {self.processing_state}. "
                f"Allowed values: {sorted(_PROCESSING_STATES)}"
            )
        if self.source_location is not None:
            self.source_location = _require_text(self.source_location, "source_location", allow_empty=True)
        if self.access_restrictions is not None:
            self.access_restrictions = _require_text(self.access_restrictions, "access_restrictions", allow_empty=True)
        if self.license is not None:
            self.license = _require_text(self.license, "license", allow_empty=True)
        if self.normalized_content is not None:
            self.normalized_content = _require_text(self.normalized_content, "normalized_content", allow_empty=True)
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "resource_id": self.resource_id,
            "resource_class": self.resource_class,
            "content_type": self.content_type,
            "title": self.title,
            "source": self.source.to_dict() if self.source is not None else None,
            "contexts": list(self.contexts),
            "provenance": self.provenance.to_dict() if self.provenance is not None else None,
            "processing_state": self.processing_state,
            "source_location": self.source_location,
            "access_restrictions": self.access_restrictions,
            "license": self.license,
            "metadata": dict(self.metadata),
            "normalized_content": self.normalized_content,
            "derived_artifacts": list(self.derived_artifacts),
            "issues": [issue.to_dict() for issue in self.issues],
            "created_at": self.created_at,
            "manifest_version": self.manifest_version,
        }
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResourceManifest":
        """Restore a manifest from its JSON-compatible representation."""
        source = payload.get("source")
        provenance = payload.get("provenance")
        return cls(
            resource_id=payload["resource_id"],
            resource_class=payload["resource_class"],
            content_type=payload["content_type"],
            title=payload["title"],
            source=_coerce_source(source),
            contexts=payload.get("contexts", []),
            provenance=_coerce_provenance(provenance),
            processing_state=payload.get("processing_state", "registered"),
            source_location=payload.get("source_location"),
            access_restrictions=payload.get("access_restrictions"),
            license=payload.get("license"),
            metadata=payload.get("metadata", {}),
            normalized_content=payload.get("normalized_content"),
            derived_artifacts=payload.get("derived_artifacts", []),
            issues=[
                issue if isinstance(issue, Issue) else Issue.from_dict(issue)
                for issue in payload.get("issues", [])
            ],
            created_at=payload.get("created_at"),
            manifest_version=payload.get("manifest_version", "1.0"),
        )

    def write_json(self, destination: str | os.PathLike[str], *, allow_overwrite: bool = False) -> Path:
        path = Path(destination)
        if path.exists() and not allow_overwrite:
            raise FileExistsError(f"Manifest already exists at {path}; set allow_overwrite=True to replace it.")

        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.tmp")
        if temp_path.exists():
            temp_path.unlink()

        try:
            payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
            with temp_path.open("w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.write("\n")
            os.replace(temp_path, path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise
        return path

    @classmethod
    def from_json(cls, path: str | os.PathLike[str]) -> "ResourceManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


def create_resource_manifest(
    metadata: Mapping[str, Any],
    destination: str | os.PathLike[str],
    *,
    allow_overwrite: bool = False,
) -> ResourceManifest:
    """Validate and serialize a manifest for a known resource.

    The manifest supports both explicit metadata and a free-form ``metadata``
    payload for optional or project-specific keys that do not already have a
    formal PARSE contract.
    """

    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")

    reserved_keys = {
        "resource_id",
        "resource_class",
        "content_type",
        "title",
        "source",
        "contexts",
        "context_labels",
        "provenance",
        "processing_state",
        "source_location",
        "access_restrictions",
        "license",
        "normalized_content",
        "derived_artifacts",
        "issues",
        "created_at",
        "metadata",
    }
    extra_metadata = {key: value for key, value in metadata.items() if key not in reserved_keys}
    manifest_metadata: dict[str, Any] = dict(metadata.get("metadata", {}) or {})
    if extra_metadata:
        manifest_metadata.update(extra_metadata)

    resource_class = metadata.get("resource_class")
    if resource_class is None:
        raise ValueError("resource_class is required")
    content_type = metadata.get("content_type")
    if content_type is None:
        raise ValueError("content_type is required")
    title = metadata.get("title")
    if title is None:
        raise ValueError("title is required")

    source = _coerce_source(metadata.get("source"))
    source_location = metadata.get("source_location")
    if source is not None and source_location is None:
        source_location = source.locator

    contexts = metadata.get("contexts")
    if contexts is None:
        contexts = metadata.get("context_labels", [])
    contexts = _as_text_list(contexts, "contexts")

    manifest = ResourceManifest(
        resource_id=str(metadata.get("resource_id") or _stable_resource_id(
            resource_class=str(resource_class),
            title=str(title),
            content_type=str(content_type),
            source=source,
            source_location=source_location,
            contexts=contexts,
        )),
        resource_class=str(resource_class),
        content_type=str(content_type),
        title=str(title),
        source=source,
        contexts=contexts,
        provenance=_coerce_provenance(metadata.get("provenance")),
        processing_state=str(metadata.get("processing_state", "registered")),
        source_location=source_location,
        access_restrictions=metadata.get("access_restrictions"),
        license=metadata.get("license"),
        metadata=manifest_metadata,
        normalized_content=metadata.get("normalized_content"),
        derived_artifacts=metadata.get("derived_artifacts", []),
        issues=[
            issue if isinstance(issue, Issue) else Issue.from_dict(issue)
            for issue in metadata.get("issues", [])
        ],
        created_at=metadata.get("created_at"),
    )
    manifest.write_json(destination, allow_overwrite=allow_overwrite)
    return manifest


__all__ = [
    "ResourceManifest",
    "create_resource_manifest",
]
