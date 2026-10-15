"""Persistent registration and normalization for local PARSE sources."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import zipfile
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.paths import (
    SOURCE_CATALOGUE_DB,
    SOURCE_NORMALIZED_DIR,
    SOURCE_ORIGINALS_DIR,
    SOURCE_PUBLIC_ORIGINALS_DIR,
)
from data.loader import load_and_chunk_data
from parse.core.contracts import Issue, KnowledgeItem, Provenance, SourceRef
from parse.core.resource_manifest import ResourceManifest
from parse.eda_ui import load_uploaded_dataset

_SOURCE_TYPES = {
    ".txt": ("document", "internal_technical_document"),
    ".csv": ("csv", "private_operational_data"),
    ".xlsx": ("spreadsheet", "private_operational_data"),
    ".xls": ("spreadsheet", "private_operational_data"),
}
_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass
class CatalogueEntry:
    manifest: ResourceManifest
    source_locations: list[str]
    artifacts: list[KnowledgeItem]


class SourceIngestionError(RuntimeError):
    """A registered source could not be normalized."""

    def __init__(self, message: str, entry: CatalogueEntry):
        super().__init__(message)
        self.entry = entry


class SourceCategoryConflict(ValueError):
    """Identical content is already registered under a different category."""


class SourceCatalogue:
    """SQLite metadata registry with immutable originals and derived artifacts."""

    def __init__(
        self,
        database_path: str | os.PathLike[str] = SOURCE_CATALOGUE_DB,
        *,
        originals_dir: str | os.PathLike[str] = SOURCE_ORIGINALS_DIR,
        public_originals_dir: str | os.PathLike[str] = SOURCE_PUBLIC_ORIGINALS_DIR,
        normalized_dir: str | os.PathLike[str] = SOURCE_NORMALIZED_DIR,
    ) -> None:
        self.database_path = Path(database_path)
        self.originals_dir = Path(originals_dir)
        self.public_originals_dir = Path(public_originals_dir)
        self.normalized_dir = Path(normalized_dir)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.originals_dir.mkdir(parents=True, exist_ok=True)
        self.public_originals_dir.mkdir(parents=True, exist_ok=True)
        self.normalized_dir.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS resources (
                    resource_id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    manifest_json TEXT NOT NULL,
                    UNIQUE(source_type, content_hash)
                );
                CREATE TABLE IF NOT EXISTS source_locations (
                    resource_id TEXT NOT NULL REFERENCES resources(resource_id),
                    location TEXT NOT NULL,
                    PRIMARY KEY(resource_id, location)
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    resource_id TEXT NOT NULL REFERENCES resources(resource_id),
                    item_json TEXT NOT NULL,
                    UNIQUE(resource_id, artifact_id)
                );
                """
            )

    @staticmethod
    def _safe_filename(filename: str) -> str:
        if not isinstance(filename, str):
            raise TypeError("filename must be a string")
        basename = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
        safe = _INVALID_FILENAME_CHARS.sub("_", basename).rstrip(". ")
        if not safe or safe in {".", ".."}:
            raise ValueError("A non-empty source filename is required")
        return safe

    @staticmethod
    def _source_type(filename: str) -> tuple[str, str, str]:
        extension = Path(filename).suffix.lower()
        try:
            source_type, resource_class = _SOURCE_TYPES[extension]
        except KeyError as exc:
            raise ValueError(
                "Supported source files are .txt, .csv, .xlsx, and .xls"
            ) from exc
        content_type = {
            ".txt": "document",
            ".csv": "csv",
            ".xlsx": "spreadsheet",
            ".xls": "spreadsheet",
        }[extension]
        return source_type, resource_class, content_type

    def _originals_root(self, resource_class: str) -> tuple[str, Path]:
        if resource_class in {"public_research", "public_dataset", "standards_reference"}:
            return "public", self.public_originals_dir
        return "private", self.originals_dir

    def register_file(
        self,
        source_path: str | os.PathLike[str],
        *,
        title: str | None = None,
        resource_class: str | None = None,
    ) -> CatalogueEntry:
        path = Path(source_path)
        if not path.is_file():
            raise FileNotFoundError(f"Source file does not exist: {path}")
        return self.register_upload(
            path.name,
            path.read_bytes(),
            title=title,
            resource_class=resource_class,
            source_location=str(path.resolve()),
        )

    def register_upload(
        self,
        filename: str,
        content: bytes,
        *,
        title: str | None = None,
        resource_class: str | None = None,
        source_location: str | None = None,
    ) -> CatalogueEntry:
        if not isinstance(content, bytes):
            raise TypeError("content must be bytes")
        safe_filename = self._safe_filename(filename)
        source_type, default_resource_class, content_type = self._source_type(safe_filename)
        resource_class = resource_class or default_resource_class
        location = source_location or f"upload:{safe_filename}"
        content_hash = hashlib.sha256(content).hexdigest()
        identity_digest = hashlib.sha256(
            source_type.encode("utf-8") + b"\0" + content
        ).hexdigest()
        resource_id = f"resource-{identity_digest}"
        source_id = f"sha256:{identity_digest}"
        self._initialize()
        original_scope, original_root = self._originals_root(resource_class)
        source = SourceRef(
            source_id=source_id,
            source_type=source_type,
            locator=location,
            label=title or safe_filename,
        )
        candidate_manifest = ResourceManifest(
            resource_id=resource_id,
            resource_class=resource_class,
            content_type=content_type,
            title=title or safe_filename,
            source=source,
            provenance=Provenance(
                sources=[source],
                operation="source_registration",
                method="parse.source_catalogue",
            ),
            processing_state="registered",
            source_location=location,
            metadata={
                "content_sha256": content_hash,
                "byte_size": len(content),
                "original_filename": safe_filename,
                "original_storage_scope": original_scope,
                "preserved_original": str(Path(resource_id) / safe_filename),
            },
        )
        original_path = original_root / resource_id / safe_filename

        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT manifest_json FROM resources WHERE source_type = ? AND content_hash = ?",
                (source_type, content_hash),
            ).fetchone()
            if row is not None:
                existing_manifest = ResourceManifest.from_dict(
                    json.loads(row["manifest_json"])
                )
                if existing_manifest.resource_class != resource_class:
                    raise SourceCategoryConflict(
                        "Identical source content is already registered as "
                        f"{existing_manifest.resource_class}; it cannot also be "
                        f"registered as {resource_class}."
                    )
            if row is None:
                manifest = candidate_manifest
                self._write_immutable(original_path, content)
                connection.execute(
                    "INSERT INTO resources(resource_id, source_type, content_hash, manifest_json) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        resource_id,
                        source_type,
                        content_hash,
                        json.dumps(manifest.to_dict(), ensure_ascii=False),
                    ),
                )
            else:
                manifest = ResourceManifest.from_dict(json.loads(row["manifest_json"]))
                if manifest.resource_class != resource_class:
                    raise SourceCategoryConflict(
                        f"This content is already registered as "
                        f"{manifest.resource_class!r}, not {resource_class!r}."
                    )
                original_name = str(
                    manifest.metadata.get("original_filename", safe_filename)
                )
                stored_scope = str(
                    manifest.metadata.get("original_storage_scope", "private")
                )
                _, stored_root = self._originals_root(
                    "public_dataset" if stored_scope == "public" else "private_operational_data"
                )
                self._write_immutable(
                    stored_root / resource_id / original_name,
                    content,
                )
            connection.execute(
                "INSERT OR IGNORE INTO source_locations(resource_id, location) VALUES (?, ?)",
                (resource_id, location),
            )

        entry = self.get_source(resource_id)
        if entry is None:
            raise RuntimeError(f"Registered source could not be read back: {resource_id}")
        if entry.manifest.processing_state in {"normalized", "indexed"} and entry.artifacts and all(
            (self._artifact_path(item)).is_file() for item in entry.artifacts
        ):
            return entry

        manifest_source = entry.manifest.source
        if manifest_source is None:
            raise RuntimeError(f"Registered source has no SourceRef: {resource_id}")
        try:
            artifact, artifact_path, metadata, issues = self._normalize(
                str(entry.manifest.metadata.get("original_filename", safe_filename)),
                content,
                manifest_source,
                resource_id,
                self._originals_root(
                    "public_dataset"
                    if entry.manifest.metadata.get("original_storage_scope") == "public"
                    else "private_operational_data"
                )[1],
            )
        except (OSError, UnicodeError, ValueError, ImportError, zipfile.BadZipFile) as exc:
            issue = Issue(
                code="source_normalization_failed",
                severity="error",
                message=str(exc),
                source=source,
            )
            self._save_manifest(
                resource_id,
                entry.manifest,
                processing_state="failed",
                issues=[issue],
            )
            failed = self.get_source(resource_id)
            if failed is None:
                raise RuntimeError(f"Could not retrieve failed source {resource_id}") from exc
            raise SourceIngestionError(
                f"Source was registered but could not be normalized: {exc}",
                failed,
            ) from exc

        relative_artifact = str(artifact_path.relative_to(self.normalized_dir))
        self._save_manifest(
            resource_id,
            entry.manifest,
            processing_state="normalized",
            derived_artifacts=[relative_artifact],
            metadata=metadata,
            issues=issues,
        )
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT OR REPLACE INTO artifacts(artifact_id, resource_id, item_json) "
                "VALUES (?, ?, ?)",
                (
                    artifact.item_id,
                    resource_id,
                    json.dumps(artifact.to_dict(), ensure_ascii=False),
                ),
            )
        completed = self.get_source(resource_id)
        if completed is None:
            raise RuntimeError(f"Normalized source could not be read back: {resource_id}")
        return completed

    def register_confirmed_upload(
        self,
        filename: str,
        original_content: bytes,
        *,
        final_content: bytes | None = None,
        title: str | None = None,
        resource_class: str | None = None,
        source_location: str | None = None,
        contexts: list[str] | None = None,
        workflow_metadata: dict[str, Any] | None = None,
    ) -> CatalogueEntry:
        """Register approved source bytes and the chosen normalized artifact.

        ``original_content`` determines source identity and is always retained.
        For tabular inputs, ``final_content`` is the normalized CSV selected by
        the reviewer; omitted content falls back to the unchanged source.
        """
        safe_filename = self._safe_filename(filename)
        workflow_metadata = json.loads(
            json.dumps(workflow_metadata or {}, ensure_ascii=False, default=str)
        )
        source_type, _default_class, _content_type = self._source_type(safe_filename)
        if source_type == "document" and final_content is not None:
            raise ValueError("Text document artifacts are produced by normalization")
        if source_type in {"csv", "spreadsheet"} and final_content is None:
            from parse.eda_ui import load_uploaded_dataset

            frame, _source_ref = load_uploaded_dataset(safe_filename, original_content)
            final_content = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
        if source_type in {"csv", "spreadsheet"}:
            if not isinstance(final_content, bytes):
                raise TypeError("final_content must be bytes")
            from parse.eda_ui import load_uploaded_dataset

            load_uploaded_dataset("selected-artifact.csv", final_content)

        entry = self.register_upload(
            safe_filename,
            original_content,
            title=title,
            resource_class=resource_class,
            source_location=source_location,
        )
        if source_type == "document":
            self._update_manifest_metadata(
                entry.manifest.resource_id,
                contexts=contexts or [],
                workflow_metadata=workflow_metadata or {},
                final_content_sha256=entry.manifest.metadata["content_sha256"],
                processing_stage="normalized",
            )
            return self.get_source(entry.manifest.resource_id) or entry

        if final_content is None:
            raise RuntimeError("Tabular normalization produced no final content")
        from parse.eda_ui import load_uploaded_dataset

        final_frame, _ = load_uploaded_dataset("selected-artifact.csv", final_content)
        final_hash = hashlib.sha256(final_content).hexdigest()
        relative_path = Path("datasets") / f"{entry.manifest.resource_id}-{final_hash[:16]}.csv"
        artifact_path = self.normalized_dir / relative_path
        self._write_immutable(artifact_path, final_content)
        artifact_id = f"artifact-{entry.manifest.resource_id}-{final_hash[:16]}"
        missing_counts = {
            str(column): int(count)
            for column, count in final_frame.isna().sum().items()
            if count
        }
        issues = [
            Issue(
                code="missing_values_present",
                severity="warning",
                message=f"{count} missing value(s) remain in column {column!r}.",
                field=column,
                source=entry.manifest.source,
            )
            for column, count in missing_counts.items()
        ]
        artifact = KnowledgeItem(
            item_id=artifact_id,
            kind="structured_record",
            attributes={
                "artifact_type": "normalized_dataset",
                "path": str(relative_path),
                "content_sha256": final_hash,
                "row_count": len(final_frame.index),
                "column_count": len(final_frame.columns),
                "columns": [str(column) for column in final_frame.columns],
                "missing_value_counts": missing_counts,
            },
            provenance=Provenance(
                sources=[entry.manifest.source],
                operation="register_confirmed_dataset",
                method="parse.source_catalogue",
                parent_ids=[
                    entry.manifest.source.source_id,
                    *(
                        str(item)
                        for item in (workflow_metadata or {}).get(
                            "approved_transformations", []
                        )
                    ),
                ],
            ),
            issues=issues,
            incomplete=bool(missing_counts),
        )
        self._store_artifact(entry.manifest.resource_id, artifact)
        self._update_manifest_metadata(
            entry.manifest.resource_id,
            contexts=contexts or [],
            workflow_metadata=workflow_metadata or {},
            final_content_sha256=final_hash,
            selected_artifact=str(relative_path),
            processing_state="normalized",
            processing_stage="normalized",
            issues=issues,
        )
        result = self.get_source(entry.manifest.resource_id)
        if result is None:
            raise RuntimeError(
                f"Registered source could not be read back: {entry.manifest.resource_id}"
            )
        return result

    def mark_indexed(self, resource_id: str) -> CatalogueEntry:
        """Record successful document inclusion in the active retrieval index."""
        self._set_processing_state(resource_id, "indexed", None)
        entry = self.get_source(resource_id)
        if entry is None:
            raise KeyError(f"Unknown source resource: {resource_id}")
        return entry

    def mark_index_failed(self, resource_id: str, message: str) -> CatalogueEntry:
        """Record a failed retrieval-index refresh without claiming success."""
        entry = self.get_source(resource_id)
        if entry is None:
            raise KeyError(f"Unknown source resource: {resource_id}")
        issue = Issue(
            "retrieval_index_failed",
            "error",
            message,
            source=entry.manifest.source,
        )
        self._set_processing_state(resource_id, "failed", issue)
        result = self.get_source(resource_id)
        if result is None:
            raise RuntimeError(f"Failed source could not be read back: {resource_id}")
        return result

    def _set_processing_state(
        self,
        resource_id: str,
        state: str,
        issue: Issue | None,
    ) -> None:
        entry = self.get_source(resource_id)
        if entry is None:
            raise KeyError(f"Unknown source resource: {resource_id}")
        manifest = entry.manifest
        manifest.processing_state = state
        if issue is not None:
            manifest.issues = [
                existing
                for existing in manifest.issues
                if existing.code != issue.code
            ] + [issue]
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "UPDATE resources SET manifest_json = ? WHERE resource_id = ?",
                (json.dumps(manifest.to_dict(), ensure_ascii=False), resource_id),
            )

    def _store_artifact(
        self,
        resource_id: str,
        artifact: KnowledgeItem,
    ) -> None:
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT manifest_json FROM resources WHERE resource_id = ?",
                (resource_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown source resource: {resource_id}")
            manifest = ResourceManifest.from_dict(json.loads(row["manifest_json"]))
            connection.execute(
                "INSERT OR IGNORE INTO artifacts(artifact_id, resource_id, item_json) "
                "VALUES (?, ?, ?)",
                (
                    artifact.item_id,
                    resource_id,
                    json.dumps(artifact.to_dict(), ensure_ascii=False),
                ),
            )
            relative_path = str(artifact.attributes["path"])
            if relative_path not in manifest.derived_artifacts:
                manifest.derived_artifacts.append(relative_path)
            connection.execute(
                "UPDATE resources SET manifest_json = ? WHERE resource_id = ?",
                (json.dumps(manifest.to_dict(), ensure_ascii=False), resource_id),
            )

    def _update_manifest_metadata(
        self,
        resource_id: str,
        *,
        contexts: list[str] | None = None,
        workflow_metadata: dict[str, Any] | None = None,
        final_content_sha256: str | None = None,
        selected_artifact: str | None = None,
        processing_state: str | None = None,
        processing_stage: str | None = None,
        indexed_chunk_count: int | None = None,
        index_error: str | None = None,
        issues: list[Issue] | None = None,
    ) -> None:
        if workflow_metadata is not None:
            json.dumps(workflow_metadata)
        entry = self.get_source(resource_id)
        if entry is None:
            raise KeyError(f"Unknown source resource: {resource_id}")
        manifest = entry.manifest
        if contexts is not None:
            manifest.contexts = list(contexts)
        if final_content_sha256 is not None:
            manifest.metadata["final_content_sha256"] = final_content_sha256
        if workflow_metadata is not None:
            manifest.metadata["workflow"] = workflow_metadata
        if selected_artifact is not None:
            manifest.metadata["selected_artifact"] = selected_artifact
        if processing_state is not None:
            manifest.processing_state = processing_state
        if processing_stage is not None:
            manifest.metadata["processing_stage"] = processing_stage
        if indexed_chunk_count is not None:
            manifest.metadata["indexed_chunk_count"] = indexed_chunk_count
        if index_error is not None or processing_stage in {"indexed", "registration"}:
            manifest.metadata["index_error"] = index_error
        if issues is not None:
            manifest.issues = issues
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "UPDATE resources SET manifest_json = ? WHERE resource_id = ?",
                (json.dumps(manifest.to_dict(), ensure_ascii=False), resource_id),
            )

    def mark_indexed(self, resource_id: str, chunk_count: int) -> CatalogueEntry:
        if chunk_count < 1:
            raise ValueError("chunk_count must be positive")
        entry = self.get_source(resource_id)
        if entry is None:
            raise KeyError(f"Unknown source resource: {resource_id}")
        self._update_manifest_metadata(
            resource_id,
            processing_state="indexed",
            processing_stage="indexed",
            indexed_chunk_count=chunk_count,
            index_error=None,
            issues=[
                issue
                for issue in entry.manifest.issues
                if issue.code not in {
                    "source_indexing_failed",
                    "source_registration_failed",
                }
            ],
        )
        entry = self.get_source(resource_id)
        if entry is None:
            raise KeyError(f"Unknown source resource: {resource_id}")
        return entry

    def mark_index_failed(self, resource_id: str, error: str) -> CatalogueEntry:
        if not error.strip():
            raise ValueError("error must be a non-empty string")
        entry = self.get_source(resource_id)
        if entry is None:
            raise KeyError(f"Unknown source resource: {resource_id}")
        issue = Issue(
            code="source_indexing_failed",
            severity="error",
            message=error,
            source=entry.manifest.source,
        )
        self._update_manifest_metadata(
            resource_id,
            processing_state="failed",
            processing_stage="indexing",
            index_error=error,
            issues=[
                *[
                    existing
                    for existing in entry.manifest.issues
                    if existing.code not in {
                        "source_indexing_failed",
                        "source_registration_failed",
                    }
                ],
                issue,
            ],
        )
        entry = self.get_source(resource_id)
        if entry is None:
            raise KeyError(f"Unknown source resource: {resource_id}")
        return entry

    def mark_registration_failed(
        self,
        resource_id: str,
        error: str,
    ) -> CatalogueEntry:
        if not error.strip():
            raise ValueError("error must be a non-empty string")
        entry = self.get_source(resource_id)
        if entry is None:
            raise KeyError(f"Unknown source resource: {resource_id}")
        issue = Issue(
            code="source_registration_failed",
            severity="error",
            message=error,
            source=entry.manifest.source,
        )
        self._update_manifest_metadata(
            resource_id,
            processing_state="failed",
            processing_stage="registration",
            index_error=None,
            issues=[
                *[
                    existing
                    for existing in entry.manifest.issues
                    if existing.code not in {
                        "source_registration_failed",
                        "source_indexing_failed",
                    }
                ],
                issue,
            ],
        )
        failed = self.get_source(resource_id)
        if failed is None:
            raise KeyError(f"Unknown source resource: {resource_id}")
        return failed

    @staticmethod
    def _write_immutable(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(content).digest():
                raise FileExistsError(f"Refusing to replace existing source file: {path}")
            return
        temp_path = path.with_name(f".{path.name}.tmp")
        try:
            with temp_path.open("xb") as handle:
                handle.write(content)
            os.replace(temp_path, path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _normalize(
        self,
        filename: str,
        content: bytes,
        source: SourceRef,
        resource_id: str,
        originals_dir: Path,
    ) -> tuple[KnowledgeItem, Path, dict[str, Any], list[Issue]]:
        extension = Path(filename).suffix.lower()
        artifact_id = f"artifact-{resource_id}"
        issues: list[Issue] = []
        if extension == ".txt":
            original_path = originals_dir / resource_id / filename
            chunks = load_and_chunk_data(original_path.parent, strict=True)
            if not chunks:
                raise ValueError("Text source contains no indexable sections")
            normalized_path = self.normalized_dir / "documents" / f"{resource_id}.txt"
            normalized_content = "\n".join(
                [f"TOPIC: {chunks[0].topic}"]
                + [
                    f"\nSECTION: {chunk.section}\n{chunk.content}"
                    for chunk in chunks
                ]
            )
            self._write_immutable(normalized_path, normalized_content.encode("utf-8"))
            attributes = {
                "artifact_type": "normalized_document",
                "path": str(normalized_path.relative_to(self.normalized_dir)),
                "chunk_count": len(chunks),
                "sections": list(dict.fromkeys(chunk.section for chunk in chunks)),
            }
            kind = "document"
        else:
            frame, _ = load_uploaded_dataset(filename, content)
            normalized_path = self.normalized_dir / "datasets" / f"{resource_id}.csv"
            normalized_content = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
            self._write_immutable(normalized_path, normalized_content)
            missing_counts = {
                str(column): int(count)
                for column, count in frame.isna().sum().items()
                if count
            }
            issues.extend(
                Issue(
                    code="missing_values_present",
                    severity="warning",
                    message=f"{count} missing value(s) were retained in column {column!r}.",
                    field=column,
                    source=source,
                )
                for column, count in missing_counts.items()
            )
            attributes = {
                "artifact_type": "normalized_dataset",
                "path": str(normalized_path.relative_to(self.normalized_dir)),
                "row_count": len(frame.index),
                "column_count": len(frame.columns),
                "columns": [str(column) for column in frame.columns],
                "missing_value_counts": missing_counts,
            }
            kind = "structured_record"

        artifact = KnowledgeItem(
            item_id=artifact_id,
            kind=kind,
            attributes=attributes,
            provenance=Provenance(
                sources=[source],
                operation="normalize_source",
                method="parse.source_catalogue",
                parent_ids=[source.source_id],
            ),
        )
        return artifact, normalized_path, attributes, issues

    def _artifact_path(self, item: KnowledgeItem) -> Path:
        return self.normalized_dir / str(item.attributes["path"])

    def _save_manifest(
        self,
        resource_id: str,
        original: ResourceManifest,
        *,
        processing_state: str,
        issues: list[Issue],
        derived_artifacts: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        manifest = ResourceManifest.from_dict(original.to_dict())
        manifest.processing_state = processing_state
        manifest.issues = issues
        if derived_artifacts is not None:
            manifest.derived_artifacts = derived_artifacts
        if metadata is not None:
            manifest.metadata.update(metadata)
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "UPDATE resources SET manifest_json = ? WHERE resource_id = ?",
                (json.dumps(manifest.to_dict(), ensure_ascii=False), resource_id),
            )

    def get_source(self, resource_id: str) -> CatalogueEntry | None:
        if not self.database_path.is_file():
            return None
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT manifest_json FROM resources WHERE resource_id = ?",
                (resource_id,),
            ).fetchone()
            if row is None:
                return None
            locations = connection.execute(
                "SELECT location FROM source_locations WHERE resource_id = ? ORDER BY location",
                (resource_id,),
            ).fetchall()
            artifacts = connection.execute(
                "SELECT item_json FROM artifacts WHERE resource_id = ? ORDER BY artifact_id",
                (resource_id,),
            ).fetchall()
        return CatalogueEntry(
            manifest=ResourceManifest.from_dict(json.loads(row["manifest_json"])),
            source_locations=[item["location"] for item in locations],
            artifacts=[
                KnowledgeItem.from_dict(json.loads(item["item_json"]))
                for item in artifacts
            ],
        )

    def list_sources(self) -> list[CatalogueEntry]:
        if not self.database_path.is_file():
            return []
        with closing(self._connect()) as connection, connection:
            resource_ids = connection.execute(
                "SELECT resource_id FROM resources ORDER BY rowid DESC"
            ).fetchall()
        entries = [self.get_source(row["resource_id"]) for row in resource_ids]
        return [entry for entry in entries if entry is not None]

    @property
    def normalized_documents_dir(self) -> Path:
        return self.normalized_dir / "documents"


__all__ = [
    "CatalogueEntry",
    "SourceCatalogue",
    "SourceCategoryConflict",
    "SourceIngestionError",
]
