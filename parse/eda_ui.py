"""Pure helpers for the Streamlit Data Explorer integration."""

from __future__ import annotations

import csv
import json
import zipfile
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from parse.core.contracts import SourceRef
from parse.analysis import AnalysisRequest
from parse.eda import DataUnderstanding, LegacyEDAResult
from parse.semantic_analysis import AnalysisBundle, AnalysisPipeline, RelevanceRequest

MAX_SOURCE_BYTES = 100 * 1024 * 1024

_MSG_EMPTY_OR_UNREADABLE = (
    "The file appears to be empty or contains no readable data."
)
_MSG_CORRUPT_OR_UNSUPPORTED = (
    "The file could not be read as a valid CSV/Excel/JSON file -- it may be "
    "corrupted or in an unsupported format."
)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for key, value in pairs:
        if key in record:
            raise ValueError(f"Duplicate JSON key {key!r} is not supported.")
        record[key] = value
    return record


def _read_upload_payload(suffix: str, content: bytes) -> pd.DataFrame:
    """Parse supported structured data bytes without flattening nested JSON."""
    try:
        if suffix == ".csv":
            header_reader = csv.reader(StringIO(content.decode("utf-8-sig")))
            try:
                header = next(header_reader)
            except StopIteration:
                raise ValueError(_MSG_EMPTY_OR_UNREADABLE) from None
            if len(header) != len(set(header)):
                raise ValueError("Duplicate column names are not supported.")
            frame = pd.read_csv(BytesIO(content))
        elif suffix in {".xlsx", ".xls"}:
            frame = pd.read_excel(BytesIO(content), sheet_name=0)
        elif suffix == ".json":
            records = _read_json_records(content)
            frame = pd.DataFrame.from_records(records)
        elif suffix == ".jsonl":
            records = [
                _validate_json_record(
                    json.loads(line, object_pairs_hook=_reject_duplicate_json_keys)
                )
                for line in content.decode("utf-8-sig").splitlines()
                if line.strip()
            ]
            frame = pd.DataFrame.from_records(records)
        else:
            raise ValueError("Upload a CSV, Excel, JSON, or JSON Lines file")
    except pd.errors.EmptyDataError:
        raise ValueError(_MSG_EMPTY_OR_UNREADABLE) from None
    except pd.errors.ParserError:
        raise ValueError(_MSG_CORRUPT_OR_UNSUPPORTED) from None
    except zipfile.BadZipFile:
        raise ValueError(_MSG_CORRUPT_OR_UNSUPPORTED) from None
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"The structured file is malformed: {exc}") from None
    except ValueError as exc:
        message = str(exc)
        if "Excel file format" in message or "engine manually" in message:
            raise ValueError(_MSG_CORRUPT_OR_UNSUPPORTED) from None
        raise
    if frame.empty:
        raise ValueError(_MSG_EMPTY_OR_UNREADABLE)
    if not frame.columns.is_unique:
        raise ValueError("Duplicate column names are not supported.")
    return frame


def _validate_json_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("JSON datasets must contain records represented as objects.")
    if any(isinstance(item, (dict, list)) for item in value.values()):
        raise ValueError("Nested JSON values are not supported in tabular datasets.")
    return value


def _read_json_records(content: bytes) -> list[dict[str, Any]]:
    payload = json.loads(
        content.decode("utf-8-sig"),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(payload, list) or not payload:
        raise ValueError("JSON datasets must be a non-empty array of record objects.")
    return [_validate_json_record(record) for record in payload]


def load_uploaded_dataset(filename: str, content: bytes) -> tuple[pd.DataFrame, SourceRef]:
    """Read an uploaded supported table into a frame and attach its source type."""
    suffix = Path(filename).suffix.lower()
    if len(content) > MAX_SOURCE_BYTES:
        raise ValueError(
            f"Source files must be no larger than {MAX_SOURCE_BYTES // (1024 * 1024)} MiB."
        )
    if suffix == ".csv":
        frame = _read_upload_payload(suffix, content)
        source_type = "csv"
    elif suffix in {".xlsx", ".xls"}:
        frame = _read_upload_payload(suffix, content)
        source_type = "spreadsheet"
    elif suffix in {".json", ".jsonl"}:
        frame = _read_upload_payload(suffix, content)
        source_type = "json"
    else:
        raise ValueError("Upload a CSV, Excel, JSON, or JSON Lines file")

    source = SourceRef(
        source_id=f"upload:{filename}",
        source_type=source_type,
        locator=filename,
        label=filename,
    )
    return frame, source


def profile_uploaded_dataset(filename: str, content: bytes) -> LegacyEDAResult:
    """Read an uploaded CSV/Excel payload and return a read-only EDA result.
    
    TODO(Task 27): Migrate return type to canonical EDAResult once downstream callers
    expecting LegacyEDAResult (such as tests/test_eda_ui.py) are updated in PARSE 2.0.
    Currently retained for backward compatibility with existing tests.
    """
    frame, source = load_uploaded_dataset(filename, content)
    return DataUnderstanding(source).profile(frame)


def analyze_uploaded_dataset(
    filename: str,
    content: bytes,
    objective: str | None = None,
    selected_attributes: tuple[str, ...] = (),
) -> AnalysisBundle:
    """Run complete structured, semantic, and optional task-relevance analysis."""
    frame, source = load_uploaded_dataset(filename, content)
    return analyze_loaded_dataset(frame, source, objective, selected_attributes)


def analyze_loaded_dataset(
    frame: pd.DataFrame,
    source: SourceRef,
    objective: str | None = None,
    selected_attributes: tuple[str, ...] = (),
) -> AnalysisBundle:
    """Analyze an already-loaded frame without reading or copying the upload again."""
    request = AnalysisRequest(frame, source)
    relevance = RelevanceRequest(objective, selected_attributes) if objective else None
    return AnalysisPipeline().analyze(request, relevance)
