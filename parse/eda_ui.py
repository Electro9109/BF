"""Pure helpers for the Streamlit Data Explorer integration."""

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd

from parse.core.contracts import SourceRef
from parse.analysis import AnalysisRequest
from parse.eda import DataUnderstanding, LegacyEDAResult
from parse.semantic_analysis import AnalysisBundle, AnalysisPipeline, RelevanceRequest

_MSG_EMPTY_OR_UNREADABLE = (
    "The file appears to be empty or contains no readable data."
)
_MSG_CORRUPT_OR_UNSUPPORTED = (
    "The file could not be read as a valid CSV/Excel file -- it may be "
    "corrupted or in an unsupported format."
)


def _read_upload_payload(suffix: str, content: bytes) -> pd.DataFrame:
    """Parse CSV/Excel bytes; translate library errors to user-facing messages."""
    try:
        if suffix == ".csv":
            return pd.read_csv(BytesIO(content))
        return pd.read_excel(BytesIO(content), sheet_name=0)
    except pd.errors.EmptyDataError:
        raise ValueError(_MSG_EMPTY_OR_UNREADABLE) from None
    except pd.errors.ParserError:
        raise ValueError(_MSG_CORRUPT_OR_UNSUPPORTED) from None
    except zipfile.BadZipFile:
        raise ValueError(_MSG_CORRUPT_OR_UNSUPPORTED) from None
    except ValueError as exc:
        message = str(exc)
        if "Excel file format" in message or "engine manually" in message:
            raise ValueError(_MSG_CORRUPT_OR_UNSUPPORTED) from None
        raise


def load_uploaded_dataset(filename: str, content: bytes) -> tuple[pd.DataFrame, SourceRef]:
    """Read an uploaded CSV/Excel payload without changing its contents."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        frame = _read_upload_payload(suffix, content)
        source_type = "csv"
    elif suffix in {".xlsx", ".xls"}:
        frame = _read_upload_payload(suffix, content)
        source_type = "spreadsheet"
    else:
        raise ValueError("Upload a CSV or Excel file")

    source = SourceRef(
        source_id=f"upload:{filename}",
        source_type=source_type,
        locator=filename,
        label=filename,
    )
    return frame, source


def profile_uploaded_dataset(filename: str, content: bytes) -> LegacyEDAResult:
    """Read an uploaded CSV/Excel payload and return a read-only EDA result."""
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
    request = AnalysisRequest(frame, source)
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
