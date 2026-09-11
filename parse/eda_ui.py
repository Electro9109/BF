"""Pure helpers for the Streamlit Data Explorer integration."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd

from parse.core.contracts import SourceRef
from parse.analysis import AnalysisRequest
from parse.eda import DataUnderstanding, EDAResult
from parse.semantic_analysis import AnalysisBundle, AnalysisPipeline, RelevanceRequest


def load_uploaded_dataset(filename: str, content: bytes) -> tuple[pd.DataFrame, SourceRef]:
    """Read an uploaded CSV/Excel payload without changing its contents."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(BytesIO(content))
        source_type = "csv"
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(BytesIO(content), sheet_name=0)
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


def profile_uploaded_dataset(filename: str, content: bytes) -> EDAResult:
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
