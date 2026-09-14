"""Regression: Data Explorer must use parse.analysis.EDAResult attribute paths."""

from pathlib import Path

import pandas as pd

from parse.core.contracts import SourceRef
from parse.eda_ui import analyze_loaded_dataset

APP_WEB_PATH = Path(__file__).resolve().parent.parent / "app_web.py"

# Legacy LegacyEDAResult / Finding fields that must not appear on eda_result in app_web.
_FORBIDDEN_EDA_ACCESSES = (
    "eda_result.row_count",
    "eda_result.column_count",
    "eda_result.next_actions",
    "eda_result.columns",
    "finding.kind",
    "finding.message",
)

_REQUIRED_EDA_ACCESSES = (
    "eda_result.dataset_profile.row_count",
    "eda_result.dataset_profile.column_count",
    "eda_result.attributes",
    "finding.knowledge_state",
    "finding.observation",
)


def test_app_web_does_not_use_legacy_eda_attribute_paths():
    source = APP_WEB_PATH.read_text(encoding="utf-8")
    assert "from parse.eda import DataUnderstanding" not in source
    for pattern in _FORBIDDEN_EDA_ACCESSES:
        assert pattern not in source, f"app_web.py still uses legacy access: {pattern!r}"
    for pattern in _REQUIRED_EDA_ACCESSES:
        assert pattern in source, f"app_web.py missing modern access: {pattern!r}"


def test_app_web_data_explorer_shape_contract():
    """Same attribute paths the Data Explorer tab reads on bundle.eda at runtime."""
    frame = pd.DataFrame({"col1": [1, 2, 3], "col2": ["A", "B", "A"]})
    source = SourceRef("test-id", "csv", "test.csv")
    bundle = analyze_loaded_dataset(frame, source, None)

    eda_result = bundle.eda

    _ = eda_result.dataset_profile.row_count
    _ = eda_result.dataset_profile.column_count
    _ = len(eda_result.findings)
    _ = eda_result.data_modified
    _ = eda_result.summary()
    _ = eda_result.attributes
    _ = eda_result.limitations
    _ = eda_result.to_dict()

    for attribute in eda_result.attributes:
        _ = attribute.to_dict()

    for finding in eda_result.findings:
        _ = finding.category
        _ = finding.knowledge_state
        _ = finding.observation
        _ = finding.limitations

    for finding in bundle.eda.findings:
        _ = finding.to_dict()
