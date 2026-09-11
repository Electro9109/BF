import pytest
import pandas as pd
import numpy as np
from parse.core.contracts import SourceRef
from parse.analysis import AnalysisRequest
from parse.semantic_analysis import AnalysisPipeline
from parse.cleaning_api import create_cleaning_context, propose_cleaning


def _analyze_and_propose(frame, purpose="descriptive_analysis"):
    source = SourceRef("controlled_dataset", "csv")
    request = AnalysisRequest(frame, source)
    bundle = AnalysisPipeline().analyze(request)
    context = create_cleaning_context(bundle, purpose=purpose)
    return propose_cleaning(context, frame)


def test_clean_numeric():
    frame = pd.DataFrame({
        "a": np.random.normal(0, 1, 100),
        "b": np.random.uniform(0, 10, 100),
    })
    issues, proposals = _analyze_and_propose(frame)
    # Shouldn't find missing or duplicates, maybe outliers but we used normal/uniform
    assert not any(i.kind == "missing_values" for i in issues)
    assert not any(i.kind == "duplicate_rows" for i in issues)


def test_missing_values_various_kinds():
    frame = pd.DataFrame({
        "numeric": [1.0, np.nan, 3.0, 4.0, 5.0],
        "categorical": ["A", "B", None, "D", "E"],
        "text_blank": ["hello", "", "world", " ", "foo"], # spaces are empty, but pandas doesn't naturally na them
    })
    issues, proposals = _analyze_and_propose(frame, purpose="prediction")
    
    missing_issues = [i for i in issues if i.kind == "missing_values"]
    assert len(missing_issues) == 2 # numeric and categorical
    
    impute_props = [p for p in proposals if p.action == "impute_missing"]
    assert len(impute_props) == 2


def test_mixed_numeric_text():
    frame = pd.DataFrame({
        "mixed": [1, 2, "three", 4, "5", np.nan]
    })
    issues, proposals = _analyze_and_propose(frame)
    
    # We should have missingness for nan
    assert any(i.kind == "missing_values" for i in issues)


def test_exact_duplicates():
    frame = pd.DataFrame({
        "id": [1, 2, 3, 3, 4],
        "val": ["A", "B", "C", "C", "D"]
    })
    issues, proposals = _analyze_and_propose(frame)
    
    dupe_issue = next(i for i in issues if i.kind == "duplicate_rows")
    assert dupe_issue.duplicate_kind == "exact"
    assert len(dupe_issue.row_indices) > 0
    
    dupe_prop = next(p for p in proposals if p.action == "remove_duplicates")
    assert len(dupe_prop.affected_records) > 0


def test_known_outlier():
    frame = pd.DataFrame({
        "val": [1, 2, 3, 4, 5, 1000]
    })
    issues, proposals = _analyze_and_propose(frame)
    
    outlier_issues = [i for i in issues if i.kind == "statistical_outliers"]
    assert len(outlier_issues) == 1
    
    # Check that we propose to investigate, not delete
    flag_props = [p for p in proposals if p.action == "flag_for_review"]
    assert len(flag_props) == 1


def test_ambiguous_semantic_attribute():
    frame = pd.DataFrame({
        "temp": [100, 200, 300, 400], # could be temperature or something else
    })
    # Just checking it doesn't crash and returns valid lists
    issues, proposals = _analyze_and_propose(frame)
    assert isinstance(issues, list)
    assert isinstance(proposals, list)
