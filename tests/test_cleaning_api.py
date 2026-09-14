import pandas as pd
from parse.cleaning_api import validate_cleaning

def test_validate_cleaning_detects_newly_introduced_duplicate_row():
    original = pd.DataFrame({"x": [10.0, None, 10.0], "y": ["A", "A", "B"]})
    cleaned = pd.DataFrame({"x": [10.0, 10.0, 10.0], "y": ["A", "A", "B"]})
    result = validate_cleaning(original, cleaned)
    assert len(result.newly_introduced_issues) >= 1
    assert any("duplicate_rows" in issue_msg for issue_msg in result.newly_introduced_issues)
