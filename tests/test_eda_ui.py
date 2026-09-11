import pandas as pd
import pytest

from parse.eda_ui import profile_uploaded_dataset


def test_uploaded_csv_is_profiled_with_upload_provenance():
    content = pd.DataFrame({"id": [1, 2], "target": [3.0, 4.0]}).to_csv(index=False).encode()

    result = profile_uploaded_dataset("sample.csv", content)

    assert result.source.source_id == "upload:sample.csv"
    assert result.source.source_type == "csv"
    assert result.row_count == 2
    assert result.data_modified is False


def test_uploaded_unsupported_file_is_rejected():
    with pytest.raises(ValueError, match="CSV or Excel"):
        profile_uploaded_dataset("sample.json", b"{}")
