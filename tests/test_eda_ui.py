import pandas as pd
import pytest

from parse.eda_ui import load_uploaded_dataset, profile_uploaded_dataset


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


def test_empty_csv_raises_friendly_message_not_raw_pandas_error():
    with pytest.raises(ValueError) as exc_info:
        load_uploaded_dataset("empty.csv", b"")
    message = str(exc_info.value)
    assert "No columns to parse" not in message
    assert "empty or contains no readable data" in message or "could not be read" in message


def test_corrupt_excel_raises_friendly_message():
    with pytest.raises(ValueError, match="could not be read as a valid CSV/Excel"):
        load_uploaded_dataset("broken.xlsx", b"not a zip or excel file")
    with pytest.raises(ValueError) as exc_info:
        load_uploaded_dataset("broken.xlsx", b"not a zip or excel file")
    assert "engine manually" not in str(exc_info.value)


def test_unsupported_extension_message_unchanged():
    with pytest.raises(ValueError, match="^Upload a CSV or Excel file$"):
        load_uploaded_dataset("notes.txt", b"hello")
