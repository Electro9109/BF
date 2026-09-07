import pandas as pd
import pytest

from data.experiments_loader import ExperimentValidationWarning, load_experiments_df
from data.loader import load_and_chunk_data
from data.schemas import Chunk
from data.sinter_schemas import ExperimentRow


def test_document_loader_returns_generic_chunks():
    chunks = load_and_chunk_data("docs")

    assert chunks
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(chunk.source and chunk.content for chunk in chunks)


def test_document_loader_strict_mode_reports_unreadable_file(tmp_path):
    document = tmp_path / "broken.txt"
    document.write_bytes(b"\xff\xfe")

    with pytest.raises(ValueError, match="Could not read document"):
        load_and_chunk_data(tmp_path, strict=True)


def test_experiment_loader_repairs_derived_temperature(tmp_path):
    source = tmp_path / "experiments.csv"
    pd.DataFrame(
        [{
            "Sinter": 0.7,
            "Ore": 0.3,
            "Pellet": 0.0,
            "T Fe %": 57.0,
            "FeO %": 7.0,
            "SiO2 %": 4.0,
            "CaO %": 7.0,
            "Al2O3 %": 2.5,
            "MgO%": 2.0,
            "Basicity": 1.7,
            "Ts": 1300,
            "Td": 1400,
            "Tm": 1500,
            "Tm-Ts": 999,
            "DelP": 3.0,
            "NDM": 12.0,
        }]
    ).to_csv(source, index=False)

    result = load_experiments_df(source)

    assert result.loc[0, "Tm_Ts"] == 200


def test_existing_experiment_gaps_are_reported_and_retained():
    with pytest.warns(ExperimentValidationWarning, match="missing value"):
        result = load_experiments_df()

    assert len(result) == 131
    assert result["row_index"].iloc[0] == 1
    assert ExperimentRow.__module__ == "data.sinter_schemas"