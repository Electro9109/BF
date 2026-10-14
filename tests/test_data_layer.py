import pandas as pd
import pytest
import json

from pathlib import Path

from departments.blast_furnace.experiments_loader import ExperimentValidationWarning, load_experiments_df
from data.loader import load_and_chunk_data
from data.schemas import Chunk
from departments.blast_furnace.bf_experiment_schema import ExperimentRow
from config.paths import EXPERIMENTS_CSV


def test_document_loader_returns_generic_chunks():
    chunks = load_and_chunk_data("docs")

    assert chunks
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(chunk.source and chunk.content for chunk in chunks)


def test_retrieval_eval_labels_reference_existing_corpus_sections():
    chunks = load_and_chunk_data("docs")
    available = {(chunk.source.casefold(), chunk.section.casefold()) for chunk in chunks}
    eval_set = json.loads(
        Path("eval/retrieval_eval_set.json").read_text(encoding="utf-8")
    )

    for question in eval_set["questions"]:
        for passage in question["relevant_passages"]:
            locator = (passage["source"].casefold(), passage["section"].casefold())
            assert locator in available, f"{question['id']} refers to missing {locator}"


def test_document_loader_strict_mode_reports_unreadable_file(tmp_path):
    document = tmp_path / "broken.txt"
    document.write_bytes(b"\xff\xfe\x00")

    with pytest.raises(ValueError, match=r"Could not read document.*broken\.txt.*decode|Could not read document.*broken\.txt.*codec"):
        load_and_chunk_data(tmp_path, strict=True)


def test_document_loader_logs_skipped_unreadable_file_and_keeps_good_files(tmp_path, caplog):
    (tmp_path / "broken.txt").write_bytes(b"\xff\xfe\x00")
    (tmp_path / "valid.txt").write_text(
        "OVERVIEW\nThis valid document contains enough words to become a retained chunk.",
        encoding="utf-8",
    )

    chunks = load_and_chunk_data(tmp_path)

    assert len(chunks) == 1
    assert chunks[0].source == "valid.txt"
    assert "broken.txt" in caplog.text


def test_document_loader_handles_utf8_bom_and_utf16_files(tmp_path):
    (tmp_path / "utf8_bom.txt").write_bytes(
        b"\xef\xbb\xbfOVERVIEW\nA short UTF-8 BOM section remains available."
    )
    (tmp_path / "utf16.txt").write_bytes(
        "OVERVIEW\nA short UTF-16 section remains available.".encode("utf-16")
    )

    chunks = load_and_chunk_data(tmp_path)

    assert len(chunks) == 2
    assert {chunk.topic for chunk in chunks} == {"utf8_bom", "utf16"}


def test_document_loader_decodes_cp1252_fallback(tmp_path):
    document = tmp_path / "legacy.txt"
    document.write_bytes(
        "OVERVIEW\nThe furnace’s temperature is measured.".encode("cp1252")
    )

    chunks = load_and_chunk_data(tmp_path)

    assert len(chunks) == 1
    assert "furnace’s temperature" in chunks[0].content


def test_document_loader_skips_empty_and_whitespace_documents(tmp_path, caplog):
    (tmp_path / "empty.txt").write_bytes(b"")
    (tmp_path / "whitespace.txt").write_text(" \n\t", encoding="utf-8")

    assert load_and_chunk_data(tmp_path) == []
    assert "empty.txt" in caplog.text
    assert "whitespace.txt" in caplog.text


def test_document_loader_warns_when_document_has_no_indexable_sections(tmp_path, caplog):
    (tmp_path / "references.txt").write_text(
        "REFERENCES\nA short citation.", encoding="utf-8"
    )

    assert load_and_chunk_data(tmp_path) == []
    assert "no indexable sections" in caplog.text


def test_document_loader_keeps_only_text_files(tmp_path):
    (tmp_path / "guide.txt").write_text(
        "OVERVIEW\nA short technical explanation is preserved.", encoding="utf-8"
    )
    (tmp_path / "generated.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.bin").write_bytes(b"\x00\x01")

    chunks = load_and_chunk_data(tmp_path)

    assert len(chunks) == 1
    assert Path(chunks[0].source).suffix == ".txt"


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


@pytest.mark.skipif(
    not EXPERIMENTS_CSV.exists(),
    reason=f"requires real plant data at {EXPERIMENTS_CSV} (gitignored, not present in this environment)",
)
def test_existing_experiment_gaps_are_reported_and_retained():
    with pytest.warns(ExperimentValidationWarning, match="missing value"):
        result = load_experiments_df()

    assert len(result) == 131
    assert result["row_index"].iloc[0] == 1
    assert ExperimentRow.__module__ == "departments.blast_furnace.bf_experiment_schema"