"""Tests for the qualitative experiment evaluation harness, role preservation, and pilot ingestion."""

import json
import pytest
from pathlib import Path

from config.paths import DOCS_DIR
from data.loader import load_and_chunk_data
from data.parser import extract_role, extract_topic
from eval.qualitative_eval import (
    evaluate_qualitative_scenarios,
    score_directional_accuracy,
    score_target_variable_coverage,
    score_required_concepts,
    score_caveat_handling,
)
from pipeline.rag_pipeline import build_engine


def test_document_role_extraction():
    # Explicit ROLE header
    text_with_role = "TOPIC: Test\nROLE: experiment_scenario\n\nContent..."
    assert extract_role(text_with_role) == "experiment_scenario"

    # Default to reference
    text_ref = "TOPIC: Test\n\nContent..."
    assert extract_role(text_ref) == "reference"

    # Inferred from file path
    assert extract_role(text_ref, "docs/experiments/test.txt") == "experiment_scenario"


def test_chunking_preserves_role_and_scenario_headings():
    chunks = load_and_chunk_data(str(DOCS_DIR / "experiments"))
    assert len(chunks) >= 8
    for c in chunks:
        assert c.role == "experiment_scenario"

    # Check scenario 1 section label exists
    labels = [c.section.lower() for c in chunks]
    assert any("scenario 1" in l for l in labels)


def test_pilot_document_ingestion_and_chunking():
    pilot_path = DOCS_DIR / "coke_quality.txt"
    assert pilot_path.exists(), "Pilot document docs/coke_quality.txt must exist"

    engine, chunks = build_engine(str(DOCS_DIR))
    coke_chunks = [c for c in chunks if "coke_quality.txt" in c.source]
    assert len(coke_chunks) >= 4
    for c in coke_chunks:
        assert c.role == "reference"

    # Distinctive retrieval test
    matches = engine.search("What is Coke Strength after Reaction CSR and Coke Reactivity Index CRI?", top_n=3)
    assert len(matches) > 0
    top_chunk, top_score = matches[0]
    assert "coke_quality.txt" in top_chunk.source
    assert top_score > 0.4


def test_qualitative_evaluation_scoring_functions():
    # Directional scoring
    resp = "Softening temperature Ts is expected to decrease significantly while Tm-Ts widens."
    assert score_directional_accuracy(resp, {"Ts": "decrease", "Tm-Ts": "increase"}) == 1.0

    # Target variable coverage
    assert score_target_variable_coverage(resp, ["Ts", "Tm-Ts"]) == 1.0
    assert score_target_variable_coverage(resp, ["Ts", "reducibility", "pressure drop"]) == 1 / 3

    # Required concepts
    assert score_required_concepts(resp, ["softening temperature", "slag viscosity"]) == 0.5

    # Caveat handling
    assert score_caveat_handling("This depends on the initial ore pre-reduction state.", ["caveat"]) == 1.0
    assert score_caveat_handling("This is an absolute constant.", ["caveat"]) == 0.0


def test_qualitative_scenario_eval_suite():
    eval_path = Path(__file__).resolve().parents[1] / "eval" / "qualitative_scenario_eval_set.json"
    assert eval_path.exists()

    with open(eval_path, encoding="utf-8") as f:
        data = json.load(f)

    cases = data.get("cases", [])
    assert len(cases) >= 8

    engine, _ = build_engine(str(DOCS_DIR))
    report = evaluate_qualitative_scenarios(engine, cases, top_k=5)

    agg = report["aggregate_metrics"]
    assert agg["total_cases_evaluated"] == len(cases)
    assert agg["eligible_verified_directional_accuracy"] >= 0.85
    assert agg["mean_target_variable_coverage"] >= 0.90
    assert agg["mean_caveat_handling"] >= 0.80
