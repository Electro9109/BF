"""Qualitative Scenario Evaluation Harness for PARSE.

Evaluates RAG retrieval and answer reasoning on qualitative experiment scenarios,
covering directional accuracy, target-variable coverage, required concepts,
conditional outcome handling, and uncertainty awareness.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.paths import DOCS_DIR
from pipeline.rag_pipeline import build_engine

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO_EVAL_SET = Path(__file__).with_name("qualitative_scenario_eval_set.json")


def score_directional_accuracy(response_text: str, expected_directions: dict[str, str]) -> float:
    """Score whether expected directional claims (increase, decrease, narrow, wide) appear appropriately."""
    if not expected_directions:
        return 1.0

    text_lower = response_text.lower()
    matched = 0
    total = len(expected_directions)

    dir_synonyms = {
        "increase": ["increase", "increases", "increased", "higher", "rise", "rises", "widen", "widens", "widest", "expanding"],
        "decrease": ["decrease", "decreases", "decreased", "lower", "drop", "drops", "narrow", "narrows", "narrowest", "reducing"],
        "highest": ["highest", "maximum", "peak", "highest softening"],
        "narrow": ["narrow", "narrows", "narrower", "narrowest", "thin"],
        "wide": ["wide", "widens", "wider", "widest", "broad"],
        "conditional": ["depend", "depends", "conditional", "tradeoff", "trade-off", "varies", "context"],
        "unresolved": ["unknown", "cannot", "insufficient", "not possible", "uncharacterized", "lacks", "missing"],
    }

    for var, direction in expected_directions.items():
        syns = dir_synonyms.get(direction.lower(), [direction.lower()])
        # Look for co-occurrence or directional synonym presence
        if any(s in text_lower for s in syns):
            matched += 1

    return matched / total if total > 0 else 1.0


def score_target_variable_coverage(response_text: str, target_variables: list[str]) -> float:
    """Score whether all required target variables are discussed in the response."""
    if not target_variables:
        return 1.0
    import re
    matched = 0
    for v in target_variables:
        pattern = r"\b" + re.escape(v.strip().lower()) + r"\b"
        if re.search(pattern, response_text.lower()):
            matched += 1
    return matched / len(target_variables)


def score_required_concepts(response_text: str, required_concepts: list[str]) -> float:
    """Score whether required qualitative concepts or mechanisms are represented."""
    if not required_concepts:
        return 1.0
    text_lower = response_text.lower()
    matched = 0
    for concept in required_concepts:
        # Check if keywords in the concept are covered
        words = [w.lower() for w in concept.split() if len(w) > 3]
        if words and any(w in text_lower for w in words):
            matched += 1
    return matched / len(required_concepts)


def score_caveat_handling(response_text: str, required_caveats: list[str]) -> float:
    """Score whether required qualifications, limitations, or caveats are acknowledged."""
    if not required_caveats:
        return 1.0
    text_lower = response_text.lower()
    caveat_indicators = [
        "depend", "however", "although", "caveat", "limitation", "provided",
        "assum", "conditional", "tradeoff", "cannot", "insufficient", "context"
    ]
    has_caveat_marker = any(ind in text_lower for ind in caveat_indicators)
    return 1.0 if has_caveat_marker else 0.0


def evaluate_qualitative_scenarios(
    engine,
    cases: list[dict[str, Any]],
    top_k: int = 5,
) -> dict[str, Any]:
    """Execute retrieval and qualitative scoring on retrieved context across all benchmark scenario cases.

    NOTE: This mode scores the retrieved context passages, NOT generated LLM answers.
    """
    case_results = []

    verified_directional_scores = []
    variable_coverage_scores = []
    concept_coverage_scores = []
    caveat_scores = []
    scenario_hit_scores = []

    status_counts = {"verified": 0, "conditional": 0, "illustrative": 0, "unresolved": 0}

    for case in cases:
        cid = case["case_id"]
        vstatus = case.get("verification_status", "verified")
        status_counts[vstatus] = status_counts.get(vstatus, 0) + 1

        base_query = case["question"]
        scenario_id = case.get("scenario_id")
        # If scenario_id is specific (e.g. "Scenario 1"), include it in retrieval query
        if scenario_id and "Scenario" in scenario_id:
            query = f"{scenario_id}: {base_query}"
        else:
            query = base_query

        matches = engine.search(query, top_n=top_k) if engine else []

        # Check retrieval source match
        req_source = case.get("evidence_requirements")
        retrieved_sources = [c.source for c, _score in matches]
        scenario_hit = False
        if isinstance(req_source, str) and ("docs/" in req_source or ".txt" in req_source):
            expected_stem = Path(req_source).name
            scenario_hit = any(expected_stem in s for s in retrieved_sources)
        elif isinstance(req_source, list):
            scenario_hit = any(any(Path(r).name in s for s in retrieved_sources) for r in req_source)
        else:
            scenario_hit = True
        scenario_hit_scores.append(1.0 if scenario_hit else 0.0)

        # Concatenate retrieved text as reference evidence for deterministic scoring
        retrieved_text = " ".join(c.content for c, _ in matches)

        # Directional scoring
        exp_dirs = case.get("expected_directions", {})
        dir_acc = score_directional_accuracy(retrieved_text, exp_dirs)
        if vstatus == "verified":
            verified_directional_scores.append(dir_acc)

        # Variable coverage
        var_cov = score_target_variable_coverage(retrieved_text, case.get("target_variables", []))
        variable_coverage_scores.append(var_cov)

        # Concept coverage
        con_cov = score_required_concepts(retrieved_text, case.get("required_concepts", []))
        concept_coverage_scores.append(con_cov)

        # Caveat score
        cav_score = score_caveat_handling(retrieved_text, case.get("required_caveats", []))
        caveat_scores.append(cav_score)

        case_results.append({
            "case_id": cid,
            "verification_status": vstatus,
            "case_type": case.get("case_type"),
            "scenario_hit": scenario_hit,
            "retrieved_sources": retrieved_sources[:3],
            "directional_accuracy": dir_acc,
            "target_variable_coverage": var_cov,
            "required_concept_coverage": con_cov,
            "caveat_score": cav_score,
        })

    aggregate_metrics = {
        "evaluation_mode": "retrieval_context",
        "total_cases_evaluated": len(cases),
        "status_breakdown": status_counts,
        "eligible_verified_directional_accuracy": (
            sum(verified_directional_scores) / len(verified_directional_scores)
            if verified_directional_scores else 0.0
        ),
        "verified_case_count": len(verified_directional_scores),
        "mean_target_variable_coverage": sum(variable_coverage_scores) / len(variable_coverage_scores) if variable_coverage_scores else 0.0,
        "mean_required_concept_coverage": sum(concept_coverage_scores) / len(concept_coverage_scores) if concept_coverage_scores else 0.0,
        "mean_caveat_handling": sum(caveat_scores) / len(caveat_scores) if caveat_scores else 0.0,
        "scenario_retrieval_hit_rate": sum(scenario_hit_scores) / len(scenario_hit_scores) if scenario_hit_scores else 0.0,
    }

    return {
        "aggregate_metrics": aggregate_metrics,
        "case_results": case_results,
    }


def evaluate_live_scenarios(
    engine,
    cases: list[dict[str, Any]],
    tokenizer=None,
    model=None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Execute live LLM answer generation and separate answer evaluation."""
    from pipeline.rag_pipeline import answer_question_structured

    case_results = []
    verified_directional_scores = []
    variable_coverage_scores = []
    concept_coverage_scores = []
    caveat_scores = []
    scenario_hit_scores = []

    status_counts = {"verified": 0, "conditional": 0, "illustrative": 0, "unresolved": 0}

    for case in cases:
        cid = case["case_id"]
        vstatus = case.get("verification_status", "verified")
        status_counts[vstatus] = status_counts.get(vstatus, 0) + 1

        base_query = case["question"]
        scenario_id = case.get("scenario_id")
        if scenario_id and "Scenario" in scenario_id:
            search_query = f"{scenario_id}: {base_query}"
        else:
            search_query = base_query

        rag_response = answer_question_structured(
            question=base_query,
            search_query=search_query,
            engine=engine,
            tokenizer=tokenizer,
            model=model,
            top_n=top_k,
        )

        retrieved_sources = [c.source for c, _score in rag_response.matches]
        req_source = case.get("evidence_requirements")
        scenario_hit = False
        if isinstance(req_source, str) and ("docs/" in req_source or ".txt" in req_source):
            expected_stem = Path(req_source).name
            scenario_hit = any(expected_stem in s for s in retrieved_sources)
        elif isinstance(req_source, list):
            scenario_hit = any(any(Path(r).name in s for s in retrieved_sources) for r in req_source)
        else:
            scenario_hit = True
        scenario_hit_scores.append(1.0 if scenario_hit else 0.0)

        answer_text = rag_response.answer or ""

        exp_dirs = case.get("expected_directions", {})
        dir_acc = score_directional_accuracy(answer_text, exp_dirs)
        if vstatus == "verified":
            verified_directional_scores.append(dir_acc)

        var_cov = score_target_variable_coverage(answer_text, case.get("target_variables", []))
        variable_coverage_scores.append(var_cov)

        con_cov = score_required_concepts(answer_text, case.get("required_concepts", []))
        concept_coverage_scores.append(con_cov)

        cav_score = score_caveat_handling(answer_text, case.get("required_caveats", []))
        caveat_scores.append(cav_score)

        case_results.append({
            "case_id": cid,
            "verification_status": vstatus,
            "case_type": case.get("case_type"),
            "scenario_hit": scenario_hit,
            "retrieved_sources": retrieved_sources[:3],
            "generated_answer": answer_text,
            "directional_accuracy": dir_acc,
            "target_variable_coverage": var_cov,
            "required_concept_coverage": con_cov,
            "caveat_score": cav_score,
        })

    aggregate_metrics = {
        "evaluation_mode": "live_llm_answer",
        "total_cases_evaluated": len(cases),
        "status_breakdown": status_counts,
        "eligible_verified_directional_accuracy": (
            sum(verified_directional_scores) / len(verified_directional_scores)
            if verified_directional_scores else 0.0
        ),
        "verified_case_count": len(verified_directional_scores),
        "mean_target_variable_coverage": sum(variable_coverage_scores) / len(variable_coverage_scores) if variable_coverage_scores else 0.0,
        "mean_required_concept_coverage": sum(concept_coverage_scores) / len(concept_coverage_scores) if concept_coverage_scores else 0.0,
        "mean_caveat_handling": sum(caveat_scores) / len(caveat_scores) if caveat_scores else 0.0,
        "scenario_retrieval_hit_rate": sum(scenario_hit_scores) / len(scenario_hit_scores) if scenario_hit_scores else 0.0,
    }

    return {
        "aggregate_metrics": aggregate_metrics,
        "case_results": case_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_SCENARIO_EVAL_SET)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live generation through the LLM pipeline rather than scoring retrieved text alone.",
    )
    args = parser.parse_args()

    engine, chunks = build_engine(str(args.docs_dir))
    with open(args.eval_set, encoding="utf-8") as f:
        data = json.load(f)

    cases = data.get("cases", [])
    if args.live:
        try:
            from llm.loader import load_model
            tokenizer, model = load_model()
        except Exception as exc:
            print(f"Error loading model dependencies for live evaluation: {exc}")
            return
        report = evaluate_live_scenarios(engine, cases, tokenizer=tokenizer, model=model, top_k=args.top_k)
    else:
        report = evaluate_qualitative_scenarios(engine, cases, top_k=args.top_k)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
