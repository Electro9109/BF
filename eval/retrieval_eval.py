"""Evaluate document retrieval against manually labelled passage locators."""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable

from pipeline.rag_pipeline import build_engine

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_SET = Path(__file__).with_name("retrieval_eval_set.json")
DEFAULT_DOCS_DIR = ROOT / "docs"


def _passage_key(source: str, section: str) -> tuple[str, str]:
    return source.replace("\\", "/").casefold(), section.strip().casefold()


def _validate_questions(questions: list[dict]) -> None:
    ids = set()
    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("Each evaluation question must be a JSON object.")
        question_id = question.get("id")
        if not isinstance(question_id, str) or not question_id or question_id in ids:
            raise ValueError(f"Missing or duplicate evaluation question id: {question_id!r}")
        ids.add(question_id)

        query = question.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Question {question_id!r} has no query.")
        answerable = question.get("answerable")
        passages = question.get("relevant_passages")
        if not isinstance(answerable, bool) or not isinstance(passages, list):
            raise ValueError(f"Question {question_id!r} has invalid labels.")
        if any(
            not isinstance(passage, dict)
            or not isinstance(passage.get("source"), str)
            or not passage["source"].strip()
            or not isinstance(passage.get("section"), str)
            or not passage["section"].strip()
            for passage in passages
        ):
            raise ValueError(
                f"Question {question_id!r} has an invalid passage locator."
            )
        if answerable != bool(passages):
            raise ValueError(
                f"Question {question_id!r} must have relevant passages exactly "
                "when it is labelled answerable."
            )
        if not answerable and not question.get("unanswerable_reason"):
            raise ValueError(
                f"Unanswerable question {question_id!r} needs a reason."
            )


def evaluate(
    engine,
    questions: list[dict],
    ks: tuple[int, ...] = (1, 3, 5),
    repeats: int = 3,
    clock: Callable[[], float] = time.perf_counter,
) -> dict:
    """Run retrieval and calculate ranking, coverage, and latency metrics."""
    if (
        not ks
        or any(not isinstance(k, int) or isinstance(k, bool) or k <= 0 for k in ks)
        or tuple(sorted(set(ks))) != ks
    ):
        raise ValueError("ks must be unique, positive, and sorted ascending.")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats <= 0:
        raise ValueError("repeats must be a positive integer.")
    _validate_questions(questions)
    answerable = [item for item in questions if item["answerable"]]
    unanswerable = [item for item in questions if not item["answerable"]]
    if not answerable:
        raise ValueError("The evaluation set must contain answerable questions.")

    engine.search(questions[0]["query"], top_n=max(ks))
    rows = []
    latencies = []
    for question in questions:
        samples = []
        for _ in range(repeats):
            started = clock()
            matches = engine.search(question["query"], top_n=max(ks))
            samples.append((clock() - started) * 1000)
        elapsed_ms = statistics.median(samples)
        latencies.append(elapsed_ms)
        retrieved = [
            _passage_key(chunk.source, chunk.section)
            for chunk, _score in matches
        ]
        relevant = {
            _passage_key(passage["source"], passage["section"])
            for passage in question["relevant_passages"]
        }
        relevant_sources = {source for source, _section in relevant}
        rows.append({
            "id": question["id"],
            "answerable": question["answerable"],
            "elapsed_ms": elapsed_ms,
            "retrieved": [
                {"source": chunk.source, "section": chunk.section}
                for chunk, _score in matches
            ],
            "first_relevant_rank": next(
                (rank for rank, key in enumerate(retrieved, start=1) if key in relevant),
                None,
            ),
            "_retrieved_keys": retrieved,
            "_relevant_keys": relevant,
            "_relevant_sources": relevant_sources,
        })

    metrics = {}
    for k in ks:
        answerable_rows = [row for row in rows if row["answerable"]]
        hits = [
            row for row in answerable_rows
            if any(key in row["_relevant_keys"] for key in row["_retrieved_keys"][:k])
        ]
        passage_coverage = [
            len(row["_relevant_keys"].intersection(row["_retrieved_keys"][:k]))
            / len(row["_relevant_keys"])
            for row in answerable_rows
        ]
        source_coverage = [
            len(
                row["_relevant_sources"]
                & {source for source, _section in row["_retrieved_keys"][:k]}
            ) / len(row["_relevant_sources"])
            for row in answerable_rows
        ]
        metrics[f"recall@{k}"] = statistics.mean(passage_coverage)
        metrics[f"hit_rate@{k}"] = len(hits) / len(answerable_rows)
        metrics[f"source_coverage@{k}"] = statistics.mean(source_coverage)
        metrics[f"complete_passage_coverage@{k}"] = sum(
            value == 1.0 for value in passage_coverage
        ) / len(passage_coverage)

    metrics["mrr"] = statistics.mean(
        1 / row["first_relevant_rank"]
        if row["first_relevant_rank"] is not None else 0.0
        for row in rows if row["answerable"]
    )
    metrics["mean_query_latency_ms"] = statistics.mean(latencies)
    metrics["median_query_latency_ms"] = statistics.median(latencies)
    metrics["unanswerable_retrieval_nonempty_rate"] = (
        sum(bool(row["_retrieved_keys"]) for row in rows if not row["answerable"])
        / len(unanswerable)
        if unanswerable else None
    )

    for row in rows:
        del row["_retrieved_keys"]
        del row["_relevant_keys"]
        del row["_relevant_sources"]
    return {
        "question_count": len(questions),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "ks": list(ks),
        "warmup_searches": 1,
        "latency_repeats": repeats,
        "metrics": metrics,
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    eval_set = json.loads(args.eval_set.read_text(encoding="utf-8"))
    if eval_set.get("version") != 1 or not isinstance(eval_set.get("questions"), list):
        raise ValueError(f"Unsupported or malformed evaluation set: {args.eval_set}")

    engine, chunks = build_engine(str(args.docs_dir))
    if engine is None:
        raise RuntimeError(f"No searchable document chunks found in {args.docs_dir}")
    result = evaluate(
        engine,
        eval_set["questions"],
        tuple(args.ks),
        repeats=args.repeats,
    )
    result["corpus_chunk_count"] = len(chunks)
    result["corpus_documents"] = len({chunk.source for chunk in chunks})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
