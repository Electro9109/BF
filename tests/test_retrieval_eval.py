import pytest

from data.schemas import Chunk
from eval.retrieval_eval import evaluate


def make_chunk(source, section="overview"):
    return Chunk("evidence", "evidence", source, "topic", section)


class FakeEngine:
    def __init__(self, results_by_query):
        self.results_by_query = results_by_query

    def search(self, query, top_n):
        return self.results_by_query[query][:top_n]


def test_evaluate_reports_ranking_source_coverage_latency_and_unknowns():
    first = make_chunk("a.txt")
    second = make_chunk("b.txt", "mechanism")
    engine = FakeEngine({
        "single": [(first, 0.9), (second, 0.8)],
        "multi": [(first, 0.9), (second, 0.8)],
        "unknown": [(first, 0.9)],
    })
    questions = [
        {
            "id": "single",
            "query": "single",
            "answerable": True,
            "relevant_passages": [{"source": "a.txt", "section": "overview"}],
        },
        {
            "id": "multi",
            "query": "multi",
            "answerable": True,
            "relevant_passages": [
                {"source": "a.txt", "section": "overview"},
                {"source": "b.txt", "section": "mechanism"},
            ],
        },
        {
            "id": "unknown",
            "query": "unknown",
            "answerable": False,
            "relevant_passages": [],
            "unanswerable_reason": "No evidence in fixture.",
        },
    ]
    ticks = iter(index * 0.010 for index in range(18))

    result = evaluate(
        engine,
        questions,
        (1, 2),
        repeats=3,
        clock=lambda: next(ticks),
    )

    assert result["metrics"]["recall@1"] == 0.75
    assert result["metrics"]["hit_rate@1"] == 1.0
    assert result["metrics"]["source_coverage@1"] == 0.75
    assert result["metrics"]["complete_passage_coverage@1"] == 0.5
    assert result["metrics"]["recall@2"] == 1.0
    assert result["metrics"]["mrr"] == 1.0
    assert result["metrics"]["mean_query_latency_ms"] == pytest.approx(10.0)
    assert result["metrics"]["unanswerable_retrieval_nonempty_rate"] == 1.0
    assert result["results"][0]["first_relevant_rank"] == 1
    assert result["warmup_searches"] == 1
    assert result["latency_repeats"] == 3


def test_evaluate_rejects_unsorted_or_nonpositive_k_values():
    with pytest.raises(ValueError, match="positive, and sorted"):
        evaluate(FakeEngine({}), [], (0, 1))


def test_evaluate_rejects_nonpositive_repetition_count():
    with pytest.raises(ValueError, match="repeats must be a positive"):
        evaluate(FakeEngine({}), [], repeats=0)


@pytest.mark.parametrize("ks", [(1.5,), (True,)])
def test_evaluate_rejects_noninteger_k_values(ks):
    with pytest.raises(ValueError, match="positive, and sorted"):
        evaluate(FakeEngine({}), [], ks=ks)


@pytest.mark.parametrize("repeats", [1.5, True])
def test_evaluate_rejects_noninteger_repetition_count(repeats):
    with pytest.raises(ValueError, match="repeats must be a positive integer"):
        evaluate(FakeEngine({}), [], repeats=repeats)
