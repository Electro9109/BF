import pytest

from data.schemas import Chunk
from llm.domain_prompts import SYSTEM_INSTRUCTION, build_domain_instructions
from llm.prompts import build_prompt
from parse.adapters.bf import chunk_to_knowledge_item
from pipeline.rag_pipeline import answer_question, answer_question_structured


def test_prompt_formatter_accepts_domain_policy():
    instructions = build_domain_instructions("explain reducibility")
    prompt = build_prompt("What is reducibility?", "Evidence text", instructions)

    assert SYSTEM_INSTRUCTION.startswith("You are a metallurgical")
    assert "Evidence text" in prompt
    assert "mechanisms" in prompt


def test_rag_pipeline_handles_missing_engine():
    answer, matches = answer_question("question", "question", None, None, None, 3)

    assert "No document index" in answer
    assert matches == []


def test_structured_rag_response_preserves_retrieval_and_context_evidence(monkeypatch):
    first = Chunk(
        content="Sinter process evidence supports this answer.",
        embed_text="",
        source="experiments/experiment_scenarios.txt",
        topic="experiments",
        section="overview",
    )
    second = Chunk(
        content="Different evidence supports another detail.",
        embed_text="",
        source="experiments/experiment_scenarios.txt",
        topic="experiments",
        section="mechanism",
    )
    matches = [(first, 0.91), (second, 0.82)]
    captured = {}

    def fake_generate(question, context, tokenizer, model, *args, **kwargs):
        captured["context"] = context
        return "Sinter process evidence supports this answer."

    monkeypatch.setattr("pipeline.rag_pipeline.generate", fake_generate)

    class FakeEngine:
        def search(self, query, top_n):
            return matches

    first_passage = (
        "[Source 1: experiments/experiment_scenarios.txt | Role: reference | Section: overview]\n"
        + first.content
    )
    result = answer_question_structured(
        "Question?",
        "Search question",
        FakeEngine(),
        None,
        None,
        2,
        max_context_chars=len(first_passage),
    )

    first_item = chunk_to_knowledge_item(first)
    second_item = chunk_to_knowledge_item(second)
    assert result.answer == "Sinter process evidence supports this answer."
    assert result.matches == matches
    assert result.retrieval is not None
    assert [item.item_id for item in result.retrieval.items] == [
        first_item.item_id,
        second_item.item_id,
    ]
    assert [item.score for item in result.retrieval.items] == [0.91, 0.82]
    assert result.retrieval.items[0].metadata["source"] == (
        "experiments/experiment_scenarios.txt"
    )
    assert result.retrieval.items[0].metadata["section"] == "overview"
    assert result.synthesis is not None
    assert result.synthesis.content == result.answer
    assert [ref.ref_id for ref in result.synthesis.evidence_refs] == [
        first_item.item_id
    ]
    assert "Section: overview" in captured["context"]
    assert "Section: mechanism" not in captured["context"]


def test_legacy_rag_api_remains_a_two_item_tuple():
    result = answer_question("question", "question", None, None, None, 3)

    assert type(result) is tuple
    assert len(result) == 2


def test_structured_rag_no_matches_and_generation_failure_keep_failure_shape(monkeypatch):
    class EmptyEngine:
        def search(self, query, top_n):
            return []

    empty = answer_question_structured(
        "question",
        "question",
        EmptyEngine(),
        None,
        None,
        3,
    )
    assert empty.retrieval is not None
    assert empty.retrieval.fallback == "no_matches"
    assert empty.synthesis is None

    chunk = Chunk(
        "Sinter process evidence supports this answer.",
        "",
        "sinter.txt",
        "sinter",
        "overview",
    )

    class OneMatchEngine:
        def search(self, query, top_n):
            return [(chunk, 0.9)]

    def fail_generation(*args):
        raise RuntimeError("inference failed")

    monkeypatch.setattr("pipeline.rag_pipeline.generate", fail_generation)
    failed = answer_question_structured(
        "question",
        "question",
        OneMatchEngine(),
        None,
        None,
        3,
    )

    assert "Answer generation failed" in failed.answer
    assert failed.retrieval is not None
    assert len(failed.retrieval.items) == 1
    assert failed.synthesis is None


def test_hybrid_pipeline_exposes_structured_rag_results_without_changing_legacy_keys(
    monkeypatch,
):
    from types import SimpleNamespace

    from parse.adapters.bf import answer_to_synthesis, retrieval_matches_to_result
    from pipeline.hybrid_pipeline import explain_prediction
    from pipeline.rag_pipeline import RAGResponse

    chunk = Chunk("Sinter process evidence.", "", "sinter.txt", "sinter", "overview")
    matches = [(chunk, 0.9)]
    retrieval = retrieval_matches_to_result("explain sinter", matches)
    synthesis = answer_to_synthesis("A supported answer.", retrieval)
    rag_response = RAGResponse("A supported answer.", matches, retrieval, synthesis)

    class FakePredictionResult:
        predictions = {"Ts": 1300.0}
        parser_warnings = []

        def to_rag_query(self):
            return "Explain the predicted result."

    class FakePredictionPipeline:
        def __init__(self, model_dir):
            pass

        def run(self, **features):
            return FakePredictionResult()

    monkeypatch.setattr(
        "pipeline.hybrid_pipeline.PredictionPipeline",
        FakePredictionPipeline,
    )
    monkeypatch.setattr(
        "pipeline.hybrid_pipeline.find_nearest_experiments",
        lambda chemistry, k: [],
    )
    monkeypatch.setattr(
        "pipeline.hybrid_pipeline.answer_question_structured",
        lambda **kwargs: rag_response,
    )

    result = explain_prediction(
        {"chemistry": {}},
        engine=object(),
        tokenizer=None,
        model=None,
    )

    assert result["explanation"] == "A supported answer."
    assert result["retrieved_sources"] == matches
    assert result["retrieval_result"] is retrieval
    assert result["synthesis_result"] is synthesis


def test_rag_pipeline_reports_generation_failure():
    chunk = Chunk(
        content="Reducibility describes how readily an oxide is reduced.",
        embed_text="Reducibility describes how readily an oxide is reduced.",
        source="reducibility.txt",
        topic="reducibility",
        section="overview",
    )

    class FakeEngine:
        def search(self, query, top_n):
            return [(chunk, 0.9)]

    answer, matches = answer_question(
        "What is reducibility?",
        "reducibility",
        FakeEngine(),
        None,
        None,
        3,
    )

    assert "Answer generation failed" in answer
    assert matches[0][0].source == "reducibility.txt"

def test_order_for_context_places_top_ranks_at_edges():
    from pipeline.rag_pipeline import _order_for_context

    ranks = [1, 2, 3, 4, 5]
    ordered = _order_for_context(ranks)

    assert ordered[0] == 1  # rank 1 -> start
    assert ordered[-1] == 2  # rank 2 -> end
    assert ordered[len(ordered) // 2] == 5  # weakest rank -> nearest the middle


def test_answer_question_uses_position_aware_context_order(monkeypatch):
    chunks = [
        Chunk(content=f"content {i}", embed_text=f"content {i}", source=f"doc{i}.txt", topic="t", section="s")
        for i in range(1, 6)
    ]
    matches = [(chunk, 1.0 - i * 0.1) for i, chunk in enumerate(chunks)]

    captured = {}

    def fake_generate(question, context, tokenizer, model, *args, **kwargs):
        captured["context"] = context
        return "an answer that mentions content"

    monkeypatch.setattr("pipeline.rag_pipeline.generate", fake_generate)

    class FakeEngine:
        def search(self, query, top_n):
            return matches

    answer_question("q", "q", FakeEngine(), None, None, 5)

    context = captured["context"]
    # Rank 1 (doc1) first, rank 2 (doc2) last, weakest rank (doc5) nearest the middle.
    assert context.index("doc1.txt") < context.index("doc3.txt")
    assert context.index("doc2.txt") > context.index("doc3.txt")
    # Matches returned to the caller (not tested here) remain rank-ordered -
    # covered by the assertion above using original rank labels ("Source N").
    assert "[Source 1: doc1.txt | Role: reference | Section: s]" in context
    assert "[Source 2: doc2.txt | Role: reference | Section: s]" in context


def test_context_assembly_keeps_passages_complete_and_labels_sections():
    from pipeline.rag_pipeline import _assemble_context

    first = Chunk(
        content="First complete evidence passage.",
        embed_text="",
        source="first.txt",
        topic="topic",
        section="overview",
    )
    second = Chunk(
        content="Second complete evidence passage.",
        embed_text="",
        source="second.txt",
        topic="topic",
        section="mechanism",
    )
    matches = [(1, (first, 0.9)), (2, (second, 0.8))]

    context = _assemble_context(matches, 120)

    assert "[Source 1: first.txt | Role: reference | Section: overview]" in context
    assert "First complete evidence passage." in context
    assert "[Source 2: second.txt" not in context
    assert len(context) <= 120


def test_context_assembly_marks_an_individually_truncated_passage():
    from pipeline.rag_pipeline import _assemble_context

    chunk = Chunk(
        content="technical evidence " * 30,
        embed_text="",
        source="long.txt",
        topic="topic",
        section="mechanism",
    )

    context = _assemble_context([(1, (chunk, 0.9))], 140)

    assert len(context) <= 140
    assert "[Passage truncated to fit context budget.]" in context
    assert context.startswith("[Source 1: long.txt | Role: reference | Section: mechanism]")


def test_answer_question_skips_generation_if_context_budget_cannot_fit_evidence(monkeypatch):
    chunk = Chunk("evidence", "", "doc.txt", "topic", "overview")

    class FakeEngine:
        def search(self, query, top_n):
            return [(chunk, 0.9)]

    monkeypatch.setattr(
        "pipeline.rag_pipeline.generate",
        lambda *args: pytest.fail("generation must not receive empty evidence"),
    )

    answer, matches = answer_question("question", "query", FakeEngine(), None, None, 1, 1)

    assert "too small" in answer
    assert matches[0][0] is chunk


def test_lexical_grounding_guard_false_rejects_paraphrase_and_false_accepts_causal_claim():
    from pipeline.rag_pipeline import _apply_hallucination_guard

    context = "Reducibility describes how readily iron oxide loses oxygen."
    paraphrase = "The ore gives up oxygen with ease."
    unsupported_cause = "Reducibility causes oxygen loss in iron oxide."

    assert _apply_hallucination_guard(paraphrase, context).startswith(
        "I could not find a confident answer"
    )
    assert _apply_hallucination_guard(unsupported_cause, context) == unsupported_cause


def test_generation_settings_passed_correctly(monkeypatch):
    """Verify max_new_tokens and repetition_penalty propagate to generate."""
    captured = {}

    def fake_generate(question, context, tokenizer, model, max_new_tokens, repetition_penalty):
        captured["max_new_tokens"] = max_new_tokens
        captured["repetition_penalty"] = repetition_penalty
        return "Complete generated answer with sufficient technical evidence."

    monkeypatch.setattr("pipeline.rag_pipeline.generate", fake_generate)

    class FakeEngine:
        def search(self, query, top_n):
            c = Chunk("Complete generated answer with sufficient technical evidence.", "", "doc.txt", "topic", "overview")
            return [(c, 0.9)]

    res = answer_question_structured(
        "test question",
        "test query",
        FakeEngine(),
        None,
        None,
        1,
        max_new_tokens=768,
        repetition_penalty=1.12,
    )
    assert captured["max_new_tokens"] == 768
    assert captured["repetition_penalty"] == 1.12
    assert "Complete generated answer" in res.answer


def test_complete_answer_not_artificially_truncated_by_pipeline():
    """Verify long answers up to hundreds of words are returned without truncation."""
    from pipeline.rag_pipeline import _apply_hallucination_guard
    long_answer = "This is a comprehensive response discussing slag basicity, softening temperature, and cohesive zone dynamics. " * 20
    context = "slag basicity softening temperature cohesive zone"
    result = _apply_hallucination_guard(long_answer, context)
    assert len(result) == len(long_answer)
    assert result.endswith("cohesive zone dynamics. ")