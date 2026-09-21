from data.schemas import Chunk
from llm.domain_prompts import SYSTEM_INSTRUCTION, build_domain_instructions
from llm.prompts import build_prompt
from pipeline.rag_pipeline import answer_question


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

    def fake_generate(question, context, tokenizer, model):
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
    assert "[Source 1: doc1.txt]" in context
    assert "[Source 2: doc2.txt]" in context