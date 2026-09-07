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