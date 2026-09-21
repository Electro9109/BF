"""
pipeline/rag_pipeline.py - question -> retrieve -> generate -> answer.

This is the only module app_web.py's chatbot page should import for
retrieval+generation. It does not know about FAISS, embeddings, or the
ML/prediction side - if either of those changes, this file is
unaffected.
"""

from data.loader import load_and_chunk_data
from data.schemas import Chunk
from llm.generator import generate
from llm.loader import load_model
from retrieval.retriever import RetrievalEngine


def build_engine(docs_dir: str) -> tuple[RetrievalEngine | None, list[Chunk]]:
    """Load and chunk all docs, then build the retrieval engine."""
    chunks = load_and_chunk_data(docs_dir)
    engine = RetrievalEngine(chunks) if chunks else None
    return engine, chunks


def _order_for_context(matches: list) -> list:
    """Reorder ranked matches so top-ranked evidence sits at the start AND
    end of the assembled context, with lower-ranked matches pushed toward
    the middle.

    "Lost in the Middle" (Liu et al.) found LLMs are least reliable at
    using information placed in the middle of a long context, even for
    models built for long windows -- a plain top-to-bottom concatenation
    of ranked matches puts everything after rank 1 progressively deeper
    into that weak spot. This does not change what's returned to callers
    (the UI diagnostics panel and hallucination-guard overlap check both
    depend on the original rank order) -- only the order the text is
    concatenated in for the LLM prompt.
    """
    ordered = [None] * len(matches)
    left, right = 0, len(matches) - 1
    for i, match in enumerate(matches):
        if i % 2 == 0:
            ordered[left] = match
            left += 1
        else:
            ordered[right] = match
            right -= 1
    return ordered


def answer_question(
    question: str,
    search_query: str,
    engine: RetrievalEngine,
    tokenizer,
    model,
    top_n: int,
    max_context_chars: int = 12000,
) -> tuple[str, list[tuple[Chunk, float]]]:
    """
    Run retrieval + generation for one question.

    `search_query` may differ from `question` (e.g. rolling context for
    short follow-ups) - retrieval uses search_query, but the LLM prompt
    uses the original `question`.

    Returns (answer_text, matches) where matches is the same
    [(chunk, score), ...] list shown in the diagnostics panel.
    """
    if engine is None:
        return "No document index is available.", []

    try:
        matches = engine.search(search_query, top_n=top_n)
    except Exception as exc:
        return f"Document retrieval failed: {exc}", []

    if not matches:
        return (
            "No relevant content found for that question. "
            "Try rephrasing, or upload the relevant document using the sidebar."
        ), matches

    ranked = list(enumerate(matches, 1))  # [(original_rank, (chunk, score)), ...]
    context_parts = []
    for original_rank, (chunk, score) in _order_for_context(ranked):
        context_parts.append(f"[Source {original_rank}: {chunk.source}]\n{chunk.content}")
    context = "\n\n".join(context_parts)[:max_context_chars]

    try:
        response = generate(question, context, tokenizer, model)
    except Exception as exc:
        return f"Answer generation failed: {exc}", matches

    response = _apply_hallucination_guard(response, context)
    return response, matches


def _apply_hallucination_guard(response: str, context: str) -> str:
    """
    If the response shares fewer than 3 content words (4+ chars) with the
    retrieved context, the model likely ignored the context and generated
    from its own weights. Replace with a safe fallback.
    """
    response_words = set(response.lower().split())
    context_words = set(context.lower().split())
    content_overlap = [w for w in (response_words & context_words) if len(w) > 3]

    if len(content_overlap) < 3:
        return (
            "I could not find a confident answer in your documents. "
            "The retrieved paragraphs may not contain enough detail. "
            "Try rephrasing your question or adding more content to the document."
        )
    return response


__all__ = ["build_engine", "answer_question", "load_model"]