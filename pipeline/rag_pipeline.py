"""
pipeline/rag_pipeline.py - question -> retrieve -> generate -> answer.

This is the only module app_web.py's chatbot page should import for
retrieval+generation. It does not know about FAISS, embeddings, or the
ML/prediction side - if either of those changes, this file is
unaffected.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from config.paths import SOURCE_NORMALIZED_DIR
from data.loader import load_and_chunk_data
from data.schemas import Chunk
from llm.generator import generate
from llm.loader import load_model
from parse.adapters.bf import (
    answer_to_synthesis,
    chunk_to_knowledge_item,
    retrieval_matches_to_result,
)
from parse.core.contracts import RetrievalResult, SynthesisResult
from retrieval.retriever import RetrievalEngine

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Legacy RAG output together with PARSE retrieval and synthesis results."""

    answer: str
    matches: list[tuple[Chunk, float]]
    retrieval: RetrievalResult | None
    synthesis: SynthesisResult | None


def build_engine(docs_dir: str) -> tuple[RetrievalEngine | None, list[Chunk]]:
    """Load project and catalogued documents, then build the retrieval engine."""
    chunks = load_and_chunk_data(docs_dir)
    catalogued_docs = SOURCE_NORMALIZED_DIR / "documents"
    if catalogued_docs.is_dir():
        chunks.extend(load_and_chunk_data(catalogued_docs))
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


def _assemble_context_with_matches(
    ranked_matches: list,
    max_context_chars: int,
) -> tuple[str, list[tuple[Chunk, float]]]:
    """Pack source-labelled passages and return chunks included in context."""
    if max_context_chars <= 0:
        return "", []

    parts = []
    included_matches = []
    for original_rank, (chunk, score) in _order_for_context(ranked_matches):
        sec_label = chunk.section
        role_label = getattr(chunk, "role", "reference")
        header = (
            f"[Source {original_rank}: {chunk.source} | "
            f"Role: {role_label} | "
            f"Section: {sec_label}]\n"
        )
        passage = header + chunk.content
        separator = "\n\n" if parts else ""
        remaining = max_context_chars - sum(map(len, parts)) - len(separator)
        if len(passage) <= remaining:
            parts.append(separator + passage)
            included_matches.append((chunk, score))
            continue

        if parts:
            break

        marker = "\n[Passage truncated to fit context budget.]"
        content_budget = remaining - len(header) - len(marker)
        if content_budget <= 0:
            return "", included_matches
        parts.append(header + chunk.content[:content_budget] + marker)
        included_matches.append((chunk, score))
        break

    return "".join(parts), included_matches


def _assemble_context(ranked_matches: list, max_context_chars: int) -> str:
    """Pack complete source-labelled passages within the character budget."""
    return _assemble_context_with_matches(ranked_matches, max_context_chars)[0]


from config.models import MAX_NEW_TOKENS, REPETITION_PENALTY


def answer_question_structured(
    question: str,
    search_query: str,
    engine: RetrievalEngine,
    tokenizer,
    model,
    top_n: int,
    max_context_chars: int = 12000,
    max_new_tokens: int = MAX_NEW_TOKENS,
    repetition_penalty: float = REPETITION_PENALTY,
) -> RAGResponse:
    """
    Run retrieval and generation while preserving PARSE evidence contracts.

    `search_query` may differ from `question` (e.g. rolling context for
    short follow-ups) - retrieval uses search_query, but the LLM prompt
    uses the original `question`. Retrieval records all ranked matches while
    synthesis references only chunks included in the generation context.
    """
    if engine is None:
        return RAGResponse("No document index is available.", [], None, None)

    try:
        matches = engine.search(search_query, top_n=top_n)
    except Exception:
        logger.exception("Document retrieval failed for query %r", search_query)
        return RAGResponse(
            "Document retrieval failed. Please try again.",
            [],
            None,
            None,
        )

    retrieval = None
    if isinstance(search_query, str) and search_query.strip():
        try:
            retrieval = retrieval_matches_to_result(search_query, matches)
        except (TypeError, ValueError):
            logger.exception(
                "Could not represent retrieved evidence for query %r",
                search_query,
            )

    if not matches:
        return RAGResponse(
            "No relevant content found for that question. "
            "Try rephrasing, or upload the relevant document using the sidebar.",
            matches,
            retrieval,
            None,
        )

    ranked = list(enumerate(matches, 1))
    context, included_matches = _assemble_context_with_matches(
        ranked,
        max_context_chars,
    )
    if not context:
        return RAGResponse(
            "The context limit is too small to include source evidence.",
            matches,
            retrieval,
            None,
        )

    try:
        response = generate(
            question,
            context,
            tokenizer,
            model,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
    except Exception:
        logger.exception("Answer generation failed for question %r", question)
        return RAGResponse(
            "Answer generation failed. Please retry the question.",
            matches,
            retrieval,
            None,
        )

    response = _apply_hallucination_guard(response, context)
    synthesis = None
    if retrieval is not None:
        included_item_ids = {
            chunk_to_knowledge_item(chunk).item_id
            for chunk, _score in included_matches
        }
        try:
            synthesis = answer_to_synthesis(
                response,
                retrieval,
                included_item_ids=included_item_ids,
            )
        except (TypeError, ValueError):
            logger.exception(
                "Could not represent generated answer as PARSE synthesis"
            )
    return RAGResponse(response, matches, retrieval, synthesis)


def answer_question(
    question: str,
    search_query: str,
    engine: RetrievalEngine,
    tokenizer,
    model,
    top_n: int,
    max_context_chars: int = 12000,
    max_new_tokens: int = MAX_NEW_TOKENS,
    repetition_penalty: float = REPETITION_PENALTY,
) -> tuple[str, list[tuple[Chunk, float]]]:
    """Compatibility wrapper preserving the established answer/matches tuple."""
    result = answer_question_structured(
        question,
        search_query,
        engine,
        tokenizer,
        model,
        top_n,
        max_context_chars,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
    )
    return result.answer, result.matches


def sanitize_metallurgical_units(text: str) -> str:
    """Ensure basicity values are not erroneously formatted with temperature units.

    Basicity (B2 = CaO/SiO2, B4) is a dimensionless ratio. Generative models
    sometimes bleed adjacent temperature units (°C) from softening-melting contexts
    onto basicity figures.
    """
    if not text:
        return text
    # Strip °C or deg C following basicity references and ranges
    cleaned = re.sub(
        r'(\bbasicity\b[^.\n]*?\b\d+(?:\.\d+)?(?:\s*(?:-|–|—|to)\s*\d+(?:\.\d+)?)?)\s*(?:°\s*C|deg(?:rees?)?\s*C\b)',
        r'\1',
        text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r'(\bbasicity\b[^.\n]*?\(\s*around\s*\d+(?:\.\d+)?(?:\s*(?:-|–|—|to)\s*\d+(?:\.\d+)?)?\s*\))\s*(?:°\s*C|deg(?:rees?)?\s*C\b)',
        r'\1',
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned


def _apply_hallucination_guard(response: str, context: str) -> str:
    """
    This lightweight lexical-overlap check catches some ungrounded outputs,
    but it is not an entailment or semantic-support check.
    """
    response_words = set(re.findall(r"\b[\w'-]+\b", response.lower()))
    context_words = set(re.findall(r"\b[\w'-]+\b", context.lower()))
    content_overlap = [w for w in (response_words & context_words) if len(w) > 3]

    if len(content_overlap) < 3:
        return (
            "I could not find a confident answer in your documents. "
            "The retrieved paragraphs may not contain enough detail. "
            "Try rephrasing your question or adding more content to the document."
        )
    return sanitize_metallurgical_units(response)


__all__ = [
    "build_engine",
    "answer_question",
    "answer_question_structured",
    "RAGResponse",
    "load_model",
]
