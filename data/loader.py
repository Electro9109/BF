"""
data/loader.py - Reads docs/*.txt and produces the chunk list consumed
by retrieval/retriever.py.

This is the function previously called load_and_chunk_data() in
rag_engine.py - extracted with no behavioural changes other than the
bug repairs documented in data/parser.py.
"""

from pathlib import Path

from data.parser import extract_topic, split_into_sections, strip_topic_header
from data.schemas import Chunk


def load_and_chunk_data(docs_dir: str) -> list[Chunk]:
    """
    Recursively reads every .txt file under docs_dir, splits into semantic
    sections, and returns a list of Chunk dicts:

        {
            "content"   : str,   # clean text -> sent to LLM
            "embed_text": str,   # TOPIC+SECTION prefix -> sent to embedder
            "source"    : str,   # path relative to docs_dir
            "topic"     : str,   # canonical topic name
            "section"   : str,   # normalised section label
        }

    Recommended: add  TOPIC: <name>  as the first line of each .txt file.
    Falls back to the filename stem if the header is absent.
    """
    path = Path(docs_dir)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return []

    chunks: list[Chunk] = []
    for file_path in sorted(path.rglob("*.txt")):
        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception:
            continue

        rel_source = str(file_path.relative_to(path)).replace("\\", "/")

        # FIX: topic must be extracted BEFORE stripping the header line,
        # and BEFORE the chunk loop (previously `topic` was referenced
        # but never assigned at all).
        topic = extract_topic(text, file_path.stem)
        body = strip_topic_header(text)

        for section in split_into_sections(body):
            embed_text = (
                f"TOPIC: {topic}\n"
                f"SECTION: {section['section']}\n"
                f"{section['content']}"
            )

            chunks.append(Chunk(
                content=section["content"],
                embed_text=embed_text,
                source=rel_source,
                topic=topic,
                section=section["section"],
            ))

    return chunks
