"""
data/loader.py - Reads docs/*.txt and produces the chunk list consumed
by retrieval/retriever.py.

This is the function previously called load_and_chunk_data() in
rag_engine.py - extracted with no behavioural changes other than the
bug repairs documented in data/parser.py.
"""

import logging
from pathlib import Path

from data.parser import extract_topic, split_into_sections, strip_topic_header
from data.schemas import Chunk

logger = logging.getLogger(__name__)


def _read_document_text(file_path: Path) -> str:
    raw = file_path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")

    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as utf8_error:
        text = raw.decode("cp1252")
        if any(ord(char) < 32 and char not in "\n\r\t\f" for char in text):
            raise utf8_error
        return text


def load_and_chunk_data(docs_dir: str, strict: bool = False) -> list[Chunk]:
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

    When ``strict`` is true, unreadable files raise a ``ValueError`` instead
    of being skipped.
    """
    path = Path(docs_dir)
    if path.exists() and not path.is_dir():
        raise ValueError(f"Document path is not a directory: {path}")
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return []

    chunks: list[Chunk] = []
    for file_path in sorted(path.rglob("*.txt")):
        try:
            text = _read_document_text(file_path)
        except (OSError, UnicodeError) as exc:
            if strict:
                raise ValueError(f"Could not read document {file_path}: {exc}") from exc
            logger.warning("Skipping unreadable document %s: %s", file_path, exc)
            continue

        if not text.strip():
            logger.warning("Skipping empty document %s", file_path)
            continue

        rel_source = str(file_path.relative_to(path)).replace("\\", "/")

        # FIX: topic must be extracted BEFORE stripping the header line,
        # and BEFORE the chunk loop (previously `topic` was referenced
        # but never assigned at all).
        topic = extract_topic(text, file_path.stem)
        body = strip_topic_header(text)
        sections = split_into_sections(body)
        if not sections:
            logger.warning("Document has no indexable sections: %s", file_path)
            continue

        for section in sections:
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
