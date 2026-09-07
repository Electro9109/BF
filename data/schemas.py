"""Generic document contracts shared by processing and retrieval."""

from dataclasses import asdict, dataclass


@dataclass
class Chunk:
    """A single retrievable unit of theory text."""
    content: str        # clean text -> sent to LLM
    embed_text: str      # "TOPIC: <t>\nSECTION: <s>\n<content>" -> sent to embedder
    source: str           # path relative to docs_dir
    topic: str             # canonical topic name
    section: str            # normalised section label

    def as_dict(self) -> dict:
        return asdict(self)
