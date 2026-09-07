"""
config/retrieval.py
────────────────────
Generic retrieval tuning knobs: result counts, filtering thresholds,
and section-based re-rank weights. BF/Sinter vocabulary lives in
config.retrieval_domain.
"""

# Number of chunks returned by RetrievalEngine.search() by default
TOP_K = 5

# Minimum candidate chunks required for a topic filter to be considered
# "valid". If a topic match returns fewer than this, fall back to the
# full corpus (see retrieval/retriever.py).
MIN_TOPIC_CANDIDATES = 3

# Section headings whose content adds no retrieval value.
# Chunks whose first words match any of these are silently dropped from the index.
JUNK_HEADINGS = {
    "references", "citations", "sources", "bibliography",
    "further reading", "see also", "notes", "acknowledgements",
    "acknowledgments", "appendix",
}

# Sections kept in the index but penalised for explanatory queries
LOW_VALUE_SECTIONS = {"limitations", "disadvantages", "challenges", "drawbacks"}

# Re-rank multipliers
EXPLANATORY_BOOST = 1.15
LOW_VALUE_PENALTY = 0.75

# Query-type keyword sets (used by reranker.py and llm/prompts.py)
PROCESS_KEYWORDS = {"explain", "how", "process", "mechanism", "describe", "what is", "what are"}
EFFECT_KEYWORDS = {"effect", "affect", "impact", "influence", "why", "reason", "cause"}
COMPARE_KEYWORDS = {"compare", "difference", "versus", "vs", "better", "worse"}
EXPLANATORY_QUERY_KEYWORDS = {
    "explain", "what is", "what are", "how does", "describe",
    "define", "overview", "process", "mechanism",
}
